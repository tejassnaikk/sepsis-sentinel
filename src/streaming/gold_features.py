"""
Gold-layer batch job for SepsisSentinel.

Joins Silver windowed vitals, windowed lab features, and stay-level sepsis
labels into the final model-ready Gold feature table.

Pipeline:
    Silver vitals (Delta)
        ├── left join lab windows (parquet → windowed aggregation)
        └── left join sepsis labels (parquet, stay-level)
    → Gold Delta table with exactly 43 columns

Note on labs: labs_filtered.parquet is keyed by hadm_id, not stay_id.
The join to ICU stays uses hadm_id (not stay_id as the spec nominally states)
because that is the foreign key present in the labs file.

Run:
    python -m src.streaming.gold_features
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAB_ITEMIDS:  list[int] = [50912, 50813, 51301, 50885]
VITAL_ITEMIDS: list[int] = [220045, 220179, 220210, 220277, 223761, 223900]

SILVER_PATH   = Path("/Volumes/Tejas SSD/sepsis-sentinel/data/delta/silver/vitals_features")
LABS_PATH     = Path("/Volumes/Tejas SSD/sepsis-sentinel/data/labs_filtered.parquet")
FEATURES_PATH = Path("/Volumes/Tejas SSD/sepsis-sentinel/data/features_v1.parquet")
GOLD_PATH     = Path("/Volumes/Tejas SSD/sepsis-sentinel/data/delta/gold/features")
_ICU_PATH     = Path("/Volumes/Tejas SSD/sepsis-sentinel/data/raw/icustays.csv.gz")

WINDOW_HOURS: int = 6

# Exact final column order — must produce 43 columns and match training schema
_FINAL_COLUMNS: list[str] = (
    ["stay_id", "window_end"]
    + [
        f"vital_{iid}_{s}"
        for iid in VITAL_ITEMIDS
        for s in ("mean", "min", "max", "std", "missing")
    ]
    + [
        col
        for iid in LAB_ITEMIDS
        for col in (f"lab_{iid}_last", f"lab_{iid}_missing")
    ]
    + ["icu_hours_elapsed", "time_of_day", "sepsis_label"]
)

assert len(_FINAL_COLUMNS) == 43, f"Schema misconfigured: {len(_FINAL_COLUMNS)} columns"


# ---------------------------------------------------------------------------
# Spark session
# ---------------------------------------------------------------------------


def create_spark_session() -> SparkSession:
    """Create a SparkSession configured for Delta Lake reads and writes.

    Returns
    -------
    Active SparkSession.
    """
    spark = (
        SparkSession.builder.appName("SepsisSentinel")
        .config("spark.jars.packages", "io.delta:delta-spark_2.13:4.2.0")
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.caseSensitive", "true")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ---------------------------------------------------------------------------
# Main job
# ---------------------------------------------------------------------------


def build_gold_features() -> None:
    """Build the Gold feature table by joining Silver vitals, lab windows, and labels.

    Raises
    ------
    FileNotFoundError
        If Silver Delta path, labs parquet, or features parquet are missing.
    """
    spark = create_spark_session()

    try:
        # ── Guard: clear errors before any Spark plan is built ──────────────
        for path, label in [
            (SILVER_PATH,   "Silver Delta table"),
            (LABS_PATH,     "labs_filtered.parquet"),
            (FEATURES_PATH, "features_v1.parquet"),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"{label} not found: {path}\n"
                    "Ensure the upstream pipeline has run before building Gold."
                )

        # ── Step 1 — Load Silver ─────────────────────────────────────────────
        print("Loading Silver vitals...")
        vital_cols = [
            f"vital_{iid}_{s}"
            for iid in VITAL_ITEMIDS
            for s in ("mean", "min", "max", "std", "missing")
        ]
        silver = spark.read.format("delta").load(str(SILVER_PATH)).select(
            "stay_id", "window_id", "window_end", *vital_cols
        )
        n_silver = silver.count()
        print(f"Loading Silver vitals... {n_silver:,} rows")

        # ── Step 2 — Load and window labs ────────────────────────────────────
        print("Loading and windowing labs...")

        # labs_filtered.parquet is keyed by hadm_id; icustays maps hadm_id → stay_id
        # and provides intime for admission-anchored windowing.
        icustays = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(str(_ICU_PATH))
            .select("stay_id", "hadm_id", "intime")
        )

        labs_raw = (
            spark.read.parquet(str(LABS_PATH))
            .select("hadm_id", "charttime", "itemid", "valuenum")
            .filter(F.col("itemid").isin(LAB_ITEMIDS))
        )

        # Join on hadm_id (the key present in the labs file) to get stay_id + intime
        labs_joined = labs_raw.join(icustays, on="hadm_id", how="inner")

        labs_windowed = (
            labs_joined
            # labs charttime is TIMESTAMP_NTZ, which cannot be cast directly to
            # BIGINT in Spark 3.x.  Routing through STRING → TIMESTAMP gives a
            # regular (timezone-aware) timestamp that can be cast to epoch seconds.
            .withColumn(
                "hours_elapsed",
                (
                    F.col("charttime").cast("string").cast("timestamp").cast("long")
                    - F.to_timestamp("intime").cast("long")
                )
                / 3600.0,
            )
            .filter(F.col("hours_elapsed") >= 0)
            .withColumn(
                "window_id",
                F.floor(F.col("hours_elapsed") / WINDOW_HOURS).cast(IntegerType()),
            )
            # Sort ascending so F.last(ignorenulls=True) returns the most recent value
            .orderBy("charttime")
        )

        # Conditional aggregation — no pivot() so column order is deterministic.
        lab_agg_exprs = []
        for iid in LAB_ITEMIDS:
            cond     = F.col("itemid") == iid
            val_when = F.when(cond, F.col("valuenum"))

            lab_agg_exprs += [
                # Most recent non-null value for this itemid within the window
                F.last(val_when, ignorenulls=True).alias(f"lab_{iid}_last"),
                # 1 if no readings for this itemid exist in the window
                F.when(
                    F.count(F.when(cond, F.col("valuenum"))) == 0, F.lit(1)
                )
                .otherwise(F.lit(0))
                .alias(f"lab_{iid}_missing"),
            ]

        lab_windows = labs_windowed.groupBy("stay_id", "window_id").agg(*lab_agg_exprs)
        n_labs = lab_windows.count()
        print(f"Lab windows computed... {n_labs:,} rows")

        # ── Step 3 — Extract sepsis labels ───────────────────────────────────
        print("Loading sepsis labels...")
        labels = (
            spark.read.parquet(str(FEATURES_PATH))
            .select("stay_id", "sepsis_label")
            .dropDuplicates(["stay_id"])
        )
        n_labels = labels.count()
        print(f"Sepsis labels loaded... {n_labels:,} stays")

        # ── Step 4 — Join everything ─────────────────────────────────────────
        print("Joining all layers...")

        joined = (
            silver
            .join(lab_windows, on=["stay_id", "window_id"], how="left")
            .join(labels,      on="stay_id",                how="left")
        )

        # Null lab_*_missing means the stay had no lab events at all — treat as missing
        lab_missing_fills = {f"lab_{iid}_missing": 1 for iid in LAB_ITEMIDS}
        joined = joined.fillna(lab_missing_fills)

        gold_raw = (
            joined
            # Midpoint of the window in hours since ICU admission
            .withColumn(
                "icu_hours_elapsed",
                (F.col("window_id") * WINDOW_HOURS + WINDOW_HOURS / 2.0).cast("double"),
            )
            # Hour of day at window end, matching the training feature schema
            .withColumn("time_of_day", F.hour("window_end").cast(IntegerType()))
            # Stays not in features_v1 get label 0; cast byte → integer per spec
            .withColumn(
                "sepsis_label",
                F.coalesce(F.col("sepsis_label").cast(IntegerType()), F.lit(0)),
            )
        )

        # ── Step 5 — Select final 43 columns in exact order ──────────────────
        gold = gold_raw.select(*_FINAL_COLUMNS)
        assert len(gold.columns) == 43, (
            f"Expected 43 columns, got {len(gold.columns)}: {gold.columns}"
        )

        # ── Step 6 — Write Gold ───────────────────────────────────────────────
        print("Writing Gold layer...")
        GOLD_PATH.mkdir(parents=True, exist_ok=True)

        (
            gold
            # window_date is a partition column only — not part of the 43-column schema
            .withColumn("window_date", F.to_date("window_end"))
            .write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .partitionBy("window_date")
            .save(str(GOLD_PATH))
        )

        # ── Readback and verification ─────────────────────────────────────────
        result = spark.read.format("delta").load(str(GOLD_PATH))
        n      = result.count()
        c      = len(result.columns)
        print(f"Gold layer complete. Rows: {n:,}, Columns: {c}")

        print("\nSchema:")
        result.printSchema()

        print("\nSepsis label distribution:")
        result.groupBy("sepsis_label").count().show()

        print("First 3 rows (key columns):")
        result.select(
            "stay_id", "window_end",
            "vital_220045_mean", "lab_50912_last",
            "icu_hours_elapsed", "time_of_day", "sepsis_label",
        ).show(3, truncate=False)

    finally:
        spark.stop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_gold_features()
