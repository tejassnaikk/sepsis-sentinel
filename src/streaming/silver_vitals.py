"""
Silver-layer batch job for SepsisSentinel.

Reads the Bronze Delta vitals table, joins with ICU stay admission times,
computes 6-hour admission-anchored window aggregations for each vital itemid,
and writes the result to the Silver Delta layer.

Bronze → Silver transformation:
  - Raw event rows  →  one row per (stay_id, window_id)
  - 14k+ Bronze rows  →  N_stays × N_windows feature rows
  - Conditional aggregation (no pivot) guarantees deterministic column order

Run:
    python -m src.streaming.silver_vitals
"""

from __future__ import annotations

import os
from pathlib import Path

# Set before importing PySpark so the driver binds to loopback only
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

BRONZE_PATH     = _PROJECT_ROOT / "data" / "delta" / "bronze" / "vitals"
ICU_STAYS_PATH  = _PROJECT_ROOT / "data" / "raw" / "icustays.csv.gz"
SILVER_PATH     = _PROJECT_ROOT / "data" / "delta" / "silver" / "vitals_features"

VITAL_ITEMIDS: list[int] = [220045, 220179, 220210, 220277, 223761, 223900]

WINDOW_HOURS: int = 6


# ---------------------------------------------------------------------------
# Spark session
# ---------------------------------------------------------------------------


def create_spark_session() -> SparkSession:
    """Create a SparkSession configured for Delta Lake.

    Uses spark.jars.packages so the Delta connector is resolved by Ivy at
    startup — no separate configure_spark_with_delta_pip call needed.

    Returns
    -------
    Active SparkSession.
    """
    spark = (
        SparkSession.builder.appName("SepsisSentinel")
        .config(
            "spark.jars.packages",
            "io.delta:delta-spark_2.13:4.2.0",
        )
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        # Case-sensitive so column names like stay_id and stay_ID never silently
        # collide after a join introduces duplicate capitalisation variants.
        .config("spark.sql.caseSensitive", "true")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ---------------------------------------------------------------------------
# Main job
# ---------------------------------------------------------------------------


def build_silver_vitals() -> None:
    """Read Bronze vitals, aggregate into 6-hour windows, write Silver layer.

    Raises
    ------
    FileNotFoundError
        If the Bronze Delta table directory or icustays CSV is missing.
    """
    spark = create_spark_session()

    try:
        # ── Guard: fail fast with a clear message rather than a cryptic Spark error ──
        if not BRONZE_PATH.exists():
            raise FileNotFoundError(
                f"Bronze Delta table not found: {BRONZE_PATH}\n"
                "Run src/streaming/spark_consumer.py first to populate the Bronze layer."
            )
        if not ICU_STAYS_PATH.exists():
            raise FileNotFoundError(
                f"ICU stays file not found: {ICU_STAYS_PATH}\n"
                "Download MIMIC-IV v3.1 from PhysioNet and place it in data/raw/."
            )

        # ── 1. Load Bronze vitals ────────────────────────────────────────────
        print("Loading Bronze vitals...")
        bronze = (
            spark.read.format("delta")
            .load(str(BRONZE_PATH))
            .select("stay_id", "charttime", "itemid", "valuenum")
        )
        print(f"  Bronze rows: {bronze.count():,}")

        # ── 2. Load ICU stays ────────────────────────────────────────────────
        print("Loading ICU stays...")
        icustays = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(str(ICU_STAYS_PATH))
            .select("stay_id", "intime")
        )

        # ── 3. Join and compute admission-anchored windows ───────────────────
        print("Joining and computing windows...")
        joined = bronze.join(icustays, on="stay_id", how="inner")

        windowed = (
            joined
            # Parse string timestamps to proper timestamp type.
            # Bronze charttime is ISO-8601; icustays intime is YYYY-MM-DD HH:MM:SS.
            .withColumn("charttime_ts", F.to_timestamp("charttime"))
            .withColumn("intime_ts",    F.to_timestamp("intime"))
            # hours_elapsed: float hours since ICU admission
            .withColumn(
                "hours_elapsed",
                (F.col("charttime_ts").cast("long") - F.col("intime_ts").cast("long"))
                / 3600.0,
            )
            # Drop measurements recorded before ICU admission (negative offset)
            .filter(F.col("hours_elapsed") >= 0)
            # Assign each measurement to its non-overlapping 6-hour window.
            # floor(elapsed / 6) produces 0-based indices matching compute_features().
            .withColumn(
                "window_id",
                F.floor(F.col("hours_elapsed") / WINDOW_HOURS).cast(IntegerType()),
            )
        )

        # ── 4. Aggregate features ────────────────────────────────────────────
        print("Aggregating features...")

        # Build conditional aggregation expressions — one set per vital itemid.
        # Using F.when() instead of Spark's pivot() guarantees that column names
        # and order are determined by VITAL_ITEMIDS, not by the data distribution.
        agg_exprs = []
        for iid in VITAL_ITEMIDS:
            # Mask: selects only rows belonging to this vital itemid;
            # F.when returns null for other itemids, which all aggregate functions ignore.
            val = F.when(F.col("itemid") == iid, F.col("valuenum"))

            agg_exprs += [
                F.mean(val).alias(f"vital_{iid}_mean"),
                F.min(val).alias(f"vital_{iid}_min"),
                F.max(val).alias(f"vital_{iid}_max"),
                F.stddev(val).alias(f"vital_{iid}_std"),
                # missing: 1 when no non-null readings exist for this itemid in the window
                F.when(
                    F.count(F.when(F.col("itemid") == iid, F.col("valuenum"))) == 0,
                    F.lit(1),
                )
                .otherwise(F.lit(0))
                .alias(f"vital_{iid}_missing"),
            ]

        # Include intime_ts in groupBy so it is available for window_end arithmetic
        # below — safe because intime_ts is functionally dependent on stay_id.
        agg_df = windowed.groupBy("stay_id", "window_id", "intime_ts").agg(*agg_exprs)

        # Compute window_end = intime + (window_id + 1) * 6 hours.
        # Cast to epoch seconds for arithmetic, then back to timestamp.
        silver = (
            agg_df
            .withColumn(
                "window_end",
                (
                    F.col("intime_ts").cast("long")
                    + (F.col("window_id") + 1) * WINDOW_HOURS * 3600
                ).cast("timestamp"),
            )
            # Partition column: date of window_end enables efficient date-range queries
            .withColumn("window_date", F.to_date("window_end"))
            # Enforce the required final column order
            .select(
                "stay_id",
                "window_id",
                "window_end",
                *[
                    f"vital_{iid}_{stat}"
                    for iid in VITAL_ITEMIDS
                    for stat in ("mean", "min", "max", "std", "missing")
                ],
                "window_date",
            )
        )

        # ── 5. Write Silver layer ────────────────────────────────────────────
        print("Writing Silver layer...")
        SILVER_PATH.mkdir(parents=True, exist_ok=True)

        (
            silver.write
            .format("delta")
            .mode("overwrite")
            # Allow schema changes on re-runs without requiring a manual TRUNCATE
            .option("overwriteSchema", "true")
            .partitionBy("window_date")
            .save(str(SILVER_PATH))
        )

        # ── 6. Verification readback ─────────────────────────────────────────
        result = spark.read.format("delta").load(str(SILVER_PATH))
        n = result.count()
        print(f"Silver layer complete. Row count: {n:,}")
        print("\nSchema:")
        result.printSchema()
        print("First 5 rows:")
        result.show(5, truncate=False)

    finally:
        # Always release Spark resources, even if an exception was raised
        spark.stop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_silver_vitals()
