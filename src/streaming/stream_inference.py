"""
Batch inference job for SepsisSentinel.

Reads the Gold Delta feature table, scores each row with the local XGBoost
model via a Pandas UDF, and writes predictions (sepsis_prob, sepsis_alert,
predicted_at) to a Delta predictions table.

Pipeline:
    Gold Delta (43 cols)
        → XGBoost Pandas UDF  (40 feature cols)
        → predictions Delta   (Gold cols + sepsis_prob + sepsis_alert + predicted_at)

Run:
    python -m src.streaming.stream_inference
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType, IntegerType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GOLD_PATH        = Path("/Volumes/Tejas SSD/sepsis-sentinel/data/delta/gold/features")
MODEL_PATH       = Path("/Volumes/Tejas SSD/sepsis-sentinel/models_local/xgboost_v1.json")
PREDICTIONS_PATH = Path("/Volumes/Tejas SSD/sepsis-sentinel/data/delta/predictions")

ALERT_THRESHOLD: float = 0.35

FEATURE_COLS: list[str] = [
    "vital_220045_mean", "vital_220045_min", "vital_220045_max", "vital_220045_std", "vital_220045_missing",
    "vital_220179_mean", "vital_220179_min", "vital_220179_max", "vital_220179_std", "vital_220179_missing",
    "vital_220210_mean", "vital_220210_min", "vital_220210_max", "vital_220210_std", "vital_220210_missing",
    "vital_220277_mean", "vital_220277_min", "vital_220277_max", "vital_220277_std", "vital_220277_missing",
    "vital_223761_mean", "vital_223761_min", "vital_223761_max", "vital_223761_std", "vital_223761_missing",
    "vital_223900_mean", "vital_223900_min", "vital_223900_max", "vital_223900_std", "vital_223900_missing",
    "lab_50912_last", "lab_50912_missing",
    "lab_50813_last", "lab_50813_missing",
    "lab_51301_last", "lab_51301_missing",
    "lab_50885_last", "lab_50885_missing",
    "icu_hours_elapsed", "time_of_day",
]
# 40 feature columns — sepsis_label is the target, not an input feature

assert len(FEATURE_COLS) == 40, f"FEATURE_COLS misconfigured: {len(FEATURE_COLS)} cols"


# ---------------------------------------------------------------------------
# Spark session
# ---------------------------------------------------------------------------


def create_spark_session() -> SparkSession:
    spark = (
        SparkSession.builder.appName("SepsisSentinel-Inference")
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
# Pandas UDF — XGBoost scoring
# ---------------------------------------------------------------------------

# Worker-side singleton: loaded once per executor JVM, reused across batches.
_model_cache: dict = {}


@pandas_udf(DoubleType())
def score_batch(*feature_series: pd.Series) -> pd.Series:
    if "model" not in _model_cache:
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(str(MODEL_PATH))
        _model_cache["model"] = model

    model = _model_cache["model"]

    import numpy as np  # noqa: F401 — used implicitly by XGBoost
    feat_df = pd.concat(list(feature_series), axis=1)
    feat_df.columns = FEATURE_COLS

    # XGBoost handles NaN natively; cast ensures no object-dtype columns sneak through
    feat_df = feat_df.astype(float)

    probs = model.predict_proba(feat_df)[:, 1]
    return pd.Series(probs)


# ---------------------------------------------------------------------------
# Main job
# ---------------------------------------------------------------------------


def run_stream_inference() -> None:
    """Score Gold Delta rows with the local XGBoost model and write predictions.

    Raises
    ------
    FileNotFoundError
        If GOLD_PATH or MODEL_PATH are not found on disk before the job starts.
    ValueError
        If the Gold table is missing expected feature columns.
    """
    if not GOLD_PATH.exists():
        raise FileNotFoundError(
            f"Gold Delta table not found: {GOLD_PATH}\n"
            "Run src/streaming/gold_features.py first."
        )
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"XGBoost model not found: {MODEL_PATH}\n"
            "Run src/models/train.py to train and save the model."
        )

    spark = create_spark_session()

    try:
        # ── Step 1 — Load Gold ────────────────────────────────────────────────
        print("Loading Gold features...")
        gold = spark.read.format("delta").load(str(GOLD_PATH))
        n_gold = gold.count()
        print(f"Loading Gold features... {n_gold:,} rows")

        missing_cols = [c for c in FEATURE_COLS if c not in gold.columns]
        if missing_cols:
            raise ValueError(
                f"Gold table is missing {len(missing_cols)} expected feature column(s):\n"
                + "  " + "\n  ".join(missing_cols)
            )

        # ── Step 2 already defined at module level (score_batch UDF) ─────────

        # ── Step 3 — Score Gold rows ──────────────────────────────────────────
        print("Applying XGBoost scoring UDF...")
        predictions = gold.withColumn(
            "sepsis_prob",
            score_batch(*[F.col(c) for c in FEATURE_COLS]),
        )
        predictions = predictions.withColumn(
            "sepsis_alert",
            F.when(F.col("sepsis_prob") >= ALERT_THRESHOLD, 1)
            .otherwise(0)
            .cast(IntegerType()),
        )
        predictions = predictions.withColumn("predicted_at", F.current_timestamp())
        print("Scoring complete.")

        # ── Step 4 — Validate predictions ────────────────────────────────────
        print("Validating predictions...")

        # Cache so the four agg calls don't re-trigger scoring
        predictions.cache()

        n_predictions = predictions.count()
        check1 = n_predictions == n_gold
        print(f"  [1] Row count matches Gold: {check1}  ({n_predictions:,} rows)")

        prob_stats = predictions.agg(
            F.min("sepsis_prob").alias("prob_min"),
            F.max("sepsis_prob").alias("prob_max"),
        ).collect()[0]
        prob_min, prob_max = prob_stats["prob_min"], prob_stats["prob_max"]
        check2 = (prob_min >= 0.0) and (prob_max <= 1.0)
        print(f"  [2] sepsis_prob in [0,1]:   {check2}  (min={prob_min:.4f}, max={prob_max:.4f})")

        alert_vals = {r[0] for r in predictions.select("sepsis_alert").distinct().collect()}
        check3 = alert_vals <= {0, 1}
        print(f"  [3] sepsis_alert in {{0,1}}:  {check3}  (distinct values: {sorted(alert_vals)})")

        n_alerts = predictions.filter(F.col("sepsis_alert") == 1).count()
        alert_rate = n_alerts / n_predictions if n_predictions else 0.0
        check4 = 0.05 <= alert_rate <= 0.95
        print(f"  [4] Alert rate reasonable:  {check4}  ({alert_rate:.1%}, {n_alerts:,} alerts)")

        if not all([check1, check2, check3, check4]):
            print("WARNING: one or more validation checks failed — review output before use.")

        # ── Step 5 — Write predictions ────────────────────────────────────────
        print("Writing predictions Delta table...")
        PREDICTIONS_PATH.mkdir(parents=True, exist_ok=True)

        (
            predictions.write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .save(str(PREDICTIONS_PATH))
        )

        # Release cache after the write materialises it
        predictions.unpersist()

        # Readback and summary
        result = spark.read.format("delta").load(str(PREDICTIONS_PATH))
        n_result = result.count()
        n_result_alerts = result.filter(F.col("sepsis_alert") == 1).count()
        print(f"Predictions complete. Rows: {n_result:,}, Alerts: {n_result_alerts:,}")

        print("\nAlert distribution:")
        result.groupBy("sepsis_alert").count().orderBy("sepsis_alert").show()

        print("Confusion-style summary (sepsis_label vs sepsis_alert):")
        result.groupBy("sepsis_label", "sepsis_alert").count().orderBy(
            "sepsis_label", "sepsis_alert"
        ).show()

        print("Top 10 highest-risk windows:")
        result.select(
            "stay_id", "window_end", "sepsis_prob", "sepsis_alert", "sepsis_label"
        ).orderBy(F.desc("sepsis_prob")).show(10, truncate=False)

    finally:
        spark.stop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_stream_inference()
