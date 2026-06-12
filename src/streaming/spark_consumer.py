"""
PySpark Structured Streaming consumer for SepsisSentinel.

Reads ICU vital-sign events from the icu-vitals Kafka topic and writes them
to a Delta Lake Bronze table for downstream feature engineering.

Bronze layer role: land raw events exactly as produced, with an ingestion
timestamp added but no transformations.  Cleaning and feature extraction
happen in the Silver layer (feature_engineering.py).

Run:
    python -m src.streaming.spark_consumer
"""

from __future__ import annotations

from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.streaming import StreamingQuery
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KAFKA_BOOTSTRAP = "localhost:9092"
VITALS_TOPIC    = "icu-vitals"

# Kafka connector coordinates must match the installed Spark's Scala binary
# version.  Spark 4.x uses Scala 2.13.
_KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1"

# Expected JSON schema for messages on the icu-vitals topic.
# Produced by src/streaming/kafka_producer.py.
_VITALS_SCHEMA = StructType(
    [
        StructField("stay_id",    LongType(),    nullable=True),
        StructField("charttime",  StringType(),  nullable=True),
        StructField("itemid",     IntegerType(), nullable=True),
        StructField("valuenum",   DoubleType(),  nullable=True),
        StructField("event_type", StringType(),  nullable=True),
    ]
)


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------


def create_spark_session() -> SparkSession:
    """Create and return a SparkSession configured for Delta Lake and Kafka.

    Delta Lake is wired in via configure_spark_with_delta_pip which injects
    the required catalog extensions and downloads the Delta jars if they are
    not already cached in ~/.ivy2.  The Kafka connector is passed as an extra
    package so both sets of jars are resolved in one Ivy resolution pass.

    Returns
    -------
    Active SparkSession with Delta Lake and Kafka support.
    """
    print("Initialising SparkSession (Delta Lake + Kafka)...")

    builder = (
        SparkSession.builder.appName("SepsisSentinel")
        # Delta Lake catalog extensions — required for reading/writing Delta tables
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )

    # configure_spark_with_delta_pip appends the Delta jar to spark.jars.packages
    # and passes extra_packages alongside it in one resolution call.
    spark = configure_spark_with_delta_pip(
        builder,
        extra_packages=[_KAFKA_PACKAGE],
    ).getOrCreate()

    # Suppress INFO-level Spark log spam; WARN keeps actionable messages visible
    spark.sparkContext.setLogLevel("WARN")

    print(f"SparkSession ready | Spark {spark.version}")
    return spark


# ---------------------------------------------------------------------------
# Stream definition
# ---------------------------------------------------------------------------


def start_vitals_stream(
    spark: SparkSession,
    output_path: str | Path,
    checkpoint_path: str | Path,
) -> StreamingQuery:
    """Start a structured stream that reads icu-vitals and lands data in Delta Lake.

    The stream runs in append mode — every micro-batch appends new rows to the
    Delta table without modifying existing ones, preserving the full event log.

    Checkpoint purpose
    ------------------
    The checkpoint directory stores the stream's progress (committed Kafka
    offsets) and the write-ahead log.  If the process crashes and restarts,
    Spark reads the checkpoint to resume exactly where it left off, guaranteeing
    exactly-once delivery to the Delta table without re-processing old events
    or dropping new ones.

    Parameters
    ----------
    spark:
        Active SparkSession from create_spark_session().
    output_path:
        Delta Lake table path for the Bronze vitals layer.
    checkpoint_path:
        Directory for Spark Structured Streaming checkpoint files.
        Must be writable and persistent across restarts.

    Returns
    -------
    Active StreamingQuery — call .awaitTermination() to block, or .stop() to
    shut down gracefully.
    """
    output_path     = str(output_path)
    checkpoint_path = str(checkpoint_path)

    print(f"Subscribing to Kafka topic '{VITALS_TOPIC}' @ {KAFKA_BOOTSTRAP}")
    print(f"Delta output      : {output_path}")
    print(f"Checkpoint location: {checkpoint_path}")

    # Read raw bytes from Kafka — each row has key, value, topic, partition,
    # offset, timestamp columns; we only need value (the JSON payload).
    raw_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", VITALS_TOPIC)
        # Start from the latest offset so the consumer does not replay the
        # full topic history on first run; the checkpoint takes over after that.
        .option("startingOffsets", "latest")
        .load()
    )

    # Decode the binary value column to a UTF-8 string, then parse as JSON
    parsed = (
        raw_stream
        .select(
            F.from_json(F.col("value").cast(StringType()), _VITALS_SCHEMA).alias("d")
        )
        .select("d.*")                                    # flatten nested struct to top-level columns
        .withColumn("ingestion_timestamp", F.current_timestamp())  # Bronze audit column
    )

    # Write to Delta Lake in append mode with fault-tolerant checkpointing
    query: StreamingQuery = (
        parsed.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", checkpoint_path)
        .start(output_path)
    )

    print("Streaming query started.")
    return query


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _OUTPUT_PATH     = Path("/Volumes/Tejas SSD/sepsis-sentinel/data/delta/bronze/vitals")
    _CHECKPOINT_PATH = Path("/Volumes/Tejas SSD/sepsis-sentinel/data/delta/checkpoints/vitals")

    spark = create_spark_session()
    query = start_vitals_stream(spark, _OUTPUT_PATH, _CHECKPOINT_PATH)

    print("Stream started — waiting for data...")
    query.awaitTermination()
