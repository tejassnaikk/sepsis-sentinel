"""
src/streaming/kafka_producer.py

MIMIC-IV ICU event replay producer for SepsisSentinel.

Reads historical chartevents and labevents, replays them to Kafka topics in
chronological order with configurable time compression. Downstream Kafka
consumers receive events in the same sequence they occurred in the ICU but
compressed into a fraction of real time — suitable for end-to-end pipeline
testing without waiting days for events to arrive.

Topics:
  icu-vitals   — vital sign events (chartevents filtered to VITAL_ITEMIDS)
  icu-labs     — laboratory events (labevents filtered to LAB_ITEMIDS)

Run:
    python -m src.streaming.kafka_producer
"""

import json
import time
from pathlib import Path
from typing import Sequence

import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KAFKA_BOOTSTRAP = "localhost:9092"
VITALS_TOPIC    = "icu-vitals"
LABS_TOPIC      = "icu-labs"

DATA_DIR        = Path("/Volumes/Tejas SSD/sepsis-sentinel/data/raw")
CHART_PATH      = DATA_DIR / "chartevents.csv.gz"
LAB_PATH        = DATA_DIR / "labevents.csv.gz"
ICU_PATH        = DATA_DIR / "icustays.csv.gz"

CHUNK_SIZE      = 500_000   # rows per chartevents read chunk (matches mimic_loader)
LOG_EVERY       = 1_000     # print progress every N messages published

# Vital-sign itemids — must match the set in mimic_loader.load_vitals()
VITAL_ITEMIDS: set[int] = {
    220045,  # Heart Rate
    220179,  # Systolic BP
    220210,  # Respiratory Rate
    220277,  # SpO2
    223761,  # Temperature (°F)
    223900,  # GCS — Total
}

# Lab itemids — must match the set in mimic_loader.load_labs()
LAB_ITEMIDS: set[int] = {
    50912,   # Creatinine
    50813,   # Lactate
    51301,   # WBC
    50885,   # Bilirubin — Total
}


# ---------------------------------------------------------------------------
# Producer factory
# ---------------------------------------------------------------------------

def _make_producer() -> KafkaProducer:
    """
    Create a KafkaProducer that serialises message values as UTF-8 JSON.

    value_serializer encodes every message at send time, so the caller only
    ever passes plain dicts — no manual json.dumps() needed at the call site.
    """
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        # Batch up to 16 KB before sending — reduces round trips when replaying
        # dense event series without meaningfully increasing latency in this
        # non-real-time replay context.
        batch_size=16_384,
        linger_ms=5,
    )


# ---------------------------------------------------------------------------
# Replay helpers
# ---------------------------------------------------------------------------

def _publish_events(
    producer: KafkaProducer,
    topic: str,
    events: pd.DataFrame,
    event_type: str,
    speed_multiplier: float,
) -> int:
    """
    Publish a time-ordered DataFrame of events to a Kafka topic, sleeping
    between messages to simulate compressed real-time event arrival.

    Speed multiplier logic
    ----------------------
    In the real ICU, if two measurements are taken 60 minutes apart the
    producer should wait 60 / speed_multiplier seconds before emitting the
    second event. With speed_multiplier=100 that is 0.6 seconds per real
    ICU hour, compressing a 24-hour stay into ~14 minutes of wall time.

    Formally:
        simulated_delay_seconds = real_time_delta_seconds / speed_multiplier

    This preserves the *relative* timing of all events (a rapid sequence
    stays rapid; a quiet overnight period stays longer) while scaling the
    absolute durations down uniformly.

    Parameters
    ----------
    producer         : KafkaProducer
    topic            : Kafka topic name
    events           : Time-ordered DataFrame with columns:
                       stay_id, charttime, itemid, valuenum
    event_type       : "vital" or "lab" — included in every message
    speed_multiplier : Real ICU seconds per 1 simulated second.
                       100 → 100x faster than real time.

    Returns
    -------
    Number of messages successfully sent.
    """
    sent      = 0
    prev_time = None   # charttime of the previous event; None before the first

    for _, row in events.iterrows():
        current_time = row["charttime"]

        # -- Timing ----------------------------------------------------------
        if prev_time is not None:
            real_delta_seconds = (current_time - prev_time).total_seconds()
            # Clamp to 0 — floating-point or NaT edge cases can produce tiny
            # negatives after sorting. A negative sleep is nonsensical.
            simulated_delay = max(0.0, real_delta_seconds / speed_multiplier)
            if simulated_delay > 0:
                time.sleep(simulated_delay)

        prev_time = current_time

        # -- Message ---------------------------------------------------------
        message = {
            "stay_id":   int(row["stay_id"]),
            "charttime": current_time.isoformat(),
            "itemid":    int(row["itemid"]),
            "valuenum":  float(row["valuenum"]),
            "event_type": event_type,
        }

        # -- Publish ---------------------------------------------------------
        try:
            producer.send(topic, value=message)
            sent += 1
        except KafkaError as exc:
            # Log and continue — a single failed send should not abort the
            # replay. The downstream consumer is designed to tolerate gaps.
            print(f"  KafkaError publishing to {topic}: {exc}")
            continue

        if sent % LOG_EVERY == 0:
            print(f"  [{event_type}] {sent:,} messages sent  "
                  f"(simulated time: {current_time})")

    # Flush any remaining buffered messages before returning.
    producer.flush()
    return sent


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def replay_vitals(
    stay_ids: Sequence[int],
    speed_multiplier: float = 100.0,
    parquet_path: str | Path | None = None,
) -> int:
    """
    Replay chartevents vital signs for the given ICU stays to the icu-vitals
    Kafka topic.

    When parquet_path is provided the pre-filtered parquet file is loaded
    directly — avoiding the multi-minute scan of the 3 GB+ chartevents CSV.
    When parquet_path is None the original chunked CSV scan is used as fallback.

    Parameters
    ----------
    stay_ids         : ICU stay IDs to replay.
    speed_multiplier : Time compression factor.
                       100 → 100 real ICU hours compressed into 1 real hour.
    parquet_path     : Optional path to a pre-filtered vitals parquet file.
                       Expected columns: stay_id, charttime, itemid, valuenum.

    Returns
    -------
    Total number of messages successfully published.
    """
    stay_id_set = set(stay_ids)

    if parquet_path is not None:
        print(f"Loading vitals from parquet: {parquet_path}")
        vitals = pd.read_parquet(parquet_path)
        vitals["charttime"] = pd.to_datetime(vitals["charttime"])
        vitals = (
            vitals[vitals["stay_id"].isin(stay_id_set) & vitals["itemid"].isin(VITAL_ITEMIDS)]
            .dropna(subset=["valuenum"])
            .sort_values("charttime")
            .reset_index(drop=True)
        )
    else:
        chunks: list[pd.DataFrame] = []

        print(f"Scanning {CHART_PATH.name} for {len(stay_id_set):,} stays "
              f"({len(VITAL_ITEMIDS)} vital itemids)...")

        reader = pd.read_csv(
            CHART_PATH,
            usecols=["stay_id", "charttime", "itemid", "valuenum"],
            parse_dates=["charttime"],
            chunksize=CHUNK_SIZE,
        )
        for i, chunk in enumerate(reader, start=1):
            mask     = chunk["stay_id"].isin(stay_id_set) & chunk["itemid"].isin(VITAL_ITEMIDS)
            filtered = chunk.loc[mask].copy()
            if len(filtered):
                chunks.append(filtered)
            if i % 10 == 0:
                print(f"  ...scanned {i * CHUNK_SIZE:,} chartevents rows")

        if not chunks:
            print("No vitals found for the given stay_ids.")
            return 0

        vitals = (
            pd.concat(chunks, ignore_index=True)
            .dropna(subset=["valuenum"])
            .sort_values("charttime")
            .reset_index(drop=True)
        )

    if vitals.empty:
        print("No vitals found for the given stay_ids.")
        return 0

    print(f"Loaded {len(vitals):,} vital events across {vitals['stay_id'].nunique():,} stays.")

    producer = _make_producer()
    try:
        total = _publish_events(producer, VITALS_TOPIC, vitals, "vital", speed_multiplier)
    finally:
        producer.close()

    return total


def replay_labs(
    stay_ids: Sequence[int],
    speed_multiplier: float = 100.0,
    parquet_path: str | Path | None = None,
) -> int:
    """
    Replay labevents laboratory results for the given ICU stays to the
    icu-labs Kafka topic.

    When parquet_path is provided the pre-filtered parquet file (which must
    already contain stay_id) is loaded directly. When parquet_path is None
    the original CSV path — which resolves stay_ids to hadm_ids via
    icustays.csv.gz before filtering labevents — is used as fallback.

    Parameters
    ----------
    stay_ids         : ICU stay IDs to replay.
    speed_multiplier : Time compression factor (same semantics as replay_vitals).
    parquet_path     : Optional path to a pre-filtered labs parquet file.
                       Expected columns: stay_id, charttime, itemid, valuenum.

    Returns
    -------
    Total number of messages successfully published.
    """
    stay_id_set = set(stay_ids)

    if parquet_path is not None:
        print(f"Loading labs from parquet: {parquet_path}")
        labs = pd.read_parquet(parquet_path)
        labs["charttime"] = pd.to_datetime(labs["charttime"])
        labs = (
            labs[labs["stay_id"].isin(stay_id_set) & labs["itemid"].isin(LAB_ITEMIDS)]
            .dropna(subset=["valuenum"])
            .sort_values("charttime")
            .reset_index(drop=True)
        )
    else:
        # -- Resolve stay_ids → hadm_ids -------------------------------------
        stay_to_hadm = pd.read_csv(ICU_PATH, usecols=["stay_id", "hadm_id"])
        mapping = stay_to_hadm[stay_to_hadm["stay_id"].isin(stay_id_set)].copy()
        hadm_id_set = set(mapping["hadm_id"])

        if not hadm_id_set:
            print("No hadm_ids found for the given stay_ids.")
            return 0

        # -- Load and filter labevents ---------------------------------------
        print(f"Loading {LAB_PATH.name} for {len(hadm_id_set):,} admissions "
              f"({len(LAB_ITEMIDS)} lab itemids)...")

        labs = pd.read_csv(
            LAB_PATH,
            usecols=["hadm_id", "charttime", "itemid", "valuenum"],
            parse_dates=["charttime"],
        )

        labs = (
            labs[labs["hadm_id"].isin(hadm_id_set) & labs["itemid"].isin(LAB_ITEMIDS)]
            .copy()
            .dropna(subset=["valuenum"])
        )

        if labs.empty:
            print("No lab events found for the given stay_ids.")
            return 0

        # -- Join stay_id back onto labs -------------------------------------
        # The Kafka message schema requires stay_id. We merge the hadm→stay
        # mapping so consumers can correlate lab events with vital events by
        # stay_id without needing access to the icustays table.
        labs = labs.merge(mapping, on="hadm_id", how="left")

        labs = (
            labs.sort_values("charttime")
            .reset_index(drop=True)
        )

    if labs.empty:
        print("No lab events found for the given stay_ids.")
        return 0

    print(f"Loaded {len(labs):,} lab events across {labs['stay_id'].nunique():,} stays.")

    producer = _make_producer()
    try:
        total = _publish_events(producer, LABS_TOPIC, labs, "lab", speed_multiplier)
    finally:
        producer.close()

    return total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.data.mimic_loader import load_cohort

    cohort   = load_cohort(DATA_DIR)
    stay_ids = cohort["stay_id"].iloc[:100].tolist()

    VITALS_PARQUET = "/Volumes/Tejas SSD/sepsis-sentinel/data/vitals_filtered.parquet"

    print(f"\nReplaying vitals for first {len(stay_ids)} stays "
          f"(speed_multiplier=100000, parquet={VITALS_PARQUET})...\n")

    total = replay_vitals(
        stay_ids,
        speed_multiplier=100_000.0,
        parquet_path=VITALS_PARQUET,
    )

    print(f"\nDone. Total messages sent: {total:,}")
