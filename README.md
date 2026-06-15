# SepsisSentinel

Real-time ICU sepsis prediction pipeline built on MIMIC-IV v3.1 (65,366 ICU
stays, 934,767 feature windows). Streams vital signs and lab events through
Kafka → Delta Lake → XGBoost, achieving AUROC 0.820 on held-out validation.

**Live API:** https://sepsis-sentinel-production-29ff.up.railway.app  
**Stack:** Kafka · PySpark · Delta Lake · DuckDB · XGBoost · FastAPI · Railway  
**Data:** MIMIC-IV v3.1 (PhysioNet credentialed access)

---

## Architecture

```
MIMIC-IV chartevents (23.8M rows)
│
▼
Kafka (icu-vitals topic)
│
▼
PySpark Structured Streaming → Delta Lake Bronze (29M rows)
│
▼
DuckDB Silver: 6h tumbling window aggregations
└─ 934,767 windows × 30 vital features (mean/min/max/std/missing)
│
▼
DuckDB Gold: join labs + sepsis labels
└─ 43-column model-ready feature table
│
▼
XGBoost v3: lag + delta features (63 total)
└─ AUROC 0.820 (val) · 0.810 (test) · Recall 86.4%
│
▼
FastAPI on Railway: /predict · /health · /features
```

---

## Model Performance

### Overall (934,767 windows, 65,366 ICU stays)

| Metric | v1 (baseline) | v3 (lag+delta) |
|--------|--------------|----------------|
| AUROC | 0.806 | **0.810** |
| AUPRC | 0.609 | 0.570 |
| Val AUROC | — | **0.820** |
| Recall@0.35 | 86.4% | — |
| Alert threshold | 0.35 | 0.49 |

### Subgroup Analysis

| Subgroup | N windows | Sepsis rate | AUROC | Recall |
|----------|-----------|-------------|-------|--------|
| Age 18–45 | 141,009 | 23.6% | 0.828 | 90.7% |
| Age 45–65 | 346,165 | 25.8% | 0.808 | 88.0% |
| Age 65–80 | 307,964 | 24.6% | 0.791 | 85.1% |
| Age 80+ | 139,629 | 22.8% | 0.763 | 80.7% |
| LOS Early (0–24h) | 247,387 | 14.6% | 0.810 | 67.6% |
| LOS Mid (24–72h) | 270,208 | 18.5% | 0.765 | 74.8% |
| LOS Extended (72h+) | 417,172 | 34.5% | 0.766 | 95.2% |
| Day shift | 471,797 | 24.5% | 0.799 | 86.4% |
| Night shift | 462,970 | 24.8% | 0.798 | 86.5% |

**Key findings:**
- Performance degrades monotonically with age — oldest patients (80+) are hardest to detect
- Early ICU windows (0–24h) have the lowest recall (67.6%) — a lower threshold is recommended for this period
- No circadian bias: day vs night shift performance is statistically identical

---

## Feature Engineering

**v1 (40 features):** Single-window aggregations for 6 vital itemids and 4 lab itemids  
**v3 (63 features):** Added lag-1 and delta (trend) features for all vital means and lab values

Top features by XGBoost importance:
1. `vital_223900_max` — GCS max (23.8%) — consciousness level is the dominant signal
2. `lab_50813_missing` — Lactate missing flag (10.0%) — absence of ordering is itself informative
3. `icu_hours_elapsed` — Time in ICU (5.0%)
4. `lab_50813_last_lag1` — Previous window lactate (3.7%) — temporal trend matters
5. `vital_223900_mean_lag1` — Previous window GCS (3.0%)

---

## Data Pipeline

### Vital sign itemids (6)

| itemid | Description |
|--------|-------------|
| 220045 | Heart Rate |
| 220179 | Systolic BP |
| 220210 | Respiratory Rate |
| 220277 | SpO2 |
| 223761 | Temperature (°F) |
| 223900 | GCS Total |

### Lab itemids (4)

| itemid | Description |
|--------|-------------|
| 50912 | Creatinine |
| 50813 | Lactate |
| 51301 | WBC |
| 50885 | Bilirubin Total |

### Sepsis label

ICD-10 A40/A41, ICD-9 99591/99592. Stay-level label applied to all windows of a stay.  
Cohort: 65,366 ICU stays · 14.4% sepsis rate (9,433 positive stays)

---

## Project Structure

```
sepsis-sentinel/
├── src/
│   ├── data/
│   │   └── mimic_loader.py          # load_cohort, load_vitals, load_labs
│   ├── features/
│   │   └── feature_engineering.py   # compute_features, 6h windowing
│   ├── models/
│   │   ├── train.py                 # v1 baseline training
│   │   ├── train_v3.py              # v3 lag+delta training
│   │   └── subgroup_analysis.py     # subgroup metrics
│   └── streaming/
│       ├── kafka_producer.py        # MIMIC-IV replay producer
│       ├── spark_consumer.py        # PySpark → Delta Lake Bronze
│       ├── silver_duckdb.py         # DuckDB Silver aggregation
│       ├── gold_duckdb.py           # DuckDB Gold feature join
│       ├── inference_direct.py      # Batch inference + AUROC
│       └── stream_inference.py      # PySpark Pandas UDF inference
├── models_local/
│   ├── xgboost_v1.json              # Baseline model (AUROC 0.806)
│   ├── xgboost_v3.json              # v3 model (Val AUROC 0.820)
│   └── feature_cols_v3.json         # 63-feature column list
├── reports/
│   ├── metrics.md                   # Full metrics history
│   └── subgroup_analysis.csv        # Subgroup breakdown table
├── configs/
│   └── paths.yaml
├── tests/
│   └── test_feature_engineering.py  # 5/5 passing
├── Procfile                         # Railway deployment
└── README.md
```

---

## Environment Setup

**Requirements:** Python 3.11, Java 17, Kafka (Docker), MIMIC-IV v3.1 access

```bash
# Clone and set up environment
git clone https://github.com/tejassnaikk/sepsis-sentinel.git
cd sepsis-sentinel
conda create -n sepsis-sentinel python=3.11
conda activate sepsis-sentinel
pip install -r requirements.txt

# Set Java 17
export JAVA_HOME=$(brew --prefix openjdk@17)

# Pin Python for PySpark workers
export PYSPARK_PYTHON=$(which python)
export PYSPARK_DRIVER_PYTHON=$(which python)
export SPARK_LOCAL_IP=127.0.0.1

# Start Kafka (requires Docker)
cd marketstream && docker compose up -d kafka

# Run full pipeline
python src/streaming/kafka_producer.py      # Replay vitals to Kafka
python src/streaming/spark_consumer.py      # Bronze layer
python src/streaming/silver_duckdb.py       # Silver layer
python src/streaming/gold_duckdb.py         # Gold layer
python src/models/train_v3.py               # Train model
python src/streaming/inference_direct.py    # Score predictions
```

---

## API Reference

**Base URL:** https://sepsis-sentinel-production-29ff.up.railway.app

```bash
# Health check
curl https://sepsis-sentinel-production-29ff.up.railway.app/health

# Predict sepsis probability for a single window
curl -X POST https://sepsis-sentinel-production-29ff.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"stay_id": 12345, "icu_hours_elapsed": 24.0, "time_of_day": 14, ...}'
```

---

## Technical Decisions

**Why DuckDB over PySpark for Silver/Gold?**  
PySpark's JVM ran OOM aggregating 23.8M rows on a MacBook Air M4 (16GB RAM) with 2703 output partitions. DuckDB runs the same GROUP BY in-process with streaming execution, completing in under 5 minutes vs repeated JVM crashes.

**Why lag+delta features?**  
Single-window aggregates hit a performance ceiling at AUROC ~0.80. Temporal trend features (lag-1 mean, delta) gave the model memory across windows — GCS trend and lactate history are the signals clinicians use. This pushed val AUROC from 0.806 to 0.820.

**Why threshold 0.35 for v1?**  
Youden's J statistic on the validation set. Clinical setting favors recall over precision — missing sepsis is worse than a false alert.

**Why temporal (not random) train/val/test split?**  
Random splitting on window level causes data leakage — windows from the same stay appear in train and test. Splitting by stay_id in chronological order simulates true deployment where the model sees future patients it was never trained on.

---

## Data Access

MIMIC-IV v3.1 requires PhysioNet credentialed access:  
https://physionet.org/content/mimiciv/3.1/

This project cannot redistribute MIMIC-IV data. All data files must be downloaded independently.

---

## Author

Tejas Naik · MS Data Science, CU Boulder (expected April 2027)  
[GitHub](https://github.com/tejassnaikk) · [LinkedIn](https://linkedin.com/in/tejassnaikk)

---

*Built as a portfolio project targeting data engineering and ML engineering roles in the Denver area.*
