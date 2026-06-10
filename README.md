# SepsisSentinel

End-to-end real-time ICU sepsis prediction system trained on 65,366 MIMIC-IV patients.

## 🚀 Live API
**Base URL:** https://sepsis-sentinel-production-29ff.up.railway.app

| Endpoint | Method | Description |
|----------|--------|-------------|
| /health | GET | Service + model status |
| /predict | POST | Sepsis risk score from ICU vitals |
| /features | GET | Expected feature names |

**Quick test:**
```bash
curl -s https://sepsis-sentinel-production-29ff.up.railway.app/health
```

## Results
| Metric | Value |
|--------|-------|
| AUROC | 0.806 |
| AUPRC | 0.609 |
| Recall @ t=0.35 | 0.878 |
| Training windows | 889,262 |
| ICU patients | 65,366 |
| Sepsis prevalence | 14.4% |

## Stack
Kafka · PySpark · Delta Lake · XGBoost · FastAPI · Railway

## Architecture
MIMIC-IV → Kafka Producer → PySpark Structured Streaming → Delta Lake → XGBoost → FastAPI

## Quickstart
```bash
git clone https://github.com/tejassnaikk/sepsis-sentinel.git
cd sepsis-sentinel
conda create -n sepsis-sentinel python=3.11 -y
conda activate sepsis-sentinel
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

## Project Structure
src/
data/         # MIMIC-IV loaders (mimic_loader.py)
features/     # Feature engineering (feature_engineering.py)
models/       # XGBoost training + evaluation (train.py)
api/          # FastAPI endpoint (main.py)
streaming/    # Kafka + PySpark pipeline (Week 2)
notebooks/      # EDA (notebooks01eda.ipynb)
reports/        # metrics.md, figures/
tests/          # pytest unit tests (5/5 passing)
models_local/   # Trained model weights
configs/        # paths.yaml

## Dataset
MIMIC-IV v3.1 — PhysioNet credentialed access required.
Data not included in repo.
Reference: Johnson et al. (2023). Scientific Data.

## Clinical Disclaimer
Research tool only. Not for clinical decision-making.

## License
MIT
