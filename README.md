# SepsisSentinel

End-to-end real-time ICU sepsis prediction system using MIMIC-IV.

## Stack
Kafka · PySpark · Delta Lake · XGBoost · FastAPI

## Live API
Coming end of Week 1.

## Architecture
cd "/Volumes/Tejas SSD/sepsis-sentinel"

cat > README.md << 'EOF'
# SepsisSentinel

End-to-end real-time ICU sepsis prediction system using MIMIC-IV.

## Stack
Kafka · PySpark · Delta Lake · XGBoost · FastAPI

## Live API
Coming end of Week 1.

## Results
| Model | AUROC | AUPRC |
|-------|-------|-------|
| XGBoost v1 | TBD | TBD |

## Quickstart
conda activate sepsis-sentinel
pip install -r requirements.txt
uvicorn src.api.main:app --reload

## Dataset
MIMIC-IV v2.2 - PhysioNet credentialed access required.
Data not included in repo.
