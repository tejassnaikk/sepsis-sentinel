import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path

P = Path('/Volumes/Tejas SSD/sepsis-sentinel')
ALERT_THRESHOLD = 0.35

FEATURE_COLS = [
    "vital_220045_mean","vital_220045_min","vital_220045_max","vital_220045_std","vital_220045_missing",
    "vital_220179_mean","vital_220179_min","vital_220179_max","vital_220179_std","vital_220179_missing",
    "vital_220210_mean","vital_220210_min","vital_220210_max","vital_220210_std","vital_220210_missing",
    "vital_220277_mean","vital_220277_min","vital_220277_max","vital_220277_std","vital_220277_missing",
    "vital_223761_mean","vital_223761_min","vital_223761_max","vital_223761_std","vital_223761_missing",
    "vital_223900_mean","vital_223900_min","vital_223900_max","vital_223900_std","vital_223900_missing",
    "lab_50912_last","lab_50912_missing",
    "lab_50813_last","lab_50813_missing",
    "lab_51301_last","lab_51301_missing",
    "lab_50885_last","lab_50885_missing",
    "icu_hours_elapsed","time_of_day"
]

print('Loading Gold features...')
gold = pd.read_parquet(str(P/'data/delta/gold/features/gold_features.parquet'))
print(f'  Gold rows: {len(gold):,}')

print('Loading XGBoost model...')
model = xgb.XGBClassifier()
model.load_model(str(P/'models_local/xgboost_v1.json'))

print('Scoring...')
X = gold[FEATURE_COLS].astype(float)
gold['sepsis_prob'] = model.predict_proba(X)[:, 1]
gold['sepsis_alert'] = (gold['sepsis_prob'] >= ALERT_THRESHOLD).astype(int)

# Validation
print(f'\nValidation:')
print(f'  Prob range: {gold.sepsis_prob.min():.4f} - {gold.sepsis_prob.max():.4f}')
print(f'  Alert rate: {gold.sepsis_alert.mean():.1%} ({gold.sepsis_alert.sum():,} alerts)')

# Confusion summary
print('\nConfusion-style summary:')
print(gold.groupby(['sepsis_label','sepsis_alert']).size().reset_index(name='count').to_string(index=False))

# AUROC
from sklearn.metrics import roc_auc_score, average_precision_score
auroc = roc_auc_score(gold['sepsis_label'], gold['sepsis_prob'])
auprc = average_precision_score(gold['sepsis_label'], gold['sepsis_prob'])
print(f'\nAUROC: {auroc:.3f}  (training was 0.806)')
print(f'AUPRC: {auprc:.3f}  (training was 0.609)')

# Top 10
print('\nTop 10 highest-risk windows:')
print(gold.nlargest(10, 'sepsis_prob')[['stay_id','window_end','sepsis_prob','sepsis_alert','sepsis_label']].to_string(index=False))

# Save
OUT = P/'data/delta/predictions'
OUT.mkdir(parents=True, exist_ok=True)
gold.to_parquet(str(OUT/'predictions.parquet'), index=False)
print(f'\nPredictions saved. Rows: {len(gold):,}')
