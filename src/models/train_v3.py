import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import xgboost as xgb

P = Path('/Volumes/Tejas SSD/sepsis-sentinel')

VITAL_ITEMIDS = [220045, 220179, 220210, 220277, 223761, 223900]
LAB_ITEMIDS   = [50912, 50813, 51301, 50885]

BASE_FEATURE_COLS = [
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
gold = gold.sort_values(['stay_id','icu_hours_elapsed']).reset_index(drop=True)
print(f'  Rows: {len(gold):,}')

print('Engineering lag and delta features...')
vital_mean_cols = [f'vital_{iid}_mean' for iid in VITAL_ITEMIDS]
lab_last_cols   = [f'lab_{iid}_last' for iid in LAB_ITEMIDS]

lag_cols = []
for col in vital_mean_cols + lab_last_cols:
    lag_col   = f'{col}_lag1'
    delta_col = f'{col}_delta'
    gold[lag_col]   = gold.groupby('stay_id')[col].shift(1)
    gold[delta_col] = gold[col] - gold[lag_col]
    lag_cols += [lag_col, delta_col]

# Missingness trend: how many vitals missing vs last window
gold['n_missing_vitals'] = gold[[f'vital_{iid}_missing' for iid in VITAL_ITEMIDS]].sum(axis=1)
gold['n_missing_vitals_lag1'] = gold.groupby('stay_id')['n_missing_vitals'].shift(1)
gold['missing_trend'] = gold['n_missing_vitals'] - gold['n_missing_vitals_lag1']

extra_cols = ['n_missing_vitals', 'n_missing_vitals_lag1', 'missing_trend']
ALL_FEATURE_COLS = BASE_FEATURE_COLS + lag_cols + extra_cols
print(f'  Total features: {len(ALL_FEATURE_COLS)} (base {len(BASE_FEATURE_COLS)} + lag/delta {len(lag_cols)} + missing {len(extra_cols)})')

print('Splitting data...')
stays = sorted(gold['stay_id'].unique())
n = len(stays)
train_stays = set(stays[:int(n*0.70)])
val_stays   = set(stays[int(n*0.70):int(n*0.85)])
test_stays  = set(stays[int(n*0.85):])

train = gold[gold['stay_id'].isin(train_stays)]
val   = gold[gold['stay_id'].isin(val_stays)]
test  = gold[gold['stay_id'].isin(test_stays)]

X_train = train[ALL_FEATURE_COLS].astype(float).values
y_train = train['sepsis_label'].values
X_val   = val[ALL_FEATURE_COLS].astype(float).values
y_val   = val['sepsis_label'].values
X_test  = test[ALL_FEATURE_COLS].astype(float).values
y_test  = test['sepsis_label'].values

neg, pos = (y_train==0).sum(), (y_train==1).sum()
scale_pos_weight = neg / pos
print(f'  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}')
print(f'  scale_pos_weight: {scale_pos_weight:.2f}')

print('\nTraining XGBoost v3 with lag features...')
model = xgb.XGBClassifier(
    max_depth=5,
    learning_rate=0.05,
    n_estimators=1000,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    scale_pos_weight=scale_pos_weight,
    objective='binary:logistic',
    eval_metric='auc',
    early_stopping_rounds=50,
    random_state=42,
    verbosity=0,
    device='cpu',
)
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          verbose=100)

val_prob  = model.predict_proba(X_val)[:,1]
val_auroc = roc_auc_score(y_val, val_prob)
print(f'\nVal AUROC: {val_auroc:.4f}  (best iter: {model.best_iteration})')

test_prob  = model.predict_proba(X_test)[:,1]
test_auroc = roc_auc_score(y_test, test_prob)
test_auprc = average_precision_score(y_test, test_prob)

fpr, tpr, thresholds = roc_curve(y_test, test_prob)
youden = tpr - fpr
best_thresh = float(thresholds[np.argmax(youden)])
test_alert  = (test_prob >= best_thresh).astype(int)
test_recall = float((test_alert[y_test==1]).mean())

print(f'\n=== TEST SET RESULTS (v3 with lag features) ===')
print(f'AUROC:          {test_auroc:.3f}  (v1 was 0.806, v2 was 0.797)')
print(f'AUPRC:          {test_auprc:.3f}  (v1 was 0.609, v2 was 0.554)')
print(f'Best threshold: {best_thresh:.4f}')
print(f'Recall@thresh:  {test_recall:.3f}')

# Feature importance top 15
import pandas as pd
fi = pd.Series(model.feature_importances_, index=ALL_FEATURE_COLS)
print('\nTop 15 features by importance:')
print(fi.nlargest(15).to_string())

model.save_model(str(P/'models_local/xgboost_v3.json'))
print(f'\nModel saved to models_local/xgboost_v3.json')

# Save feature list for inference
import json
with open(str(P/'models_local/feature_cols_v3.json'), 'w') as f:
    json.dump(ALL_FEATURE_COLS, f)
print('Feature cols saved to models_local/feature_cols_v3.json')
