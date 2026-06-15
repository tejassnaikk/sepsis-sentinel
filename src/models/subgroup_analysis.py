import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

P = Path('/Volumes/Tejas SSD/sepsis-sentinel')

print('Loading predictions...')
preds = pd.read_parquet(str(P/'data/delta/predictions/predictions.parquet'))
print(f'  Rows: {len(preds):,}')

print('Loading patients for age...')
patients = pd.read_csv(str(P/'data/raw/patients.csv.gz'), usecols=['subject_id','anchor_age'])
icustays = pd.read_csv(str(P/'data/raw/icustays.csv.gz'), usecols=['stay_id','subject_id'])
preds = preds.merge(icustays, on='stay_id', how='left').merge(patients, on='subject_id', how='left')

def subgroup_metrics(df, label='sepsis_label', prob='sepsis_prob', alert='sepsis_alert'):
    if len(df) < 100 or df[label].sum() < 10:
        return None
    auroc = roc_auc_score(df[label], df[prob])
    auprc = average_precision_score(df[label], df[prob])
    recall = df.loc[df[label]==1, alert].mean()
    precision = df.loc[df[alert]==1, label].mean() if df[alert].sum() > 0 else 0
    alert_rate = df[alert].mean()
    sepsis_rate = df[label].mean()
    return {
        'n_windows': len(df),
        'sepsis_rate': f'{sepsis_rate:.1%}',
        'AUROC': f'{auroc:.3f}',
        'AUPRC': f'{auprc:.3f}',
        'Recall@0.35': f'{recall:.1%}',
        'Precision@0.35': f'{precision:.1%}',
        'Alert_rate': f'{alert_rate:.1%}',
    }

rows = []

# --- Age buckets ---
bins = [0, 45, 65, 80, 120]
labels = ['18-45','45-65','65-80','80+']
preds['age_bucket'] = pd.cut(preds['anchor_age'], bins=bins, labels=labels)
for bucket in labels:
    sub = preds[preds['age_bucket'] == bucket]
    m = subgroup_metrics(sub)
    if m:
        rows.append({'Subgroup': f'Age {bucket}', **m})

# --- ICU LOS buckets ---
preds['los_bucket'] = pd.cut(preds['icu_hours_elapsed'],
    bins=[-1, 24, 72, 99999],
    labels=['Early (0-24h)', 'Mid (24-72h)', 'Extended (72h+)'])
for bucket in ['Early (0-24h)', 'Mid (24-72h)', 'Extended (72h+)']:
    sub = preds[preds['los_bucket'] == bucket]
    m = subgroup_metrics(sub)
    if m:
        rows.append({'Subgroup': f'LOS {bucket}', **m})

# --- Time of day ---
preds['shift'] = preds['time_of_day'].apply(lambda h: 'Day (6am-6pm)' if 6 <= h < 18 else 'Night (6pm-6am)')
for shift in ['Day (6am-6pm)', 'Night (6pm-6am)']:
    sub = preds[preds['shift'] == shift]
    m = subgroup_metrics(sub)
    if m:
        rows.append({'Subgroup': f'Shift {shift}', **m})

# --- Overall ---
m = subgroup_metrics(preds)
rows.insert(0, {'Subgroup': 'Overall', **m})

results = pd.DataFrame(rows)
print('\n' + '='*90)
print('SUBGROUP ANALYSIS')
print('='*90)
print(results.to_string(index=False))

results.to_csv(str(P/'reports/subgroup_analysis.csv'), index=False)
print(f'\nSaved to reports/subgroup_analysis.csv')
