import os
import duckdb
import pandas as pd
from pathlib import Path

P = Path('/Volumes/Tejas SSD/sepsis-sentinel')
LAB_ITEMIDS = [50912, 50813, 51301, 50885]
VITAL_ITEMIDS = [220045, 220179, 220210, 220277, 223761, 223900]
WINDOW_HOURS = 6

print('Loading Silver vitals...')
silver = pd.read_parquet(str(P/'data/delta/silver/vitals_features/silver_vitals.parquet'))
print(f'  Silver rows: {len(silver):,}')

print('Loading and windowing labs...')
con = duckdb.connect()
labs = con.execute(f"""
WITH stays AS (
    SELECT stay_id, hadm_id, intime
    FROM read_csv_auto('{P}/data/raw/icustays.csv.gz', header=true)
),
labs_raw AS (
    SELECT l.hadm_id, l.charttime, l.itemid, l.valuenum
    FROM read_parquet('{P}/data/labs_filtered.parquet') l
    WHERE l.itemid IN ({','.join(map(str, LAB_ITEMIDS))})
      AND l.valuenum IS NOT NULL
),
windowed AS (
    SELECT
        s.stay_id,
        FLOOR(DATEDIFF('second', s.intime::TIMESTAMP, l.charttime::TIMESTAMP) / 3600.0 / 6) AS window_id,
        l.itemid,
        l.valuenum,
        l.charttime
    FROM labs_raw l
    JOIN stays s ON l.hadm_id = s.hadm_id
    WHERE l.charttime::TIMESTAMP >= s.intime::TIMESTAMP
)
SELECT
    stay_id,
    CAST(window_id AS INTEGER) AS window_id,
    {', '.join([
        f"LAST(CASE WHEN itemid={iid} THEN valuenum END ORDER BY charttime) AS lab_{iid}_last,"
        f"CASE WHEN COUNT(CASE WHEN itemid={iid} THEN valuenum END) = 0 THEN 1 ELSE 0 END AS lab_{iid}_missing"
        for iid in LAB_ITEMIDS
    ])}
FROM windowed
GROUP BY stay_id, window_id
""").df()
print(f'  Lab windows: {len(labs):,}')

print('Loading sepsis labels...')
features = pd.read_parquet(str(P/'data/features_v1.parquet'), columns=['stay_id','sepsis_label'])
labels = features.drop_duplicates('stay_id')
print(f'  Labels: {len(labels):,} stays')

print('Joining all layers...')
gold = silver.merge(labs, on=['stay_id','window_id'], how='left')
gold = gold.merge(labels, on='stay_id', how='left')

# Fill missing lab flags
for iid in LAB_ITEMIDS:
    gold[f'lab_{iid}_missing'] = gold[f'lab_{iid}_missing'].fillna(1).astype(int)

# Compute temporal features
gold['icu_hours_elapsed'] = (gold['window_id'] * WINDOW_HOURS + WINDOW_HOURS / 2.0).astype(float)
gold['time_of_day'] = pd.to_datetime(gold['window_end']).dt.hour.astype(int)
gold['sepsis_label'] = gold['sepsis_label'].fillna(0).astype(int)

# Select final 43 columns
vital_cols = [f'vital_{iid}_{s}' for iid in VITAL_ITEMIDS for s in ('mean','min','max','std','missing')]
lab_cols = [c for iid in LAB_ITEMIDS for c in (f'lab_{iid}_last', f'lab_{iid}_missing')]
final_cols = ['stay_id','window_end'] + vital_cols + lab_cols + ['icu_hours_elapsed','time_of_day','sepsis_label']
gold = gold[final_cols]

assert len(gold.columns) == 43, f'Expected 43 columns, got {len(gold.columns)}'

print('Writing Gold parquet...')
OUT = P / 'data/delta/gold/features'
OUT.mkdir(parents=True, exist_ok=True)
gold.to_parquet(str(OUT / 'gold_features.parquet'), index=False)

n = len(gold)
sep_rate = gold['sepsis_label'].mean()
print(f'Gold layer complete. Rows: {n:,}, Columns: {len(gold.columns)}')
print(f'Sepsis rate: {sep_rate:.1%}')
print(gold[['stay_id','window_end','vital_220045_mean','lab_50912_last','icu_hours_elapsed','sepsis_label']].head(3))
