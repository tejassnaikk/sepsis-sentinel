import os
import duckdb
import pandas as pd
from pathlib import Path

P = Path('/Volumes/Tejas SSD/sepsis-sentinel')
VITAL_ITEMIDS = [220045, 220179, 220210, 220277, 223761, 223900]
OUT = P / 'data/delta/silver/vitals_features'
OUT.mkdir(parents=True, exist_ok=True)

print('Loading data...')
con = duckdb.connect()

result = con.execute(f"""
WITH vitals AS (
    SELECT stay_id, charttime, itemid, valuenum
    FROM read_parquet('{P}/data/vitals_filtered.parquet')
    WHERE itemid IN ({','.join(map(str, VITAL_ITEMIDS))})
      AND valuenum IS NOT NULL
),
stays AS (
    SELECT stay_id, intime
    FROM read_csv_auto('{P}/data/raw/icustays.csv.gz', header=true)
),
windowed AS (
    SELECT
        v.stay_id,
        v.itemid,
        v.valuenum,
        s.intime,
        FLOOR(DATEDIFF('second', s.intime::TIMESTAMP, v.charttime::TIMESTAMP) / 3600.0 / 6) AS window_id
    FROM vitals v
    JOIN stays s ON v.stay_id = s.stay_id
    WHERE v.charttime::TIMESTAMP >= s.intime::TIMESTAMP
)
SELECT
    stay_id,
    CAST(window_id AS INTEGER) AS window_id,
    intime,
    {', '.join([
        f"AVG(CASE WHEN itemid={iid} THEN valuenum END) AS vital_{iid}_mean,"
        f"MIN(CASE WHEN itemid={iid} THEN valuenum END) AS vital_{iid}_min,"
        f"MAX(CASE WHEN itemid={iid} THEN valuenum END) AS vital_{iid}_max,"
        f"STDDEV(CASE WHEN itemid={iid} THEN valuenum END) AS vital_{iid}_std,"
        f"CASE WHEN COUNT(CASE WHEN itemid={iid} THEN valuenum END) = 0 THEN 1 ELSE 0 END AS vital_{iid}_missing"
        for iid in VITAL_ITEMIDS
    ])}
FROM windowed
GROUP BY stay_id, window_id, intime
""").df()

print(f'Aggregated {len(result):,} Silver windows')

# Compute window_end
result['intime'] = pd.to_datetime(result['intime'])
result['window_end'] = result['intime'] + pd.to_timedelta((result['window_id'] + 1) * 6, unit='h')
result['window_date'] = result['window_end'].dt.date

# Drop intime
result = result.drop(columns=['intime'])

print('Writing Silver parquet...')
result.to_parquet(str(OUT / 'silver_vitals.parquet'), index=False)
print(f'Silver layer complete. Rows: {len(result):,}, Columns: {len(result.columns)}')
print(result[['stay_id','window_id','window_end','vital_220045_mean']].head(3))
