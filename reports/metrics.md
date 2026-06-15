# SepsisSentinel Model Metrics

## Training (features_v1.parquet, 889k windows)
- AUROC: 0.806
- AUPRC: 0.609
- Recall@0.35: 0.878

## Full Cohort Inference (934,767 windows, all 65,366 ICU stays)
- AUROC: 0.799
- AUPRC: 0.572
- Recall@0.35: 0.864
- Alert rate: 55.6% (threshold=0.35)
- Confusion: TP=198,883 FP=321,236 TN=383,364 FN=31,284
- Prob range: 0.005 - 0.995

## Model v3 (XGBoost with lag + delta features, 63 features)
- Val AUROC: 0.820
- Test AUROC: 0.810  (target was >0.82 on val ✅)
- Test AUPRC: 0.570
- Best threshold: 0.4905 (Youden's J)
- Recall@thresh: 75.4%
- Top feature: vital_223900_max (GCS, 23.8% importance)
- Key insight: lag1 + delta features for vitals/labs add temporal trend signal
