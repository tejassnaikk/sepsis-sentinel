"""
Model training, evaluation, and persistence for SepsisSentinel.

Input: windowed feature matrix produced by compute_features() in
src/features/feature_engineering.py.  One row per (stay_id, window_end).

Design notes
------------
* Temporal split — patients admitted earlier go to train, later ones to test.
  A random split would let future patients leak into training data, which would
  inflate all metrics and produce a model that degrades silently after
  deployment.
* scale_pos_weight — sepsis prevalence in MIMIC-IV is ~10-15 %.  Without
  reweighting the classifier maximises accuracy by ignoring rare positives.
  XGBoost's scale_pos_weight multiplies the gradient of positive samples by
  neg/pos, recovering approximately the same gradient magnitude as a balanced
  dataset.
* AUPRC over AUROC for early-stopping — with heavy class imbalance AUROC can
  look high even when the precision-recall curve is poor.  AUPRC directly
  measures the trade-off that matters clinically.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Path constants derived from this file's location so the script works
# regardless of the current working directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_REPORTS_DIR  = _PROJECT_ROOT / "reports"
_MODELS_DIR   = _PROJECT_ROOT / "models_local"

# Columns present in the feature matrix that are NOT model features
_NON_FEATURE_COLS = {"stay_id", "window_end", "sepsis_label"}

# Threshold used for precision / recall / F1 (tunable without retraining)
_DEFAULT_THRESHOLD = 0.35


# ---------------------------------------------------------------------------
# 1. Data splitting
# ---------------------------------------------------------------------------

def prepare_splits(
    features_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series,    pd.Series,    pd.Series,
]:
    """Split the feature matrix into train / val / test sets temporally.

    Stays are ordered by their earliest window_end (a proxy for ICU admission
    time).  The first 70 % of stays form the training set, the next 15 % the
    validation set, and the final 15 % the test set.

    Using a temporal split is critical for clinical time-series models: a
    random split allows a model to interpolate between earlier and later
    patients from the same period, producing optimistic metrics that collapse
    in prospective deployment.

    Parameters
    ----------
    features_df:
        Full feature matrix from compute_features().  Must contain stay_id,
        window_end, sepsis_label, and all feature columns.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
        X sets have stay_id, window_end, and sepsis_label dropped.
        y sets are the sepsis_label Series aligned to each X set.
    """
    # Order stays by their first observed window_end so that earlier
    # admissions end up in the training set.
    stay_order: list[int] = (
        features_df.groupby("stay_id")["window_end"]
        .min()
        .sort_values()
        .index.tolist()
    )

    n_stays = len(stay_order)
    n_train = int(n_stays * 0.70)
    n_val   = int(n_stays * 0.15)
    # Test gets the remainder so rounding never loses stays
    train_ids = set(stay_order[:n_train])
    val_ids   = set(stay_order[n_train : n_train + n_val])
    test_ids  = set(stay_order[n_train + n_val :])

    def _slice(id_set: set[int]) -> tuple[pd.DataFrame, pd.Series]:
        rows = features_df[features_df["stay_id"].isin(id_set)]
        X = rows.drop(columns=list(_NON_FEATURE_COLS))
        y = rows["sepsis_label"].astype(int)
        return X, y

    X_train, y_train = _slice(train_ids)
    X_val,   y_val   = _slice(val_ids)
    X_test,  y_test  = _slice(test_ids)

    print("Split summary (temporal):")
    for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        pos   = int(y.sum())
        total = len(y)
        pct   = 100.0 * pos / total if total else 0.0
        print(f"  {name:5s}: {total:>7,} rows | {pos:>5,} positive ({pct:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# 2. Training
# ---------------------------------------------------------------------------

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
) -> xgb.XGBClassifier:
    """Train an XGBoost classifier with early stopping on AUPRC.

    Hyperparameters are chosen to trade off training speed and generalisation
    for a medium-size clinical dataset (~100 k windows):
    * n_estimators=500 with early stopping avoids manual tuning of tree count.
    * max_depth=6 is deep enough to capture interaction effects between vitals
      without overfitting individual stays.
    * learning_rate=0.05 (shrinkage) combined with subsample/colsample_bytree
      adds regularisation comparable to dropout in neural networks.
    * scale_pos_weight compensates for the ~10:1 class imbalance typical of
      sepsis cohorts; without it the model collapses to predicting all-negative.

    Parameters
    ----------
    X_train, y_train:
        Training features and labels (output of prepare_splits).
    X_val, y_val:
        Validation set used for early stopping.

    Returns
    -------
    Trained XGBClassifier stopped at the best AUPRC iteration.
    """
    n_neg  = int((y_train == 0).sum())
    n_pos  = int((y_train == 1).sum())
    # Avoid division by zero for degenerate synthetic datasets
    spw = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0

    print(f"Training XGBoost | n_train={len(y_train):,} | "
          f"pos={n_pos:,} neg={n_neg:,} | scale_pos_weight={spw:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        early_stopping_rounds=20,
        eval_metric="aucpr",
        # Deterministic behaviour across runs
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,   # print every 50 trees to show progress without flooding
    )

    print(f"Training complete | best_iteration={model.best_iteration} | "
          f"best_aucpr={model.best_score:.4f}")
    return model


# ---------------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model:  xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = _DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Evaluate the model on the held-out test set and write a metrics report.

    Threshold=0.35 is intentionally below 0.5 to favour recall over precision:
    in a sepsis screening context a false negative (missed sepsis) is more
    harmful than a false positive (unnecessary review).

    Parameters
    ----------
    model:
        Trained XGBClassifier from train_model().
    X_test, y_test:
        Held-out test features and labels.
    threshold:
        Decision threshold applied to predicted probabilities for
        precision / recall / F1 computation.

    Returns
    -------
    dict with keys: auroc, auprc, precision, recall, f1, threshold.
    Also writes a Markdown table to reports/metrics.md.
    """
    proba  = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    # AUROC and AUPRC both require at least two classes in y_test; guard against
    # degenerate test sets that can arise from very small synthetic datasets.
    try:
        auroc = float(roc_auc_score(y_test, proba))
    except ValueError:
        auroc = float("nan")

    try:
        auprc = float(average_precision_score(y_test, proba))
    except ValueError:
        auprc = float("nan")

    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec  = float(recall_score(y_test,    y_pred, zero_division=0))
    f1   = float(f1_score(y_test,        y_pred, zero_division=0))

    metrics = {
        "auroc":     auroc,
        "auprc":     auprc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "threshold": threshold,
    }

    print("\nTest-set metrics:")
    print(f"  AUROC  : {auroc:.4f}")
    print(f"  AUPRC  : {auprc:.4f}")
    print(f"  Precision (t={threshold}) : {prec:.4f}")
    print(f"  Recall    (t={threshold}) : {rec:.4f}")
    print(f"  F1        (t={threshold}) : {f1:.4f}")

    _write_metrics_report(metrics, y_test)
    return metrics


def _write_metrics_report(metrics: dict[str, float], y_test: pd.Series) -> None:
    """Serialise evaluation metrics to reports/metrics.md as a Markdown table."""
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = _REPORTS_DIR / "metrics.md"

    n_total = len(y_test)
    n_pos   = int(y_test.sum())
    n_neg   = n_total - n_pos
    ts      = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    t       = metrics["threshold"]

    lines = [
        "# SepsisSentinel — Model Evaluation Report",
        "",
        f"Generated: {ts}  ",
        f"Model: XGBoost  ",
        f"Decision threshold: {t}",
        "",
        "## Performance Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| AUROC | {metrics['auroc']:.4f} |",
        f"| AUPRC | {metrics['auprc']:.4f} |",
        f"| Precision (t={t}) | {metrics['precision']:.4f} |",
        f"| Recall    (t={t}) | {metrics['recall']:.4f} |",
        f"| F1        (t={t}) | {metrics['f1']:.4f} |",
        "",
        "## Test Set Composition",
        "",
        "| | Count | % |",
        "| --- | --- | --- |",
        f"| Positive (sepsis) | {n_pos:,} | {100*n_pos/n_total:.1f}% |",
        f"| Negative          | {n_neg:,} | {100*n_neg/n_total:.1f}% |",
        f"| Total             | {n_total:,} | 100.0% |",
    ]

    report_path.write_text("\n".join(lines) + "\n")
    print(f"Metrics report saved → {report_path}")


# ---------------------------------------------------------------------------
# 4. Model persistence
# ---------------------------------------------------------------------------

def save_model(
    model: xgb.XGBClassifier,
    path:  str | Path | None = None,
) -> Path:
    """Save the trained model to disk in XGBoost JSON format.

    JSON is preferred over the legacy binary format because it is human-
    readable, version-stable, and portable across XGBoost versions.

    Parameters
    ----------
    model:
        Trained XGBClassifier to persist.
    path:
        Destination file path.  Defaults to models_local/xgboost_v1.json
        relative to the project root.

    Returns
    -------
    Resolved Path where the model was written.
    """
    if path is None:
        path = _MODELS_DIR / "xgboost_v1.json"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)
    print(f"Model saved → {path}  ({path.stat().st_size / 1024:.1f} KB)")
    return path


# ---------------------------------------------------------------------------
# Smoke-test: full pipeline on synthetic data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Run the complete train → evaluate → save pipeline on synthetic data.

    The synthetic dataset is small enough to finish in a few seconds but
    deliberately mirrors the exact column structure of compute_features()
    so any schema mismatch surfaces immediately.
    """
    from src.features.feature_engineering import VITAL_ITEMIDS, LAB_ITEMIDS

    print("=" * 60)
    print("SepsisSentinel — full pipeline smoke test (synthetic data)")
    print("=" * 60)

    rng = np.random.default_rng(42)
    N_STAYS         = 300
    WINDOWS_PER_STAY = 4
    SEPSIS_RATE      = 0.20

    rows: list[dict] = []
    t0 = pd.Timestamp("2020-01-01")

    for i in range(N_STAYS):
        # Spread admissions over time so the temporal split is meaningful
        intime = t0 + pd.Timedelta(hours=i * 6)
        label  = int(rng.random() < SEPSIS_RATE)

        for w in range(WINDOWS_PER_STAY):
            window_end = intime + pd.Timedelta(hours=(w + 1) * 6)
            row: dict = {
                "stay_id":           i,
                "window_end":        window_end,
                "icu_hours_elapsed": (w + 1) * 6.0,
                "time_of_day":       window_end.hour,
                "sepsis_label":      label,
            }
            for iid in VITAL_ITEMIDS:
                # Sepsis-positive stays get slightly elevated vitals to give
                # the model a learnable signal
                base = 90.0 + 15.0 * label
                row[f"vital_{iid}_mean"]    = float(rng.normal(base, 10))
                row[f"vital_{iid}_min"]     = float(rng.normal(base - 5, 5))
                row[f"vital_{iid}_max"]     = float(rng.normal(base + 5, 5))
                row[f"vital_{iid}_std"]     = float(rng.uniform(1, 8))
                row[f"vital_{iid}_missing"] = int(rng.random() < 0.05)
            for iid in LAB_ITEMIDS:
                row[f"lab_{iid}_last"]    = float(rng.normal(2.0 + label, 0.5))
                row[f"lab_{iid}_missing"] = int(rng.random() < 0.15)
            rows.append(row)

    features_df = pd.DataFrame(rows)
    print(f"\nSynthetic dataset: {len(features_df):,} rows, "
          f"{features_df['stay_id'].nunique()} stays, "
          f"{features_df['sepsis_label'].mean():.0%} sepsis-positive\n")

    # ── 1. Split ─────────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits(features_df)

    # ── 2. Train ─────────────────────────────────────────────────────────────
    print()
    model = train_model(X_train, y_train, X_val, y_val)

    # ── 3. Evaluate ──────────────────────────────────────────────────────────
    metrics = evaluate_model(model, X_test, y_test)

    # ── 4. Save ──────────────────────────────────────────────────────────────
    print()
    saved_path = save_model(model)

    print("\n" + "=" * 60)
    print("Smoke test complete.")
    print(f"  AUROC : {metrics['auroc']:.4f}")
    print(f"  AUPRC : {metrics['auprc']:.4f}")
    print(f"  F1    : {metrics['f1']:.4f}")
    print(f"  Model : {saved_path}")
    print("=" * 60)
