"""
SepsisSentinel REST API — v3.

Serves a trained XGBoost v3 sepsis-prediction model (63 features: 40 base
+ 10 lag-1 + 10 delta + n_missing_vitals + n_missing_vitals_lag1 + missing_trend)
via three endpoints: GET /health, POST /predict, GET /features.

The model file (xgboost_v3.json) and feature column list (feature_cols_v3.json)
are loaded once at startup via FastAPI's lifespan mechanism. XGBoost v3 handles
NaN natively; None values in the feature vector are passed through as NaN without
explicit imputation. Lag and delta features are computed from the optional
previous_window field when provided.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import uvicorn
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VITAL_ITEMIDS: list[int] = [220045, 220179, 220210, 220277, 223761, 223900]
LAB_ITEMIDS:   list[int] = [50912,  50813,  51301,  50885]

_ALERT_THRESHOLD: float = 0.49
_MODEL_VERSION:   str   = "v3"

_MODEL_DIR: Path = Path(__file__).resolve().parent.parent.parent / "models_local"

_MODEL_PATH: Path        = _MODEL_DIR / "xgboost_v3.json"
_FEATURE_COLS_PATH: Path = _MODEL_DIR / "feature_cols_v3.json"

# Populated at startup from feature_cols_v3.json; empty list until then
FEATURE_COLS: list[str] = []

# Base columns for which lag-1 and delta features are computed (10 total)
BASE_VITAL_MEANS: list[str] = [
    "vital_220045_mean", "vital_220179_mean", "vital_220210_mean",
    "vital_220277_mean", "vital_223761_mean", "vital_223900_mean",
]
BASE_LAB_LASTS: list[str] = [
    "lab_50912_last", "lab_50813_last", "lab_51301_last", "lab_50885_last",
]

# ---------------------------------------------------------------------------
# Pydantic input model
#
# Built programmatically so field names are derived from the same itemid
# constants used in feature_engineering.py — editing one place updates both.
# All 40 base feature fields are Optional[...] = None. The extra
# previous_window field accepts the prior window's base features for lag+delta
# computation; stay_id is echoed in the response for client-side correlation.
# ---------------------------------------------------------------------------


def _make_patient_features_model() -> type[BaseModel]:
    """Build the PatientFeatures Pydantic model from VITAL_ITEMIDS / LAB_ITEMIDS."""
    fields: dict[str, Any] = {}

    fields["stay_id"] = (
        Optional[int],
        Field(default=None, description="ICU stay identifier — echoed in the response"),
    )

    for iid in VITAL_ITEMIDS:
        for stat in ("mean", "min", "max", "std"):
            fields[f"vital_{iid}_{stat}"] = (
                Optional[float],
                Field(default=None, description=f"Vital {iid} — {stat} in window"),
            )
        fields[f"vital_{iid}_missing"] = (
            Optional[int],
            Field(default=None, ge=0, le=1,
                  description=f"1 if vital {iid} had no measurements in window"),
        )

    for iid in LAB_ITEMIDS:
        fields[f"lab_{iid}_last"] = (
            Optional[float],
            Field(default=None, description=f"Lab {iid} — most recent value in window"),
        )
        fields[f"lab_{iid}_missing"] = (
            Optional[int],
            Field(default=None, ge=0, le=1,
                  description=f"1 if lab {iid} had no results in window"),
        )

    fields["icu_hours_elapsed"] = (
        Optional[float],
        Field(default=None, ge=0.0,
              description="Hours elapsed since ICU admission at window end"),
    )
    fields["time_of_day"] = (
        Optional[int],
        Field(default=None, ge=0, le=23,
              description="Hour of day at window end (0–23)"),
    )

    fields["previous_window"] = (
        Optional[Dict[str, float]],
        Field(default=None,
              description=(
                  "Previous 6h window's base feature values (same field names as "
                  "this request). When provided, enables lag-1 and delta features."
              )),
    )

    return create_model("PatientFeatures", **fields)


PatientFeatures: type[BaseModel] = _make_patient_features_model()
PatientFeatures.__doc__ = (
    "One non-overlapping observation window for a single ICU patient. "
    "All base feature fields are optional — send None or omit any measurement "
    "that was unavailable. Supply previous_window to enable lag and delta features."
)

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    model_loaded: bool
    model_version: str


class PredictionResponse(BaseModel):
    """Response body for POST /predict."""

    stay_id: Optional[int]
    sepsis_probability: float
    sepsis_alert: bool
    threshold: float
    model_version: str


class FeaturesResponse(BaseModel):
    """Response body for GET /features."""

    features: list[str]
    count: int


class DriftRequest(BaseModel):
    """Request body for POST /drift."""

    windows: list[dict[str, float]]
    # List of feature dicts, each representing one 6h window.
    # Only base 40 features needed — lag features computed server-side.


# ---------------------------------------------------------------------------
# Application state
#
# Stored in a plain module-level dict so helpers can read it without needing
# a reference to the app object.
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {
    "model": None,           # XGBClassifier once loaded, else None
    "feature_columns": [],   # Ordered list from model.get_booster().feature_names
}

# ---------------------------------------------------------------------------
# Lifespan — model and feature column list loaded here, once, before first request
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the XGBoost v3 model and feature_cols_v3.json on startup."""
    if not _MODEL_PATH.exists():
        logger.warning(
            "Model file not found: %s  — /health will report model_loaded=false",
            _MODEL_PATH,
        )
    else:
        try:
            m = xgb.XGBClassifier()
            m.load_model(_MODEL_PATH)
            _state["model"] = m
            _state["feature_columns"] = list(m.get_booster().feature_names or [])
            logger.info(
                "Model loaded | path=%s | features=%d",
                _MODEL_PATH,
                len(_state["feature_columns"]),
            )
        except Exception as exc:
            logger.error(
                "Model load failed: %s  — /health will report model_loaded=false", exc
            )

    if not _FEATURE_COLS_PATH.exists():
        logger.warning(
            "Feature columns file not found: %s  — /features will use model feature_names",
            _FEATURE_COLS_PATH,
        )
    else:
        try:
            with open(_FEATURE_COLS_PATH) as fh:
                loaded: list[str] = json.load(fh)
            FEATURE_COLS.extend(loaded)
            logger.info("Feature columns loaded: %d columns", len(FEATURE_COLS))
        except Exception as exc:
            logger.error("Feature columns load failed: %s", exc)

    baseline_path = _MODEL_DIR / "feature_baseline.json"
    if baseline_path.exists():
        try:
            with open(baseline_path) as f:
                _state["baseline"] = json.load(f)
            logger.info("Feature baseline loaded: %d features", len(_state["baseline"].get("mean", {})))
        except Exception as exc:
            logger.error("Feature baseline load failed: %s", exc)
    else:
        logger.warning("Feature baseline not found: %s  — /drift will return 503", baseline_path)

    yield  # application runs here


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SepsisSentinel",
    description=(
        "Real-time ICU sepsis prediction API — v3 (63 features with lag+delta).  "
        "Accepts windowed vital-sign and lab features plus an optional previous window; "
        "returns a calibrated sepsis probability and binary alert flag."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------


def _build_feature_row(request: BaseModel) -> pd.DataFrame:
    """Build a 63-element feature vector from a PatientFeatures request.

    Feature composition
    -------------------
    Base (40):     all vital/lab stats + missing flags + icu_hours_elapsed + time_of_day.
                   None values are preserved as NaN — XGBoost handles them natively.
    Lag-1 (10):    {col}_lag1 for each col in BASE_VITAL_MEANS + BASE_LAB_LASTS.
                   Set from previous_window[col] when available; None otherwise.
    Delta (10):    {col}_delta = current - prev when both are present; None otherwise.
    Aggregate (3): n_missing_vitals, n_missing_vitals_lag1, missing_trend.

    Column order follows FEATURE_COLS (from feature_cols_v3.json) with fallback
    to model.get_booster().feature_names, then the static v1 derivation.
    """
    raw: dict[str, Any] = request.model_dump()

    # Extract and remove non-feature fields before building the feature dict
    stay_id = raw.pop("stay_id", None)
    prev: dict[str, float] = raw.pop("previous_window", None) or {}

    # Base feature values — None preserved for XGBoost NaN handling
    all_vals: dict[str, Optional[float]] = {
        k: (float(v) if v is not None else None) for k, v in raw.items()
    }

    # -- Lag-1 and delta features -----------------------------------------------
    for col in BASE_VITAL_MEANS + BASE_LAB_LASTS:
        current_val = all_vals.get(col)
        if col in prev:
            prev_val = float(prev[col])
            all_vals[col + "_lag1"]  = prev_val
            all_vals[col + "_delta"] = (
                (current_val - prev_val) if current_val is not None else None
            )
        else:
            all_vals[col + "_lag1"]  = None
            all_vals[col + "_delta"] = None

    # -- Aggregate missing features -----------------------------------------------
    vital_missing_cols = [f"vital_{iid}_missing" for iid in VITAL_ITEMIDS]
    n_missing_vitals = sum(1 for c in vital_missing_cols if all_vals.get(c) == 1.0)
    all_vals["n_missing_vitals"] = float(n_missing_vitals)

    if prev:
        n_missing_vitals_lag1 = sum(
            1 for c in vital_missing_cols if prev.get(c) == 1
        )
        all_vals["n_missing_vitals_lag1"] = float(n_missing_vitals_lag1)
        all_vals["missing_trend"] = float(n_missing_vitals - n_missing_vitals_lag1)
    else:
        all_vals["n_missing_vitals_lag1"] = None
        all_vals["missing_trend"] = None

    # -- Assemble in authoritative column order -----------------------------------
    feature_cols = (
        _state["feature_columns"]
        or FEATURE_COLS
        or _static_feature_columns()
    )
    return pd.DataFrame([{col: all_vals.get(col) for col in feature_cols}])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service liveness, model readiness, and model version.

    Returns
    -------
    status        : "ok" (the process is alive and the web server is healthy)
    model_loaded  : true if xgboost_v3.json loaded successfully at startup
    model_version : "v3"
    """
    return HealthResponse(
        status="ok",
        model_loaded=_state["model"] is not None,
        model_version=_MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PatientFeatures) -> PredictionResponse:
    """Accept one observation window and return a sepsis risk score.

    All 40 base feature fields are optional. None/omitted values are passed
    as NaN and handled natively by XGBoost v3. Supply previous_window (a dict
    of the same base feature names from the prior 6h window) to enable lag-1
    and delta features, which improve recall on deteriorating patients.

    Returns
    -------
    stay_id            : echoed from the request for client-side correlation
    sepsis_probability : model's predicted probability of sepsis in [0, 1]
    sepsis_alert       : true when probability >= threshold (0.49)
    threshold          : Youden's J optimised on held-out validation set
    model_version      : "v3"

    Raises
    ------
    HTTP 503 if the model failed to load at startup (see GET /health).
    """
    if _state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded — check GET /health for details",
        )

    row  = _build_feature_row(request)
    row  = row.astype(float)
    prob = float(_state["model"].predict_proba(row)[0, 1])

    return PredictionResponse(
        stay_id=getattr(request, "stay_id", None),
        sepsis_probability=round(prob, 6),
        sepsis_alert=prob >= _ALERT_THRESHOLD,
        threshold=_ALERT_THRESHOLD,
        model_version=_MODEL_VERSION,
    )


@app.get("/features", response_model=FeaturesResponse)
async def feature_list() -> FeaturesResponse:
    """Return the ordered list of 63 feature names expected by POST /predict.

    Priority: FEATURE_COLS (from feature_cols_v3.json) → model embedded
    feature_names → static v1 derivation (40 features, fallback only).

    Returns
    -------
    features : ordered list of feature name strings (63 for v3)
    count    : len(features) — convenience field for quick validation
    """
    cols = (
        FEATURE_COLS
        or _state["feature_columns"]
        or _static_feature_columns()
    )
    return FeaturesResponse(features=cols, count=len(cols))


@app.post("/drift")
def detect_drift(request: DriftRequest):
    """Score a batch of feature windows for data drift against the training baseline.

    Compares incoming feature distributions to the training-set statistics in
    feature_baseline.json. Drift is measured per feature via a combined score
    of mean-shift (in training std-devs) and fraction of values outside the
    training p10–p90 range. An overall score above 0.3 triggers drift_detected.

    Returns
    -------
    n_windows             : number of windows in the batch
    n_features_scored     : features present in both batch and baseline
    overall_drift_score   : mean drift score across all scored features
    drift_detected        : true when overall_drift_score > 0.3
    drifted_features      : list of features with drift_score > 0.5
    feature_drift         : per-feature breakdown dict
    """
    if "baseline" not in _state:
        raise HTTPException(status_code=503, detail="Baseline not loaded")
    if not request.windows:
        raise HTTPException(status_code=400, detail="No windows provided")

    baseline = _state["baseline"]
    df = pd.DataFrame(request.windows).astype(float)

    # Only score features present in both the batch and baseline
    scored_features = [f for f in baseline["mean"] if f in df.columns]

    drift_scores: dict[str, Any] = {}
    drifted_features: list[str] = []

    for feat in scored_features:
        train_mean = baseline["mean"][feat]
        train_std  = baseline["std"][feat]
        train_p10  = baseline["p10"][feat]
        train_p90  = baseline["p90"][feat]

        if train_std == 0 or pd.isna(train_std):
            continue

        incoming_mean = float(df[feat].mean())
        incoming_std  = float(df[feat].std()) if len(df) > 1 else 0.0

        # Z-score of mean shift (how many training std-devs has the mean moved)
        mean_shift = abs(incoming_mean - train_mean) / train_std

        # Fraction of incoming values outside training p10-p90 range
        out_of_range = float(((df[feat] < train_p10) | (df[feat] > train_p90)).mean())

        # Combined drift score (0 = no drift, higher = more drift)
        drift_score = round((mean_shift * 0.6 + out_of_range * 0.4), 4)

        drift_scores[feat] = {
            "drift_score": drift_score,
            "mean_shift_z": round(mean_shift, 4),
            "out_of_range_pct": round(out_of_range * 100, 2),
            "incoming_mean": round(incoming_mean, 4),
            "training_mean": round(train_mean, 4),
        }

        if drift_score > 0.5:
            drifted_features.append(feat)

    overall_drift_score = round(
        sum(v["drift_score"] for v in drift_scores.values()) / len(drift_scores)
        if drift_scores else 0.0,
        4,
    )

    return {
        "n_windows": len(df),
        "n_features_scored": len(scored_features),
        "overall_drift_score": overall_drift_score,
        "drift_detected": overall_drift_score > 0.3,
        "drifted_features": drifted_features,
        "feature_drift": drift_scores,
    }


def _static_feature_columns() -> list[str]:
    """Derive the 40-feature v1 column list as a last-resort fallback.

    Used only when both FEATURE_COLS and the model's embedded feature_names
    are unavailable (i.e., both the model and feature_cols_v3.json failed to
    load at startup).
    """
    cols: list[str] = []
    for iid in VITAL_ITEMIDS:
        for s in ("mean", "min", "max", "std", "missing"):
            cols.append(f"vital_{iid}_{s}")
    for iid in LAB_ITEMIDS:
        cols.append(f"lab_{iid}_last")
        cols.append(f"lab_{iid}_missing")
    cols.extend(["icu_hours_elapsed", "time_of_day"])
    return cols


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    uvicorn.run(app, host="0.0.0.0", port=8000)
