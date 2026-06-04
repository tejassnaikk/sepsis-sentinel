"""
SepsisSentinel REST API.

Serves a trained XGBoost sepsis-prediction model via three endpoints:
GET /health, POST /predict, GET /features.

The model file is loaded once at startup via FastAPI's lifespan mechanism.
All incoming feature fields are optional; missing values are imputed server-
side before inference so the API tolerates incomplete ICU observations.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

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

_ALERT_THRESHOLD: float = 0.35
_MODEL_VERSION:   str   = "v1"
# Model path resolved relative to this file so the API works from any cwd
_MODEL_PATH: Path = (
    Path(__file__).resolve().parent.parent.parent / "models_local" / "xgboost_v1.json"
)

# ---------------------------------------------------------------------------
# Pydantic input model
#
# Built programmatically so field names are derived from the same itemid
# constants used in feature_engineering.py — editing one place updates both.
# All 40 feature fields are Optional[...] = None because ICU data is routinely
# incomplete and missing-value handling is the server's responsibility.
# ---------------------------------------------------------------------------

def _make_patient_features_model() -> type[BaseModel]:
    """Build the PatientFeatures Pydantic model from VITAL_ITEMIDS / LAB_ITEMIDS."""
    fields: dict[str, Any] = {}

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

    return create_model("PatientFeatures", **fields)


# The type annotation is BaseModel so IDEs don't complain about the dynamic class;
# at runtime this is a full Pydantic v2 model with all 40 fields.
PatientFeatures: type[BaseModel] = _make_patient_features_model()
PatientFeatures.__doc__ = (
    "One non-overlapping observation window for a single ICU patient. "
    "All fields are optional — send None or omit any measurement that was "
    "unavailable in the window.  Missing-value imputation is handled server-side."
)

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    model_loaded: bool


class PredictionResponse(BaseModel):
    """Response body for POST /predict."""

    sepsis_probability: float
    alert: bool
    threshold: float
    model_version: str


class FeaturesResponse(BaseModel):
    """Response body for GET /features."""

    features: list[str]
    count: int


# ---------------------------------------------------------------------------
# Application state
#
# Stored in a plain module-level dict so helpers can read it without needing
# a reference to the app object.  FastAPI's app.state would work too, but
# accessing it from outside a request context requires the app handle.
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {
    "model": None,           # XGBClassifier once loaded, else None
    "feature_columns": [],   # Ordered list from model.get_booster().feature_names
}

# ---------------------------------------------------------------------------
# Lifespan — model is loaded here, once, before the first request is served
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the XGBoost model on startup; yield; nothing to close on shutdown."""
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
            # feature_names is the authoritative column order embedded in the
            # model file — using it here guarantees inference column order is
            # always consistent with training, even if constants drift.
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

    yield  # application runs here

    # XGBoost has no resources to release; nothing to do on shutdown


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SepsisSentinel",
    description=(
        "Real-time ICU sepsis prediction API.  "
        "Accepts windowed vital-sign and lab features; returns a calibrated "
        "sepsis probability and binary alert flag."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Imputation helper
# ---------------------------------------------------------------------------


def _build_feature_row(features: BaseModel) -> pd.DataFrame:
    """Impute None → 0.0, propagate missing flags, return a model-ready DataFrame.

    Two-step imputation strategy
    ----------------------------
    1.  Every None field is replaced with 0.0.  For missing-flag fields this
        means "flag = 0" (data was present) — which the second step corrects
        wherever necessary.
    2.  For each VALUE field that was originally None (vital stat or lab result),
        the corresponding *_missing flag is forced to 1.0.  This overrides
        whatever the caller sent for that flag, ensuring the model always
        receives a consistent signal when a measurement is absent.

    Column order is taken from _state["feature_columns"] (the model's own
    feature_names) rather than from the Pydantic field definition order, so
    the DataFrame always aligns with what the model was trained on.

    Parameters
    ----------
    features:
        A PatientFeatures instance (validated by FastAPI before this is called).

    Returns
    -------
    Single-row DataFrame with dtype float64 and columns in model order.
    """
    raw: dict[str, Any] = features.model_dump()

    # Record which fields were absent before any filling
    was_none: set[str] = {k for k, v in raw.items() if v is None}

    # Fill all None → 0.0 (covers both value fields and missing-flag fields)
    imputed: dict[str, float] = {
        k: float(v) if v is not None else 0.0 for k, v in raw.items()
    }

    # For every absent value field, force its missing-flag partner to 1.
    for field in was_none:
        if field.startswith("vital_") and not field.endswith("_missing"):
            # "vital_220045_mean" → ["vital", "220045", "mean"]
            # Joining the first two segments gives the shared prefix
            flag = "_".join(field.split("_")[:2]) + "_missing"
            if flag in imputed:
                imputed[flag] = 1.0

        elif field.startswith("lab_") and field.endswith("_last"):
            # "lab_50912_last" → "lab_50912_missing"
            flag = field.replace("_last", "_missing")
            if flag in imputed:
                imputed[flag] = 1.0

    return pd.DataFrame([imputed])[_state["feature_columns"]]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service liveness and model readiness.

    Returns
    -------
    status       : "ok" (the process is alive and the web server is healthy)
    model_loaded : true if the XGBoost model loaded successfully at startup;
                   false if the model file was missing or raised an exception.
                   When false, POST /predict will return HTTP 503.
    """
    return HealthResponse(status="ok", model_loaded=_state["model"] is not None)


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: PatientFeatures) -> PredictionResponse:
    """Accept one observation window and return a sepsis risk score.

    Input fields are all optional.  Any field sent as null (or omitted) is
    imputed to 0.0 server-side, with the corresponding missing-flag set to 1
    so the model is aware which features were absent.

    Returns
    -------
    sepsis_probability : model's predicted probability of sepsis in [0, 1]
    alert              : true when probability >= threshold (0.35)
    threshold          : the decision boundary applied (fixed at 0.35 — lower
                         than 0.5 to favour recall; a missed sepsis case is
                         more harmful than a spurious alert)
    model_version      : opaque identifier for the loaded weights ("v1")

    Raises
    ------
    HTTP 503 if the model failed to load at startup (see GET /health).
    """
    if _state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded — check GET /health for details",
        )

    row  = _build_feature_row(features)
    prob = float(_state["model"].predict_proba(row)[0, 1])

    return PredictionResponse(
        sepsis_probability=round(prob, 6),
        alert=prob >= _ALERT_THRESHOLD,
        threshold=_ALERT_THRESHOLD,
        model_version=_MODEL_VERSION,
    )


@app.get("/features", response_model=FeaturesResponse)
async def feature_list() -> FeaturesResponse:
    """Return the ordered list of feature names expected by POST /predict.

    API consumers can call this endpoint to validate their input schema before
    sending prediction requests.

    If the model is loaded the list reflects its embedded feature_names (the
    authoritative source).  If the model failed to load the list is derived
    statically from VITAL_ITEMIDS / LAB_ITEMIDS, which matches the output
    schema of compute_features() in src/features/feature_engineering.py.

    Returns
    -------
    features : ordered list of feature name strings
    count    : len(features) — convenience field for quick validation
    """
    cols = (
        _state["feature_columns"]
        if _state["feature_columns"]
        else _static_feature_columns()
    )
    return FeaturesResponse(features=cols, count=len(cols))


def _static_feature_columns() -> list[str]:
    """Compute the expected feature column list from module-level itemid constants.

    This is the fallback used by GET /features when the model is not loaded.
    Column order mirrors the output of compute_features() after dropping
    stay_id, window_end, and sepsis_label.
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
