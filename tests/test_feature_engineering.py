"""
Tests for src/features/feature_engineering.py.

Each test is fully self-contained: it builds its own minimal cohort, vitals,
and labs DataFrames with exactly the columns that compute_features() expects,
then makes targeted assertions about one specific property of the output.

No MIMIC files are read; no network calls are made.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import (
    LAB_ITEMIDS,
    VITAL_ITEMIDS,
    compute_features,
)

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

_T0 = pd.Timestamp("2020-01-01 00:00")


def make_cohort(
    stay_id: int,
    hadm_id: int,
    los_hours: float,
    sepsis_label: int = 0,
    t0: pd.Timestamp = _T0,
) -> pd.DataFrame:
    """Return a one-row cohort DataFrame matching the schema of load_cohort().

    Parameters
    ----------
    stay_id, hadm_id:
        Identifiers for this ICU stay / hospital admission.
    los_hours:
        Length of ICU stay in hours; determines how many windows are produced.
    sepsis_label:
        Binary sepsis label (0 or 1) to carry through to all window rows.
    t0:
        ICU admission timestamp.
    """
    return pd.DataFrame(
        {
            "subject_id":    [stay_id * 10],          # arbitrary subject
            "hadm_id":       [hadm_id],
            "stay_id":       [stay_id],
            "gender":        ["M"],
            "anchor_age":    [55],
            "admittime":     [t0 - pd.Timedelta(hours=2)],
            "icu_intime":    [t0],
            "icu_outtime":   [t0 + pd.Timedelta(hours=los_hours)],
            "los_icu_hours": [los_hours],
            "sepsis_label":  [sepsis_label],
        }
    )


def empty_vitals() -> pd.DataFrame:
    """Return an empty vitals DataFrame with the correct column schema."""
    return pd.DataFrame(columns=["stay_id", "charttime", "itemid", "valuenum"])


def empty_labs() -> pd.DataFrame:
    """Return an empty labs DataFrame with the correct column schema."""
    return pd.DataFrame(columns=["hadm_id", "charttime", "itemid", "valuenum"])


def make_vitals(
    stay_id: int,
    hour_offsets: list[float],
    itemids: list[int] | None = None,
    value: float = 80.0,
    t0: pd.Timestamp = _T0,
) -> pd.DataFrame:
    """Build a vitals DataFrame with one measurement per (hour, itemid) pair.

    Parameters
    ----------
    stay_id:
        The ICU stay these measurements belong to.
    hour_offsets:
        Hours after t0 at which measurements are charted.
    itemids:
        Subset of VITAL_ITEMIDS to generate; defaults to all six.
    value:
        Constant valuenum to assign every row (makes mean/min/max trivially equal).
    t0:
        ICU admission timestamp.
    """
    if itemids is None:
        itemids = VITAL_ITEMIDS
    rows = [
        {
            "stay_id":   stay_id,
            "charttime": t0 + pd.Timedelta(hours=h),
            "itemid":    iid,
            "valuenum":  value,
        }
        for h in hour_offsets
        for iid in itemids
    ]
    return pd.DataFrame(rows)


def make_labs(
    hadm_id: int,
    hour_offsets: list[float],
    itemids: list[int] | None = None,
    value: float = 1.5,
    t0: pd.Timestamp = _T0,
) -> pd.DataFrame:
    """Build a labs DataFrame with one result per (hour, itemid) pair.

    Parameters
    ----------
    hadm_id:
        Hospital admission these results belong to.
    hour_offsets:
        Hours after t0 at which results are charted.
    itemids:
        Subset of LAB_ITEMIDS to generate; defaults to all four.
    value:
        Constant valuenum for every row.
    t0:
        ICU admission timestamp.
    """
    if itemids is None:
        itemids = LAB_ITEMIDS
    rows = [
        {
            "hadm_id":   hadm_id,
            "charttime": t0 + pd.Timedelta(hours=h),
            "itemid":    iid,
            "valuenum":  value,
        }
        for h in hour_offsets
        for iid in itemids
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_correct_row_count():
    """compute_features() produces the right number of window rows per stay.

    A 13-hour stay with window=6 fits two full 6-hour windows (floor(13/6)=2).
    A 3-hour stay is shorter than one window but still receives exactly 1 row
    because the implementation guarantees a minimum of one window per stay.
    """
    cohort_long  = make_cohort(stay_id=1, hadm_id=10, los_hours=13)
    cohort_short = make_cohort(stay_id=2, hadm_id=20, los_hours=3)
    cohort = pd.concat([cohort_long, cohort_short], ignore_index=True)

    result = compute_features(empty_vitals(), empty_labs(), cohort, window_hours=6)

    rows_stay1 = result[result["stay_id"] == 1]
    rows_stay2 = result[result["stay_id"] == 2]

    assert len(rows_stay1) == 2, (
        f"13-hour stay with window=6 should produce 2 rows, got {len(rows_stay1)}"
    )
    assert len(rows_stay2) == 1, (
        f"3-hour stay shorter than one window should still produce 1 row, got {len(rows_stay2)}"
    )


def test_no_future_leakage():
    """compute_features() never uses measurements from after a window's end time.

    Setup: one stay with heart-rate measurements at hours 1, 3, 5, 7, 9, 11.
    With window=6:
      - Window 0 ends at hour 6:  only hours 1, 3, 5 are visible → mean = value
      - Window 1 ends at hour 12: only hours 7, 9, 11 are visible → mean = value

    Both windows use the same constant value, so the means should be identical
    to that value.  The assertion that each window's mean equals the known
    constant (computed from only its three measurements) proves no leakage:
    if measurements crossed window boundaries the means would still equal the
    constant (same value everywhere), so we also assert the window_end
    timestamps are correct to confirm the windowing structure itself is right.
    """
    VALUE_W0 = 60.0  # value for measurements in window 0 (hours 1, 3, 5)
    VALUE_W1 = 90.0  # different value for window 1 (hours 7, 9, 11)

    cohort = make_cohort(stay_id=1, hadm_id=10, los_hours=12)

    # Two distinct constant values so leakage would corrupt the mean
    rows_w0 = [
        {"stay_id": 1, "charttime": _T0 + pd.Timedelta(hours=h),
         "itemid": 220045, "valuenum": VALUE_W0}
        for h in [1, 3, 5]
    ]
    rows_w1 = [
        {"stay_id": 1, "charttime": _T0 + pd.Timedelta(hours=h),
         "itemid": 220045, "valuenum": VALUE_W1}
        for h in [7, 9, 11]
    ]
    vitals = pd.DataFrame(rows_w0 + rows_w1)
    labs   = empty_labs()

    result = compute_features(vitals, labs, cohort, window_hours=6)

    w0_end = _T0 + pd.Timedelta(hours=6)
    w1_end = _T0 + pd.Timedelta(hours=12)

    w0 = result[result["window_end"] == w0_end]
    w1 = result[result["window_end"] == w1_end]

    assert len(w0) == 1, f"Expected exactly one row for window 0, got {len(w0)}"
    assert len(w1) == 1, f"Expected exactly one row for window 1, got {len(w1)}"

    mean_w0 = w0["vital_220045_mean"].iloc[0]
    mean_w1 = w1["vital_220045_mean"].iloc[0]

    assert mean_w0 == pytest.approx(VALUE_W0), (
        f"Window 0 mean should be {VALUE_W0} (hours 1,3,5 only) but got {mean_w0}. "
        "Window 1 measurements (hours 7,9,11) must not leak into window 0."
    )
    assert mean_w1 == pytest.approx(VALUE_W1), (
        f"Window 1 mean should be {VALUE_W1} (hours 7,9,11 only) but got {mean_w1}. "
        "Window 0 measurements (hours 1,3,5) must not persist into window 1."
    )


def test_missing_flags_when_no_vitals():
    """All vital_*_missing columns equal 1 when a stay has zero vital measurements.

    A stay with no chartevents rows is valid in MIMIC (the patient may not have
    had any charted observations in the study window).  The feature matrix must
    still contain a row for that stay; all vital_*_missing flags must be 1 to
    signal the absence of data, and the corresponding stat columns (mean, min,
    max, std) must be NaN rather than zero.
    """
    cohort = make_cohort(stay_id=1, hadm_id=10, los_hours=6)
    vitals = empty_vitals()
    labs   = make_labs(hadm_id=10, hour_offsets=[2, 4])  # labs present; vitals absent

    result = compute_features(vitals, labs, cohort, window_hours=6)

    assert len(result) == 1, "Stay should still produce one row even with zero vitals"

    missing_cols = [f"vital_{iid}_missing" for iid in VITAL_ITEMIDS]
    stat_cols    = [
        f"vital_{iid}_{s}"
        for iid in VITAL_ITEMIDS
        for s in ("mean", "min", "max", "std")
    ]

    for col in missing_cols:
        val = result[col].iloc[0]
        assert val == 1, (
            f"{col} should be 1 when no vitals are present, got {val}"
        )

    for col in stat_cols:
        val = result[col].iloc[0]
        assert pd.isna(val), (
            f"{col} should be NaN when no vitals are present, got {val}"
        )


def test_missing_flags_when_no_labs():
    """All lab_*_missing columns equal 1 when a stay has zero lab results.

    Analogous to test_missing_flags_when_no_vitals for the lab feature block.
    The lab_*_last columns should be NaN when no results exist in the window.
    """
    cohort = make_cohort(stay_id=1, hadm_id=10, los_hours=6)
    vitals = make_vitals(stay_id=1, hour_offsets=[1, 3, 5])  # vitals present; labs absent
    labs   = empty_labs()

    result = compute_features(vitals, labs, cohort, window_hours=6)

    assert len(result) == 1, "Stay should still produce one row even with zero labs"

    missing_cols = [f"lab_{iid}_missing" for iid in LAB_ITEMIDS]
    last_cols    = [f"lab_{iid}_last"    for iid in LAB_ITEMIDS]

    for col in missing_cols:
        val = result[col].iloc[0]
        assert val == 1, (
            f"{col} should be 1 when no labs are present, got {val}"
        )

    for col in last_cols:
        val = result[col].iloc[0]
        assert pd.isna(val), (
            f"{col} should be NaN when no labs are present, got {val}"
        )


def test_sepsis_label_carried_through():
    """sepsis_label from cohort_df appears on every window row for that stay.

    The label is a stay-level attribute, not a window-level one.  Every row
    produced for a given stay must carry the same label as the cohort entry,
    regardless of how many windows the stay spans.  This test uses a multi-
    window stay (13 hours → 2 windows) and checks both a positive (label=1)
    and a negative (label=0) stay to rule out accidental cross-contamination.
    """
    cohort = pd.concat(
        [
            make_cohort(stay_id=1, hadm_id=10, los_hours=13, sepsis_label=1),
            make_cohort(stay_id=2, hadm_id=20, los_hours=13, sepsis_label=0),
        ],
        ignore_index=True,
    )
    vitals = empty_vitals()
    labs   = empty_labs()

    result = compute_features(vitals, labs, cohort, window_hours=6)

    rows_s1 = result[result["stay_id"] == 1]
    rows_s2 = result[result["stay_id"] == 2]

    assert len(rows_s1) == 2, (
        f"Stay 1 (13 h, window=6) should produce 2 rows, got {len(rows_s1)}"
    )
    assert len(rows_s2) == 2, (
        f"Stay 2 (13 h, window=6) should produce 2 rows, got {len(rows_s2)}"
    )

    for idx, row in rows_s1.iterrows():
        assert row["sepsis_label"] == 1, (
            f"Stay 1 row at window_end={row['window_end']} should have sepsis_label=1, "
            f"got {row['sepsis_label']}"
        )

    for idx, row in rows_s2.iterrows():
        assert row["sepsis_label"] == 0, (
            f"Stay 2 row at window_end={row['window_end']} should have sepsis_label=0, "
            f"got {row['sepsis_label']}"
        )
