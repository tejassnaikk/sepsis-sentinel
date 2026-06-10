"""
Feature engineering for SepsisSentinel.

Implements a non-overlapping sliding-window feature matrix over ICU stays.
Each window produces one output row; features are derived from vitals
(chartevents) and labs (labevents) using only measurements strictly before
the window end, preventing any future data leakage.

Window assignment strategy: rather than iterating over windows per stay,
every measurement is tagged with window_idx = floor(elapsed_hours / window_hours).
This is fully vectorised and guarantees charttime < window_end by construction —
a measurement at an exact boundary lands in the *next* window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Itemids to featurize — kept at module level so callers can inspect them
VITAL_ITEMIDS: list[int] = [220045, 220179, 220210, 220277, 223761, 223900]
LAB_ITEMIDS: list[int]   = [50912,  50813,  51301,  50885]

# Number of ICU stays processed per iteration; controls progress granularity and RAM
_BATCH_SIZE = 1000


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_window_grid(
    cohort_batch: pd.DataFrame,
    window_hours: float,
) -> pd.DataFrame:
    """Generate every (stay_id, window_idx) pair for a batch of ICU stays.

    For a stay of duration D hours with window size W, produces window indices
    0 .. floor(D/W)-1. Stays shorter than one window still receive window_idx=0
    so they are never silently dropped.

    Parameters
    ----------
    cohort_batch:
        Subset of cohort rows; must contain stay_id, hadm_id, icu_intime, icu_outtime.
    window_hours:
        Duration of each non-overlapping window in hours.

    Returns
    -------
    DataFrame with columns: stay_id, hadm_id, icu_intime, window_idx,
    window_start, window_end.
    """
    los_hours = (
        (cohort_batch["icu_outtime"] - cohort_batch["icu_intime"])
        .dt.total_seconds()
        / 3600
    )
    # At least one window per stay regardless of LOS
    los_hours = los_hours.fillna(window_hours)
    n_windows = np.maximum(1, np.floor(los_hours / window_hours).astype(int))

    # Vectorised expansion: repeat each stay_id n_windows times, then generate
    # the matching index sequence.  The np.arange loop is over stay counts
    # (O(stays)), not rows, so it is fast even for 50 k-stay cohorts.
    all_stay_ids  = np.repeat(cohort_batch["stay_id"].values, n_windows.values)
    all_window_idx = np.concatenate([np.arange(n) for n in n_windows.values])

    grid = pd.DataFrame({"stay_id": all_stay_ids, "window_idx": all_window_idx})
    grid = grid.merge(
        cohort_batch[["stay_id", "hadm_id", "icu_intime"]],
        on="stay_id",
        how="left",
    )

    # Window boundaries as absolute timestamps
    grid["window_start"] = grid["icu_intime"] + pd.to_timedelta(
        grid["window_idx"] * window_hours, unit="h"
    )
    grid["window_end"] = grid["icu_intime"] + pd.to_timedelta(
        (grid["window_idx"] + 1) * window_hours, unit="h"
    )
    return grid


def _agg_vitals_batch(
    vitals_batch: pd.DataFrame,
    grid: pd.DataFrame,
    window_hours: float,
) -> pd.DataFrame:
    """Aggregate vital signs onto the window grid for one cohort batch.

    For each (stay_id, window_idx, itemid) computes mean, min, max, std, and
    a missing_flag indicating whether any measurement existed in the window.

    Parameters
    ----------
    vitals_batch:
        Vital rows for this batch; columns: stay_id, charttime, itemid, valuenum.
    grid:
        Output of _build_window_grid for the same batch.
    window_hours:
        Must match the value passed to _build_window_grid.

    Returns
    -------
    DataFrame with columns: stay_id, window_idx, vital_{itemid}_{stat}
    for stat in (mean, min, max, std, missing).
    """
    base = grid[["stay_id", "window_idx"]].copy()

    if vitals_batch.empty:
        for iid in VITAL_ITEMIDS:
            for s in ("mean", "min", "max", "std"):
                base[f"vital_{iid}_{s}"] = np.nan
            base[f"vital_{iid}_missing"] = np.int8(1)
        return base

    # Attach icu_intime to each vital row so we can compute elapsed time
    vit = vitals_batch.merge(
        grid[["stay_id", "icu_intime"]].drop_duplicates("stay_id"),
        on="stay_id",
        how="inner",
    )
    # Drop measurements recorded before ICU admission (pre-admission chart carryover)
    vit = vit[vit["charttime"] >= vit["icu_intime"]].copy()

    if vit.empty:
        for iid in VITAL_ITEMIDS:
            for s in ("mean", "min", "max", "std"):
                base[f"vital_{iid}_{s}"] = np.nan
            base[f"vital_{iid}_missing"] = np.int8(1)
        return base

    # Assign window index via floor division of elapsed hours.
    # floor(elapsed / W) < W*(floor+1), so charttime is always < window_end — no leakage.
    elapsed = (vit["charttime"] - vit["icu_intime"]).dt.total_seconds() / 3600
    vit["window_idx"] = (elapsed / window_hours).astype(int)

    # Filter to target itemids (defensive; data should be pre-filtered by loader)
    vit = vit[vit["itemid"].isin(VITAL_ITEMIDS)]

    agg = (
        vit.groupby(["stay_id", "window_idx", "itemid"])["valuenum"]
        .agg(["mean", "min", "max", "std"])
        .reset_index()
    )

    # Pivot to wide format; MultiIndex columns become "vital_{itemid}_{stat}"
    wide = agg.pivot_table(
        index=["stay_id", "window_idx"],
        columns="itemid",
        values=["mean", "min", "max", "std"],
        aggfunc="first",
    )
    wide.columns = [f"vital_{iid}_{stat}" for stat, iid in wide.columns]
    wide = wide.reset_index()

    # Left-join so every window in the grid appears in the output — even
    # windows with no vital measurements (they will have NaN stats)
    result = base.merge(wide, on=["stay_id", "window_idx"], how="left")

    for iid in VITAL_ITEMIDS:
        mean_col = f"vital_{iid}_mean"
        # If this itemid never appeared in the batch, add its columns explicitly
        if mean_col not in result.columns:
            for s in ("mean", "min", "max", "std"):
                result[f"vital_{iid}_{s}"] = np.nan
        result[f"vital_{iid}_missing"] = result[mean_col].isna().astype(np.int8)

    return result


def _agg_labs_batch(
    labs_batch: pd.DataFrame,
    grid: pd.DataFrame,
    window_hours: float,
) -> pd.DataFrame:
    """Aggregate lab values onto the window grid for one cohort batch.

    For each lab itemid the feature is the most recent (highest charttime)
    value strictly before the window end.

    Parameters
    ----------
    labs_batch:
        Lab rows for this batch; columns: hadm_id, charttime, itemid, valuenum.
    grid:
        Output of _build_window_grid for the same batch.
    window_hours:
        Must match the value passed to _build_window_grid.

    Returns
    -------
    DataFrame with columns: stay_id, window_idx, lab_{itemid}_last,
    lab_{itemid}_missing for each itemid in LAB_ITEMIDS.
    """
    base = grid[["stay_id", "window_idx"]].copy()

    # Pre-populate so stays with no labs always get a valid output row
    for iid in LAB_ITEMIDS:
        base[f"lab_{iid}_last"]    = np.nan
        base[f"lab_{iid}_missing"] = np.int8(1)

    if labs_batch.empty:
        return base

    # Labs are indexed by hadm_id; join to pick up stay_id and icu_intime
    lab = labs_batch.merge(
        grid[["hadm_id", "stay_id", "icu_intime"]].drop_duplicates("hadm_id"),
        on="hadm_id",
        how="inner",
    )
    lab = lab[lab["charttime"] >= lab["icu_intime"]].copy()

    if lab.empty:
        return base

    elapsed = (lab["charttime"] - lab["icu_intime"]).dt.total_seconds() / 3600
    lab["window_idx"] = (elapsed / window_hours).astype(int)
    lab = lab[lab["itemid"].isin(LAB_ITEMIDS)]

    # Sort ascending so groupby.last() returns the most recent charttime per window
    lab = lab.sort_values("charttime")
    most_recent = (
        lab.groupby(["stay_id", "window_idx", "itemid"])["valuenum"]
        .last()
        .reset_index()
    )

    lab_wide = most_recent.pivot_table(
        index=["stay_id", "window_idx"],
        columns="itemid",
        values="valuenum",
        aggfunc="last",
    )
    lab_wide.columns = [f"lab_{iid}_last" for iid in lab_wide.columns]
    lab_wide = lab_wide.reset_index()

    # Left-join to get NaN for windows where the itemid was never measured
    merged = base[["stay_id", "window_idx"]].merge(
        lab_wide, on=["stay_id", "window_idx"], how="left"
    )

    # Overwrite the pre-initialised columns with actual values (NaN where absent)
    for iid in LAB_ITEMIDS:
        last_col = f"lab_{iid}_last"
        if last_col in merged.columns:
            # .values ensures positional alignment (merged preserves left-join order)
            base[last_col] = merged[last_col].values
            base[f"lab_{iid}_missing"] = merged[last_col].isna().astype(np.int8)
        # If itemid absent from merged entirely, the initialised NaN/1 remain correct

    return base


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_features(
    vitals_df: pd.DataFrame,
    labs_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    window_hours: int = 6,
) -> pd.DataFrame:
    """Build a windowed feature matrix for sepsis prediction.

    Slides a non-overlapping window of ``window_hours`` across each ICU stay
    in steps of ``window_hours`` and computes vital-sign statistics and most-
    recent lab values for each window.  Output has one row per (stay_id,
    window_end) with no future data leakage: every feature uses only
    measurements with charttime strictly before window_end.

    Parameters
    ----------
    vitals_df:
        Output of load_vitals(); columns: stay_id, charttime, itemid, valuenum.
    labs_df:
        Output of load_labs(); columns: hadm_id, charttime, itemid, valuenum.
    cohort_df:
        Output of load_cohort(); required columns: stay_id, hadm_id,
        icu_intime, icu_outtime, sepsis_label.
    window_hours:
        Duration of each non-overlapping window in hours (default 6).

    Returns
    -------
    DataFrame with one row per (stay_id, window_end) and columns:

    - stay_id, window_end
    - vital_{itemid}_{mean|min|max|std|missing}  for 6 itemids  → 30 cols
    - lab_{itemid}_{last|missing}                for 4 itemids  →  8 cols
    - icu_hours_elapsed  — hours from icu_intime to window_end
    - time_of_day        — hour of day at window_end (0–23)
    - sepsis_label       — carried from cohort_df (same for all windows of a stay)
    """
    # Build fast lookup structures once, outside the batch loop
    vitals_by_stay: dict[int, pd.DataFrame] = {
        sid: grp for sid, grp in vitals_df.groupby("stay_id", sort=False)
    }
    labs_by_hadm: dict[int, pd.DataFrame] = {
        hid: grp for hid, grp in labs_df.groupby("hadm_id", sort=False)
    }
    stay_to_hadm: dict[int, int] = cohort_df.set_index("stay_id")["hadm_id"].to_dict()

    # Verify required cohort columns are present before starting the batch loop
    required_cols = {"stay_id", "hadm_id", "icu_intime", "icu_outtime", "sepsis_label"}
    missing_cols  = required_cols - set(cohort_df.columns)
    if missing_cols:
        raise ValueError(
            f"cohort_df is missing required columns: {sorted(missing_cols)}\n"
            "Ensure cohort_df is the direct output of load_cohort()."
        )

    stay_ids = cohort_df["stay_id"].tolist()
    n_total  = len(stay_ids)
    all_batches: list[pd.DataFrame] = []

    print(f"Computing features for {n_total:,} stays "
          f"(window={window_hours}h, batch_size={_BATCH_SIZE})...")

    for batch_start in range(0, n_total, _BATCH_SIZE):
        batch_ids = stay_ids[batch_start : batch_start + _BATCH_SIZE]

        cohort_batch = cohort_df[cohort_df["stay_id"].isin(batch_ids)]

        # Collect vitals rows for this batch's stays
        vital_frames = [vitals_by_stay[sid] for sid in batch_ids if sid in vitals_by_stay]
        vitals_batch = (
            pd.concat(vital_frames, ignore_index=True)
            if vital_frames
            else pd.DataFrame(columns=["stay_id", "charttime", "itemid", "valuenum"])
        )

        # Collect lab rows; labs are keyed by hadm_id, not stay_id
        hadm_ids_batch = [stay_to_hadm[sid] for sid in batch_ids if sid in stay_to_hadm]
        lab_frames = [labs_by_hadm[hid] for hid in hadm_ids_batch if hid in labs_by_hadm]
        labs_batch = (
            pd.concat(lab_frames, ignore_index=True)
            if lab_frames
            else pd.DataFrame(columns=["hadm_id", "charttime", "itemid", "valuenum"])
        )

        grid = _build_window_grid(cohort_batch, window_hours)

        vital_feats = _agg_vitals_batch(vitals_batch, grid, window_hours)
        lab_feats   = _agg_labs_batch(labs_batch,   grid, window_hours)

        # Merge vitals and labs on (stay_id, window_idx)
        batch_features = vital_feats.merge(
            lab_feats, on=["stay_id", "window_idx"], how="outer"
        )
        # Reattach window_end for temporal features (dropped after aggregation)
        batch_features = batch_features.merge(
            grid[["stay_id", "window_idx", "window_end"]],
            on=["stay_id", "window_idx"],
            how="left",
        )

        all_batches.append(batch_features)

        n_done = min(batch_start + _BATCH_SIZE, n_total)
        print(f"  Processed {n_done:,} / {n_total:,} stays...")

    print("Assembling final feature matrix...")
    features = pd.concat(all_batches, ignore_index=True)

    # Temporal features — computed after concat so we do one pass
    features = features.merge(
        cohort_df[["stay_id", "icu_intime", "sepsis_label"]],
        on="stay_id",
        how="left",
    )
    features["icu_hours_elapsed"] = (
        (features["window_end"] - features["icu_intime"]).dt.total_seconds() / 3600
    )
    features["time_of_day"] = features["window_end"].dt.hour

    features.drop(columns=["window_idx", "icu_intime"], inplace=True)

    # Enforce canonical column order for downstream reproducibility
    fixed_cols = ["stay_id", "window_end"]
    vital_cols = [
        f"vital_{iid}_{s}"
        for iid in VITAL_ITEMIDS
        for s in ("mean", "min", "max", "std", "missing")
    ]
    lab_cols = [
        f"lab_{iid}_{s}"
        for iid in LAB_ITEMIDS
        for s in ("last", "missing")
    ]
    tail_cols = ["icu_hours_elapsed", "time_of_day", "sepsis_label"]
    ordered = [c for c in fixed_cols + vital_cols + lab_cols + tail_cols if c in features.columns]
    features = features[ordered]

    print(
        f"Feature matrix complete: {len(features):,} rows × {len(features.columns)} columns "
        f"across {features['stay_id'].nunique():,} stays"
    )
    return features
