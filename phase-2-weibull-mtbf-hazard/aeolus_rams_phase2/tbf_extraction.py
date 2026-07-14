"""
aeolus_rams_phase2.tbf_extraction
=====================================
Section 2.3 — Time-Between-Failures (TBF) Extraction.

Groups a component's linked incidents by `(farm, asset_id)` — the correct
compound key, since CARE's `asset_id` is anonymised per farm and is not
globally unique (see `linkage.link_all_farms`) — sorts chronologically
WITHIN each asset (relative ordering only, per Section 2.1.2), and
computes time-between-failures in days.

The final gap for each asset (from its last observed incident to the end
of its observation window) is RIGHT-CENSORED: the asset didn't fail
again, it just ran out of observed time. It is flagged (`censored=True`),
never silently folded into a completed-interval average.

ENHANCEMENT: With the improved linkage module (status filtering + sensor
tie-breakers), remaining ambiguous matches are now excluded from TBF
extraction to maintain data quality.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("aeolus_rams_phase2.tbf_extraction")


def extract_tbf(
    linked_events: pd.DataFrame,
    component: str,
    asset_observation_end: dict[tuple[str, object], pd.Timestamp],
    asset_observation_start: dict[tuple[str, object], pd.Timestamp] | None = None,
    component_col: str = "component_primary",
    event_label: str = "anomaly",
    event_label_col: str = "event_label",
    exclude_ambiguous: bool = True,
) -> pd.DataFrame:
    """Time-between-failures for one component, across every asset it was
    observed on.

    Rows with `asset_id is None` (failed linkage — Section 2.1.1's
    `no_match`/unresolved rows) are excluded up front and logged, rather
    than silently corrupting a chronological ordering with an unlinked
    incident.
    
    With exclude_ambiguous=True (default), rows with link_confidence
    starting with "ambiguous_" are also excluded and logged. The enhanced
    linkage module (with status filtering and sensor tie-breakers)
    dramatically reduces ambiguity, but any remaining ambiguous matches
    are too uncertain to include in TBF analysis.
    """
    subset = linked_events[
        (linked_events[component_col] == component)
        & (linked_events[event_label_col] == event_label)
    ].copy()

    unlinked = subset["asset_id"].isna().sum()
    if unlinked:
        logger.warning(
            "%s: dropping %d/%d incident(s) with no resolved asset_id "
            "before TBF extraction — fix the linkage first, don't let "
            "these silently vanish from the incident count elsewhere.",
            component, unlinked, len(subset),
        )
    subset = subset.dropna(subset=["asset_id"])
    
    # BUG FIX: Capture all assets that had an event (before we drop ambiguous ones)
    # This prevents the "false healthy asset" trap: an asset with an ambiguous event
    # that we drop should NOT be counted as "perfectly healthy" later.
    assets_with_any_event = set(subset[["farm", "asset_id"]].itertuples(index=False, name=None))
    
    # Also filter out truly ambiguous matches if requested
    # NOTE: Accept both 'unique_match' AND sensor-resolved ambiguous matches
    # (e.g., 'ambiguous_4_matches_sensor_resolved'). Only drop unresolved ambiguities.
    if exclude_ambiguous and "link_confidence" in subset.columns:
        # Keep unique matches and resolved ambiguous matches (both sensor and timespan resolved); drop unresolved ambiguous
        unresolved_ambiguous = (
            subset["link_confidence"].astype(str).str.startswith("ambiguous_") &
            ~subset["link_confidence"].astype(str).str.endswith("resolved")
        )
        n_ambiguous = unresolved_ambiguous.sum()
        if n_ambiguous > 0:
            logger.warning(
                "%s: excluding %d/%d incident(s) with unresolved ambiguous linkage. "
                "Keeping sensor-resolved ambiguous matches (e.g., 'ambiguous_*_sensor_resolved') "
                "and unique matches, which are trustworthy for TBF.",
                component, n_ambiguous, len(subset),
            )
            subset = subset[~unresolved_ambiguous]
    
    # BUG FIX: Allow 0-failure components through — the healthy asset loop handles them
    # If subset becomes empty after dropping ambiguous, still process it to capture
    # right-censored observations for all assets (including those with 0 linkable failures).

    if not subset.empty:
        subset = subset.sort_values(["farm", "asset_id", "event_start"])

    rows = []
    # Track which assets had failures for this component (to identify healthy assets later)
    assets_with_failures = set()
    
    for (farm, asset_id), group in subset.groupby(["farm", "asset_id"]) if not subset.empty else []:
        assets_with_failures.add((farm, asset_id))
        starts = group["event_start"].tolist()

        # ENHANCEMENT: Add "time to first failure" interval (uncensored)
        # This is standard in reliability engineering: the interval from asset
        # commissioning (observation_start) to the first failure is a real,
        # uncensored TBF interval. Including this dramatically improves sample
        # size for small-failure components (e.g., Pitch 4→11 intervals).
        if len(starts) > 0 and asset_observation_start:
            obs_start = asset_observation_start.get((farm, asset_id))
            if obs_start is not None and obs_start < starts[0]:
                ttf_days = (starts[0] - obs_start).total_seconds() / 86400
                rows.append({
                    "farm": farm,
                    "asset_id": asset_id,
                    "component": component,
                    "tbf_days": ttf_days,
                    "censored": False,  # Uncensored: we observed the actual first failure
                    "interval_start": obs_start,
                    "interval_end": starts[0],
                    "interval_type": "time_to_first_failure",
                })
        
        # Inter-failure intervals (between consecutive failures)
        for i in range(1, len(starts)):
            rows.append({
                "farm": farm,
                "asset_id": asset_id,
                "component": component,
                "tbf_days": (starts[i] - starts[i - 1]).total_seconds() / 86400,
                "censored": False,
                "interval_start": starts[i - 1],
                "interval_end": starts[i],
                "interval_type": "inter_failure",
            })

        # Right-censored final interval (from last failure to observation end)
        obs_end = asset_observation_end.get((farm, asset_id))
        if obs_end is None:
            logger.warning(
                "%s: no observation-end timestamp found for (farm=%s, "
                "asset_id=%s) — final interval left un-censored-flagged "
                "and OMITTED rather than guessed.", component, farm, asset_id,
            )
            continue
        if obs_end > starts[-1]:
            rows.append({
                "farm": farm,
                "asset_id": asset_id,
                "component": component,
                "tbf_days": (obs_end - starts[-1]).total_seconds() / 86400,
                "censored": True,
                "interval_start": starts[-1],
                "interval_end": obs_end,
                "interval_type": "censored_at_observation_end",
            })

    # CRITICAL FIX: Include healthy assets (zero failures) as right-censored observations
    # In survival analysis, an asset that never fails is crucial data: it's a right-censored
    # observation that proves the component survived the entire observation window.
    # Omitting healthy assets biases the distribution fit pessimistically (artificially
    # suppresses Scale/η and MTBF). This fix ensures all assets contribute, not just failures.
    # Note: Use assets_with_any_event (not assets_with_failures) to avoid the "false healthy asset"
    # trap: an asset with an ambiguous event we dropped should not be marked as perfectly healthy.
    if asset_observation_end and asset_observation_start:
        for (farm, asset_id) in asset_observation_end.keys():
            if (farm, asset_id) not in assets_with_any_event:
                # Healthy asset: no failures for this component during observation window
                obs_start = asset_observation_start.get((farm, asset_id))
                obs_end = asset_observation_end.get((farm, asset_id))
                if obs_start is not None and obs_end is not None and obs_end >= obs_start:
                    censored_interval_days = (obs_end - obs_start).total_seconds() / 86400
                    rows.append({
                        "farm": farm,
                        "asset_id": asset_id,
                        "component": component,
                        "tbf_days": censored_interval_days,
                        "censored": True,  # Survived entire observation window
                        "interval_start": obs_start,
                        "interval_end": obs_end,
                        "interval_type": "zero_failures_censored",
                    })

    result = pd.DataFrame(rows)
    if not result.empty and (result["tbf_days"] < 0).any():
        n_bad = int((result["tbf_days"] < 0).sum())
        logger.warning(
            "%s: %d interval(s) have a NEGATIVE tbf_days — almost always "
            "a linkage or observation-window bug (an event starting after "
            "its own asset's last observed timestamp). Investigate before "
            "trusting this component's fit.", component, n_bad,
        )
    return result


def extract_tbf_all_components(
    linked_events: pd.DataFrame,
    components: list[str],
    asset_observation_end: dict[tuple[str, object], pd.Timestamp],
    exclude_ambiguous: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Convenience: run `extract_tbf` for every component in `components`
    and concatenate into one long table.
    
    Parameters
    ----------
    exclude_ambiguous : bool, default=True
        Pass through to extract_tbf. If True, ambiguous-confidence rows
        are excluded from TBF calculation.
    """
    frames = [
        extract_tbf(
            linked_events, c, asset_observation_end,
            exclude_ambiguous=exclude_ambiguous, **kwargs
        )
        for c in components
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=[
            "farm", "asset_id", "component", "tbf_days", "censored",
            "interval_start", "interval_end", "interval_type",
        ])
    return pd.concat(frames, ignore_index=True)


def uncensored_tbf_days(tbf_table: pd.DataFrame, component: str | None = None) -> np.ndarray:
    """The array of COMPLETE (non-censored) tbf_days values a fitting
    routine is allowed to use as i.i.d. samples. Naive averaging or MLE
    fitting must never include censored rows as-is (Section 2.3's own
    warning) — this is the single choke point that enforces it."""
    df = tbf_table
    if component is not None:
        df = df[df["component"] == component]
    return df.loc[~df["censored"], "tbf_days"].to_numpy(dtype=float)


def tbf_summary(tbf_table: pd.DataFrame) -> pd.DataFrame:
    """Per-component summary: how many usable (uncensored) intervals vs.
    censored tails were actually extracted — this is the REAL number
    Section 2.4's fitting minimums must be checked against, not the raw
    incident count from Phase 1's fmeca_table.csv (see `config.py` and
    `pipeline.py`'s tier-revalidation step)."""
    if tbf_table.empty:
        return pd.DataFrame(columns=[
            "component", "n_uncensored", "n_censored", "n_assets",
        ])

    rows = []
    for component, group in tbf_table.groupby("component"):
        rows.append({
            "component": component,
            "n_uncensored": int((~group["censored"]).sum()),
            "n_censored": int(group["censored"].sum()),
            "n_assets": group["asset_id"].nunique(),
        })
    return pd.DataFrame(rows).sort_values("n_uncensored", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Failure Mode Grouping (Section 2.3, Enhancement)
# ---------------------------------------------------------------------------

def group_by_failure_mode(
    linked_events: pd.DataFrame,
    component: str,
    component_col: str = "component_primary",
    failure_mode_col: str = "component_secondary",
    event_label: str = "anomaly",
    event_label_col: str = "event_label",
) -> dict[str | None, pd.DataFrame]:
    """Group incidents for a component by their failure mode/cause.

    **Problem:** Treating all "Gearbox" failures as one population masks
    important differences. Gearbox bearing wear (β ≈ 1.8, accelerating)
    vs. Gearbox seal leak (β ≈ 1.2, mild wear) vs. sudden tooth breakage
    (β ≈ 1.0, random) have DIFFERENT hazard curves and DIFFERENT MTBF
    predictions.

    **Solution:** Extract TBF separately for each observed failure mode
    (identified by component_secondary, if available), allowing
    independent shape/scale fits. This preserves physics-based distinctions
    the component taxonomy already captures.

    Parameters
    ----------
    linked_events : pd.DataFrame
        Output from linkage module (must have asset_id, component_primary,
        component_secondary, event_start, event_end, etc.).
    component : str
        Parent component to group (e.g., "Gearbox").
    component_col : str
        Column name for primary component (default "component_primary").
    failure_mode_col : str
        Column name for secondary/failure-mode detail (default
        "component_secondary"). If empty, all incidents are grouped as
        "unspecified".
    event_label : str
        Filter to only these event types (default "anomaly").
    event_label_col : str
        Column name for event type (default "event_label").

    Returns
    -------
    dict[str | None, pd.DataFrame]
        Keys: failure mode names (or None for unspecified).
        Values: subset of linked_events for that mode.
    """
    subset = linked_events[
        (linked_events[component_col] == component)
        & (linked_events[event_label_col] == event_label)
    ].copy()

    if subset.empty:
        return {}

    # Group by failure mode (component_secondary), using None for empty/missing
    modes = {}
    for mode, group in subset.groupby(failure_mode_col, dropna=False):
        # Treat NaN / None as a single "unspecified" group
        mode_key = None if pd.isna(mode) else str(mode)
        modes[mode_key] = group.copy()

    return modes


def extract_tbf_by_failure_mode(
    linked_events: pd.DataFrame,
    component: str,
    asset_observation_end: dict[tuple[str, object], pd.Timestamp],
    exclude_ambiguous: bool = True,
    asset_observation_start: dict[tuple[str, object], pd.Timestamp] | None = None,
    **kwargs,
) -> dict[str | None, pd.DataFrame]:
    """Extract time-between-failures separately for each failure mode of
    a component.

    Returns a dict where each key is a failure mode (or None for
    unspecified), and each value is a TBF table for that mode (same
    schema as extract_tbf, with "failure_mode" column added).

    This enables mode-specific Weibull fitting: e.g., Gearbox bearing
    wear might have β=1.8 vs. Gearbox seal leak with β=1.2.
    """
    modes = group_by_failure_mode(linked_events, component, **kwargs)

    result = {}
    for mode_key, mode_events in modes.items():
        if mode_events.empty:
            continue

        # Extract TBF for this mode (BUG FIX: forward asset_observation_start for time-to-first-failure)
        mode_tbf = extract_tbf(
            mode_events, component, asset_observation_end,
            asset_observation_start=asset_observation_start,
            exclude_ambiguous=exclude_ambiguous,
        )

        if not mode_tbf.empty:
            # Add failure mode column
            mode_tbf["failure_mode"] = (
                "unspecified" if mode_key is None else mode_key
            )
            result[mode_key] = mode_tbf

            n_uncensored = (~mode_tbf["censored"]).sum()
            logger.info(
                "%s → %s: extracted %d uncensored TBF intervals",
                component, mode_key, n_uncensored,
            )

    return result


def tbf_summary_by_mode(
    tbf_by_mode: dict[str | None, pd.DataFrame],
    component: str,
) -> pd.DataFrame:
    """Summary statistics for each failure mode of a component."""
    rows = []
    for mode_key, mode_tbf in tbf_by_mode.items():
        mode_label = "unspecified" if mode_key is None else str(mode_key)
        rows.append({
            "component": component,
            "failure_mode": mode_label,
            "n_uncensored": int((~mode_tbf["censored"]).sum()),
            "n_censored": int(mode_tbf["censored"].sum()),
            "n_assets": mode_tbf["asset_id"].nunique(),
            "mean_tbf_days": float(mode_tbf.loc[~mode_tbf["censored"], "tbf_days"].mean())
            if (~mode_tbf["censored"]).any() else np.nan,
        })
    return pd.DataFrame(rows)
