"""
aeolus_rams_phase2.tiering
=============================
Section 2.2 — Component Data-Sufficiency Tiering.

Reads Phase 1's `fmeca_table.csv` directly rather than hardcoding the
worked example in Section 2.2's table — this keeps Phase 2 automatically
consistent if Phase 1 is ever re-run with corrected Severity/Detection
scores or a fresh CARE download.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def load_fmeca_table(path: str | Path) -> pd.DataFrame:
    """Load Phase 1's fmeca_table.csv. Raises a clear error (not a bare
    pandas KeyError downstream) if the expected incident-count column is
    missing — almost always means `path` points at the wrong file or an
    older Phase 1 schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Phase 1 fmeca_table.csv not found at {path.resolve()}. "
            "Pass --phase1-dir pointing at the folder that contains it."
        )
    df = pd.read_csv(path)
    if config.INCIDENT_COUNT_COLUMN not in df.columns:
        raise ValueError(
            f"{path} has no '{config.INCIDENT_COUNT_COLUMN}' column — is "
            "this really a Phase 1 fmeca_table.csv?"
        )
    return df


def assign_tier(n_incidents: int, thresholds: config.TierThresholds = config.TIER_THRESHOLDS) -> str:
    """Section 2.2's rule of thumb:
    >= min_a incidents  -> Tier A (2-parameter Weibull)
    min_b..min_a-1       -> Tier B (exponential only)
    < min_b               -> Tier C (literature-informed placeholder)
    """
    if n_incidents < 0:
        raise ValueError("n_incidents must be >= 0")
    if n_incidents >= thresholds.min_a:
        return config.TIER_A
    if n_incidents >= thresholds.min_b:
        return config.TIER_B
    return config.TIER_C


def tier_table(
    fmeca_df: pd.DataFrame,
    incidents_col: str = config.INCIDENT_COUNT_COLUMN,
    thresholds: config.TierThresholds = config.TIER_THRESHOLDS,
) -> pd.DataFrame:
    """Phase 1's fmeca_table.csv, with a `tier` column added. Preserves
    every other column (rank, RPN, severity, etc.) so this can be used
    directly as Phase 2's own reporting input."""
    out = fmeca_df.copy()
    out["tier"] = out[incidents_col].apply(lambda n: assign_tier(int(n), thresholds))
    out["tier_description"] = out["tier"].map(config.TIER_DESCRIPTIONS)
    return out


def components_by_tier(tiered_df: pd.DataFrame, component_col: str = "component") -> dict[str, list[str]]:
    """`{"A": [...], "B": [...], "C": [...]}` component-name lists."""
    return {
        tier: tiered_df.loc[tiered_df["tier"] == tier, component_col].tolist()
        for tier in (config.TIER_A, config.TIER_B, config.TIER_C)
    }


def tier_summary_counts(tiered_df: pd.DataFrame) -> pd.Series:
    """How many components landed in each tier — Section 2.2's headline
    number ("eleven of thirteen components landing in Tier C ... is
    exactly the honest, expected shape of the result")."""
    return tiered_df["tier"].value_counts().reindex(
        [config.TIER_A, config.TIER_B, config.TIER_C], fill_value=0
    )
