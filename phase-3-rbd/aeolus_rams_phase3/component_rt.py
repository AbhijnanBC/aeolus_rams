"""
aeolus_rams_phase3.component_rt
================================
R(t) callables for all 13 turbine components, constructed from
Phase 2's mtbf_table.csv.

Design decisions
----------------
1. ALL components use the exponential model R(t) = exp(-t / MTBF).
   The Phase 2 AIC analysis supports exponential for every component,
   including Pitch System (AIC_W=172.63 > AIC_exp=172.00, preferred
   field = False for Weibull — confirmed in source column of CSV).
   Section 3.3.3 of the spec confirms: with all exponential models,
   R_system(t) = exp(-λ_total × t), giving a clean closed-form result.

2. NaN-MTBF rows trigger Option A injection (Section 3.0.3):
   Six components have mtbf_days=NaN in Phase 2's output. These are
   assigned literature placeholders from config.PLACEHOLDER_MTBF,
   tagged as `assumed_placeholder`, and flagged via `is_placeholder=True`
   on every ComponentRT object. The pipeline NEVER silently ignores NaN
   — it either injects a placeholder or raises, never returns NaN in a
   series product.

3. MTBF, not MTTF: all Phase 2 outputs are MTBF (Mean Time Between
   Failures) for repairable components, not MTTF. Using 1/MTBF as λ
   directly is the correct operation under an exponential / HPP model
   for a repairable system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger("aeolus_rams_phase3.component_rt")

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComponentRT:
    """Everything Phase 3 needs for one component's reliability model."""
    name: str
    mtbf_days: float
    lambda_per_day: float
    tier: str
    confidence: str
    source: str
    is_placeholder: bool = False
    beta: float | None = None
    eta_days: float | None = None
    ci_low_days: float | None = None
    ci_high_days: float | None = None

    def R(self, t: float | np.ndarray) -> float | np.ndarray:
        """R(t) = exp(-λt). Works on scalar or ndarray t."""
        t_arr = np.asarray(t, dtype=float)
        result = np.exp(-self.lambda_per_day * t_arr)
        return float(result) if t_arr.ndim == 0 else result

    def R_1yr(self) -> float:
        return self.R(config.T_1YR)

    def R_5yr(self) -> float:
        return self.R(config.T_5YR)

    @property
    def uses_aic_preferred_exponential(self) -> bool:
        """True for Pitch System: Phase 2 fitted Weibull (Tier A) but AIC
        preferred exponential. We use the MTBF from the Weibull fit
        directly as the exponential rate, since β≈0.73 gives nearly
        identical MTBF to the exponential fit."""
        return self.tier == "A" and "preferred over exponential: False" in self.source


# ---------------------------------------------------------------------------
# CSV loading + placeholder injection
# ---------------------------------------------------------------------------

def load_mtbf_table(path: str | Path) -> pd.DataFrame:
    """Load Phase 2's mtbf_table.csv with validation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Phase 2 mtbf_table.csv not found at {path.resolve()}.\n"
            "Pass --mtbf-table pointing at "
            "phase-2-weibull-mtbf-hazard/outputs/mtbf_table.csv"
        )
    df = pd.read_csv(path)
    required = {"component", "tier", "mtbf_days", "confidence", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"mtbf_table.csv is missing expected columns: {sorted(missing)}. "
            "This file must be the output of aeolus_rams_phase2."
        )
    return df


def inject_option_a_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """Section 3.0.3 Option A: replace NaN mtbf_days with literature
    placeholders from config.PLACEHOLDER_MTBF. Tags replacement rows with
    confidence='assumed_placeholder' and a new boolean column
    `is_placeholder=True`.

    Raises ValueError if any component has NaN mtbf_days AND no entry
    in PLACEHOLDER_MTBF — that means Phase 3's Option A catalogue is
    incomplete, not that the data is unusable.
    """
    df = df.copy()
    df["is_placeholder"] = False

    nan_rows = df["mtbf_days"].isna()
    for _, row in df[nan_rows].iterrows():
        comp = row["component"]
        if comp not in config.PLACEHOLDER_MTBF:
            raise ValueError(
                f"Component '{comp}' has mtbf_days=NaN in Phase 2's "
                f"mtbf_table.csv AND has no entry in "
                f"config.PLACEHOLDER_MTBF. Add one before running Phase 3."
            )
        prior = config.PLACEHOLDER_MTBF[comp]
        mask = df["component"] == comp
        df.loc[mask, "mtbf_days"] = prior.mtbf_days
        df.loc[mask, "confidence"] = "assumed_placeholder"
        df.loc[mask, "source"] = prior.source
        df.loc[mask, "is_placeholder"] = True
        logger.info(
            "Injected Option A placeholder for %s: MTBF=%.0f days (%s)",
            comp, prior.mtbf_days, prior.source[:60],
        )

    remaining_nan = df["mtbf_days"].isna().sum()
    if remaining_nan:
        bad = df.loc[df["mtbf_days"].isna(), "component"].tolist()
        raise ValueError(
            f"After Option A injection, {remaining_nan} component(s) "
            f"still have NaN mtbf_days: {bad}. "
            "A NaN in the series product silently zeroes R_system(t) — "
            "this is a hard stop."
        )
    return df


# ---------------------------------------------------------------------------
# Building ComponentRT objects
# ---------------------------------------------------------------------------

def _safe_float(val: object) -> float | None:
    try:
        f = float(val)  # type: ignore
        return f if not pd.isna(f) else None
    except (TypeError, ValueError):
        return None


def build_component_rt(row: pd.Series) -> ComponentRT:
    """Construct a ComponentRT from one row of (injected) mtbf_table.csv."""
    mtbf = float(row["mtbf_days"])
    if mtbf <= 0:
        raise ValueError(
            f"Component '{row['component']}' has mtbf_days={mtbf} ≤ 0. "
            "A non-positive MTBF is physically meaningless."
        )
    return ComponentRT(
        name=str(row["component"]),
        mtbf_days=mtbf,
        lambda_per_day=1.0 / mtbf,
        tier=str(row["tier"]),
        confidence=str(row.get("confidence", "unknown")),
        source=str(row.get("source", "")),
        is_placeholder=bool(row.get("is_placeholder", False)),
        beta=_safe_float(row.get("beta")),
        eta_days=_safe_float(row.get("eta")),
        ci_low_days=_safe_float(row.get("ci_low_days")),
        ci_high_days=_safe_float(row.get("ci_high_days")),
    )


def load_all_components(
    path: str | Path,
) -> dict[str, ComponentRT]:
    """
    Master entry point for component_rt.

    Loads the CSV, runs Option A placeholder injection, builds one
    ComponentRT per row, and returns a name-keyed dict. Every value is
    guaranteed to have a finite, positive lambda_per_day — no NaN
    propagation is possible downstream.
    """
    df = load_mtbf_table(path)
    df = inject_option_a_placeholders(df)

    components: dict[str, ComponentRT] = {}
    for _, row in df.iterrows():
        crt = build_component_rt(row)
        components[crt.name] = crt

    n_placeholder = sum(1 for c in components.values() if c.is_placeholder)
    n_aic_exp = sum(1 for c in components.values() if c.uses_aic_preferred_exponential)
    logger.info(
        "Loaded %d components (%d placeholders, %d AIC-preferred exponential)",
        len(components), n_placeholder, n_aic_exp,
    )
    return components


# ---------------------------------------------------------------------------
# Convenience aggregates
# ---------------------------------------------------------------------------

def lambda_per_component(components: dict[str, ComponentRT]) -> dict[str, float]:
    """Return {name: λ} for every component. Used in turbine_rbd.py."""
    return {name: c.lambda_per_day for name, c in components.items()}


def lambda_system(components: dict[str, ComponentRT]) -> float:
    """Aggregate failure rate for the series system: λ_sys = Σ λᵢ.
    Valid under the exponential / HPP model (Section 3.3.3)."""
    return sum(c.lambda_per_day for c in components.values())


def mtbf_system(components: dict[str, ComponentRT]) -> float:
    """MTBF of the series system = 1 / λ_system."""
    return 1.0 / lambda_system(components)


def reliability_table(
    components: dict[str, ComponentRT],
    t_values: tuple[float, ...] = config.MISSION_TIMES_DAYS,
) -> pd.DataFrame:
    """Per-component and system R(t) at every mission time in `t_values`.

    Columns: component, tier, confidence, mtbf_days, is_placeholder,
    R_{t}d for each t.
    """
    rows = []
    for name, c in components.items():
        row: dict[str, object] = {
            "component": name,
            "tier": c.tier,
            "confidence": c.confidence,
            "mtbf_days": c.mtbf_days,
            "is_placeholder": c.is_placeholder,
        }
        for t in t_values:
            row[f"R_{int(t)}d"] = round(c.R(t), 6)
        rows.append(row)
    return pd.DataFrame(rows)
