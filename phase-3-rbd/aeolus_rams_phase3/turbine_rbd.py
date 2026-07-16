"""
aeolus_rams_phase3.turbine_rbd
================================
Section 3.3.3 — Analytical turbine-level R_system(t).

For a k-component series system under the exponential/HPP model:

    R_system(t) = ∏ᵢ exp(-λᵢ t) = exp(-λ_system × t)

    where λ_system = Σᵢ (1 / MTBF_i)

This closed-form solution is exact under the assumption that every
component failure time is exponentially distributed (confirmed by
Phase 2's AIC analysis) and failures are statistically independent
(standard reliability engineering assumption absent shared-cause
data, justified here because CARE's per-turbine SCADA isolation
means we can observe component-level independence).

Expected numbers from real Phase 2 data (13 components including
Option A placeholders):
  λ_system  ≈ 0.002184 / day  (≈ 0.798 / year)
  MTBF_sys  ≈ 458 days        (≈ 1.25 years)
  R(1yr)    ≈ 0.451
  R(5yr)    ≈ 0.019

The low 5-year system reliability is dominated by Pitch System
(λ=0.000517/day, 23.7% of λ_total) and Hydraulic System
(λ=0.000542/day, 24.8% of λ_total). This is the central structural
finding Phase 3's importance analysis formalises.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .component_rt import ComponentRT, lambda_system as _lambda_system


# ---------------------------------------------------------------------------
# Core analytical functions
# ---------------------------------------------------------------------------

def R_turbine_series(
    t: float | np.ndarray,
    components: dict[str, ComponentRT],
) -> np.ndarray:
    """Series-system reliability: product of all component R(t).

    Mathematically equivalent to exp(-λ_system × t) under the
    exponential model, but computed as a product so any deviation
    from exact exponential form remains detectable in tests.
    """
    t_arr = np.asarray(t, dtype=float)
    R = np.ones_like(t_arr)
    for comp in components.values():
        R *= comp.R(t_arr)
    return R


def R_turbine_exponential(
    t: float | np.ndarray,
    lam_sys: float,
) -> np.ndarray:
    """Closed-form series reliability: exp(-λ_system × t).

    Faster than the product form and numerically identical for
    pure-exponential components — used for the sensitivity sweep
    and farm-level calculations.
    """
    t_arr = np.asarray(t, dtype=float)
    return np.exp(-lam_sys * t_arr)


def lambda_system(components: dict[str, ComponentRT]) -> float:
    """Aggregate failure rate: λ_sys = Σ(1/MTBF_i). Unit: 1/day."""
    return _lambda_system(components)


def mtbf_system(components: dict[str, ComponentRT]) -> float:
    """System MTBF = 1 / λ_system (days)."""
    return 1.0 / lambda_system(components)


def availability_annualised(lam_sys: float) -> float:
    """Fraction of a year the turbine is expected to be operational,
    assuming exponential failure + negligible repair time.
    Equals exp(-λ_sys × 365.25) ≈ R_turbine at the 1-year horizon."""
    return float(np.exp(-lam_sys * config.T_1YR))


# ---------------------------------------------------------------------------
# Lambda decomposition (for importance and sensitivity)
# ---------------------------------------------------------------------------

def lambda_contributions(components: dict[str, ComponentRT]) -> pd.DataFrame:
    """Each component's λ and its share of λ_total.

    Columns: component, lambda_per_day, lambda_fraction_pct,
             mtbf_days, tier, confidence, is_placeholder.
    """
    lam_total = lambda_system(components)
    rows = []
    for name, c in components.items():
        rows.append({
            "component": name,
            "lambda_per_day": c.lambda_per_day,
            "lambda_fraction_pct": 100.0 * c.lambda_per_day / lam_total,
            "mtbf_days": c.mtbf_days,
            "tier": c.tier,
            "confidence": c.confidence,
            "is_placeholder": c.is_placeholder,
        })
    return (
        pd.DataFrame(rows)
        .sort_values("lambda_fraction_pct", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# System reliability table
# ---------------------------------------------------------------------------

def system_reliability_table(
    components: dict[str, ComponentRT],
    t_values: tuple[float, ...] = config.MISSION_TIMES_DAYS,
) -> pd.DataFrame:
    """R_turbine(t) at each mission time, plus the system-level metrics.

    Columns: t_days, t_years, R_turbine, lambda_system, mtbf_system_days,
             Q_turbine (unreliability).
    """
    lam = lambda_system(components)
    rows = []
    for t in t_values:
        r = float(np.exp(-lam * t))
        rows.append({
            "t_days": t,
            "t_years": round(t / 365.25, 2),
            "lambda_system_per_day": lam,
            "mtbf_system_days": 1.0 / lam,
            "R_turbine": r,
            "Q_turbine": 1.0 - r,
        })
    return pd.DataFrame(rows)
