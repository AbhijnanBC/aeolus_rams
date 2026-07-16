"""
aeolus_rams_phase3.farm_rbd
==============================
Section 3.5 — Farm-level RBD: k-of-N turbines in parallel feeding
a series path (Offshore Substation → Export Cable → Grid).

Why this cannot be fully solved here
--------------------------------------
The k-of-N binomial model (implemented below) assumes:
  1. All N turbines are identically and independently distributed — valid.
  2. No time-dependent repair queue or common-cause dependencies — NOT
     valid for a real farm with shared maintenance vessels, weather-access
     windows, and occasional grid curtailments.
  3. The BoP (substation + cable) combines trivially in series — valid
     under independent failure assumptions.

Assumption 2 breaks down in practice, which is why Phase 4 (Monte Carlo
simulation) exists: it can model repair queues, weather windows, and
common-cause factors that this chapter's closed-form results cannot.

This module computes the ANALYTICAL SPECIAL CASES (exactly solvable) and
provides the topology definition Phase 4 needs to build its simulator on.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from scipy.stats import binom

from . import config
from .component_rt import ComponentRT
from .turbine_rbd import R_turbine_exponential, lambda_system

logger = logging.getLogger("aeolus_rams_phase3.farm_rbd")


# ---------------------------------------------------------------------------
# BoP component reliability
# ---------------------------------------------------------------------------

def _bop_R(t_days: float) -> tuple[float, float]:
    """R(t) for Offshore Substation and Export Cable at time t_days."""
    bop = config.BALANCE_OF_PLANT
    R_sub = math.exp(-t_days / bop["Offshore Substation"].mtbf_days)
    R_cab = math.exp(-t_days / bop["Export Cable"].mtbf_days)
    return R_sub, R_cab


# ---------------------------------------------------------------------------
# k-of-N turbine reliability
# ---------------------------------------------------------------------------

def R_kofN(R_single: float, N: int, k: int) -> float:
    """Exact k-of-N system reliability for identical independent turbines.

    P(at least k of N turbines working) = Σ_{i=k}^{N} C(N,i) R^i (1-R)^(N-i)
    = 1 - Binom.CDF(k-1; N, R)

    Valid assumption: turbines are stochastically identical (same R_turbine
    from Phase 2) and statistically independent (no common-cause failures
    modelled here — that is Phase 4's job).
    """
    if not 1 <= k <= N:
        raise ValueError(f"k={k} must satisfy 1 ≤ k ≤ N={N}")
    if not 0.0 <= R_single <= 1.0:
        raise ValueError(f"R_single={R_single} must be in [0, 1]")
    return float(1.0 - binom.cdf(k - 1, N, R_single))


def R_at_least_one(R_single: float, N: int) -> float:
    """Upper bound: farm available if at least 1 of N turbines runs.
    P(≥1) = 1 - (1-R)^N.  Real farm needs k > 1 for full contractual
    delivery — this is strictly an upper bound."""
    return float(1.0 - (1.0 - R_single) ** N)


# ---------------------------------------------------------------------------
# Full farm system reliability (turbines + BoP)
# ---------------------------------------------------------------------------

def R_farm_with_bop(
    R_turbine: float,
    N: int = config.FARM_N_TURBINES,
    k: int = config.FARM_K_MIN_TURBINES,
    R_substation: float | None = None,
    R_cable: float | None = None,
    t_days: float | None = None,
) -> float:
    """Series combination: k-of-N turbines AND substation AND cable.

    Provide either (R_substation, R_cable) directly, or `t_days` to
    derive them from config.BALANCE_OF_PLANT MTBF values.
    """
    if R_substation is None or R_cable is None:
        if t_days is None:
            raise ValueError("Provide either (R_substation, R_cable) or t_days")
        R_substation, R_cable = _bop_R(t_days)

    R_kN = R_kofN(R_turbine, N, k)
    return R_kN * R_substation * R_cable


def farm_system_table(
    components: dict[str, ComponentRT],
    t_values: tuple[float, ...] = config.MISSION_TIMES_DAYS,
    N: int = config.FARM_N_TURBINES,
    k: int = config.FARM_K_MIN_TURBINES,
) -> pd.DataFrame:
    """Farm-level reliability at each mission time.

    Columns: t_days, t_years, R_turbine, R_at_least_1,
             R_kofN_no_bop, R_substation, R_cable, R_farm_total,
             NOTE (reminder that full solution requires Monte Carlo).
    """
    lam = lambda_system(components)
    rows = []
    for t in t_values:
        R_t = float(np.exp(-lam * t))
        R_sub, R_cab = _bop_R(t)
        rows.append({
            "t_days": t,
            "t_years": round(t / 365.25, 2),
            "R_single_turbine": R_t,
            "R_at_least_1_turbine": R_at_least_one(R_t, N),
            f"R_{k}of{N}_turbines": R_kofN(R_t, N, k),
            "R_substation": R_sub,
            "R_export_cable": R_cab,
            "R_farm_total": R_farm_with_bop(R_t, N, k, R_sub, R_cab),
            "analytical_note": (
                f"k-of-N assumes identical independent turbines and "
                "no repair queues. Full correlated/repair model → Phase 4."
            ),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# BoP sensitivity
# ---------------------------------------------------------------------------

def bop_sensitivity_table(
    R_turbine_kofN: float,
    cable_mtbf_range_days: tuple[float, float] = (870.0, 2610.0),
    sub_mtbf_range_days: tuple[float, float] = (7300.0, 36500.0),
    n_steps: int = 40,
) -> pd.DataFrame:
    """How much does the farm-level R depend on BoP assumptions?

    The BoP parameters carry the most uncertainty in Phase 3 (cable
    length assumption, substation failure data sparsity). This table
    sweeps both ranges at t=1825 days (5yr) and shows the resulting
    R_farm. Phase 4 Monte Carlo should include this sweep."""
    rows = []
    for cable_mtbf in np.linspace(*cable_mtbf_range_days, n_steps):
        R_cable = math.exp(-config.T_5YR / cable_mtbf)
        for sub_mtbf in np.linspace(*sub_mtbf_range_days, 5):
            R_sub = math.exp(-config.T_5YR / sub_mtbf)
            rows.append({
                "cable_mtbf_days": round(cable_mtbf),
                "substation_mtbf_days": round(sub_mtbf),
                "R_cable_5yr": R_cable,
                "R_substation_5yr": R_sub,
                "R_farm_5yr": R_turbine_kofN * R_sub * R_cable,
            })
    return pd.DataFrame(rows)
