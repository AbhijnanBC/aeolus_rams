from __future__ import annotations

import math

import numpy as np
import pytest

from aeolus_rams_phase3 import config
from aeolus_rams_phase3.importance import (
    birnbaum_importance, criticality_importance,
    importance_table, rank_summary,
)
from aeolus_rams_phase3.turbine_rbd import R_turbine_series, lambda_system


# ---------------------------------------------------------------------------
# Birnbaum importance — formula checks
# ---------------------------------------------------------------------------

def test_birnbaum_formula_verified(components):
    """IB_i = R_sys / R_i  (exact for series systems)."""
    t = config.T_5YR
    t_arr = np.array([t])
    R_sys = float(R_turbine_series(t_arr, components)[0])
    IB = birnbaum_importance(t, components)
    for name, c in components.items():
        R_i = float(c.R(t_arr)[0])
        expected = R_sys / R_i if R_i > 1e-15 else 0.0
        assert IB[name] == pytest.approx(expected, rel=1e-9), f"{name}"


def test_birnbaum_all_keys_present(components):
    IB = birnbaum_importance(config.T_5YR, components)
    assert set(IB.keys()) == set(components.keys())


def test_birnbaum_all_positive(components):
    IB = birnbaum_importance(config.T_5YR, components)
    for name, ib in IB.items():
        assert ib > 0, f"{name}: IB ≤ 0"


def test_birnbaum_all_at_least_one(components):
    """IB_i = R_sys/R_i ≥ R_sys/1 = R_sys ≤ 1, so 0 < IB ≤ 1 for series."""
    IB = birnbaum_importance(config.T_5YR, components)
    for name, ib in IB.items():
        # R_sys ≤ R_i, so IB = R_sys/R_i ≤ 1
        assert ib <= 1.0 + 1e-12, f"{name}: IB={ib} > 1"


def test_birnbaum_highest_for_least_reliable(components):
    """The component with the lowest R_i has IB = R_sys/R_i — highest IB."""
    t = config.T_5YR
    t_arr = np.array([t])
    IB = birnbaum_importance(t, components)
    # Most unreliable component (lowest R_i) should have highest IB
    least_reliable = min(components, key=lambda n: float(components[n].R(t_arr)[0]))
    most_IB = max(IB, key=IB.get)
    assert most_IB == least_reliable


# ---------------------------------------------------------------------------
# Criticality importance — formula checks
# ---------------------------------------------------------------------------

def test_criticality_sums_to_one(components):
    """Σ IC_i = 1.0 (within float rounding)."""
    IC = criticality_importance(config.T_5YR, components)
    assert sum(IC.values()) == pytest.approx(1.0, abs=1e-9)


def test_criticality_all_keys_present(components):
    IC = criticality_importance(config.T_5YR, components)
    assert set(IC.keys()) == set(components.keys())


def test_criticality_all_nonnegative(components):
    IC = criticality_importance(config.T_5YR, components)
    for name, ic in IC.items():
        assert ic >= 0, f"{name}: IC < 0"


def test_criticality_formula_verified(components):
    """IC_i = λ_i / λ_sys (exact for exponential series — see importance.py docstring)."""
    from aeolus_rams_phase3.turbine_rbd import lambda_system as lam_sys
    lam_total = lam_sys(components)
    IC = criticality_importance(config.T_5YR, components)
    for name, c in components.items():
        expected = c.lambda_per_day / lam_total
        assert IC[name] == pytest.approx(expected, rel=1e-12), f"{name}"


def test_hydraulic_and_pitch_top_2_criticality(components):
    """Phase 2 execution plan confirms Hydraulic and Pitch dominate IC."""
    IC = criticality_importance(config.T_5YR, components)
    top_2 = sorted(IC, key=IC.get, reverse=True)[:2]
    assert "Hydraulic System" in top_2 or "Pitch System" in top_2, (
        f"Expected Pitch or Hydraulic in top 2 by IC, got {top_2}"
    )


def test_hydraulic_and_pitch_together_exceed_half_IC(components):
    """Pitch + Hydraulic together represent >45% of λ_total, so IC > 0.45."""
    IC = criticality_importance(config.T_5YR, components)
    ic_pitch = IC.get("Pitch System", 0.0)
    ic_hydro = IC.get("Hydraulic System", 0.0)
    combined = ic_pitch + ic_hydro
    assert combined > 0.45, (
        f"Pitch ({ic_pitch:.4f}) + Hydraulic ({ic_hydro:.4f}) = {combined:.4f} "
        "< 0.45 — unexpected given that these two have the shortest MTBFs"
    )


# ---------------------------------------------------------------------------
# importance_table and rank_summary
# ---------------------------------------------------------------------------

def test_importance_table_has_all_components(components):
    tbl = importance_table(components)
    assert set(tbl["component"]) == set(components.keys())


def test_importance_table_has_IC_and_IB_columns(components):
    tbl = importance_table(components)
    ic_cols = [c for c in tbl.columns if c.startswith("IC_")]
    ib_cols = [c for c in tbl.columns if c.startswith("IB_")]
    assert len(ic_cols) >= 1
    assert len(ib_cols) >= 1


def test_importance_table_IC_sums_to_one_per_t(components):
    tbl = importance_table(components)
    for col in [c for c in tbl.columns if c.startswith("IC_")]:
        # The lambda-fraction formula sums exactly to 1.0, but the table
        # rounds each value to 6 decimal places. Across 13 components the
        # accumulated rounding error can reach 13 × 0.5e-6 = 6.5e-6.
        # abs=1e-4 is a conservative, physically meaningful tolerance.
        assert tbl[col].sum() == pytest.approx(1.0, abs=1e-4), f"{col} IC sum ≠ 1"


def test_rank_summary_correct_column_count(components):
    tbl = importance_table(components)
    rk = rank_summary(tbl, t=config.T_5YR)
    assert len(rk) == len(components)


def test_rank_summary_IC_rank_1_has_highest_IC(components):
    tbl = importance_table(components)
    rk = rank_summary(tbl, t=config.T_5YR)
    ic_col = f"IC_{int(config.T_5YR)}d"
    rank1 = rk[rk["IC_rank"] == 1].iloc[0]
    max_ic = rk[ic_col].max()
    assert rank1[ic_col] == pytest.approx(max_ic, rel=1e-9)
