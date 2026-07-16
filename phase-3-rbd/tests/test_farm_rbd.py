from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import binom

from aeolus_rams_phase3 import config
from aeolus_rams_phase3.farm_rbd import (
    R_kofN, R_at_least_one, R_farm_with_bop, farm_system_table,
    bop_sensitivity_table, _bop_R,
)


# ---------------------------------------------------------------------------
# R_kofN: exact binomial formula verification
# ---------------------------------------------------------------------------

def test_R_kofN_all_must_work_equals_serial_product():
    """k=N means all must be working — R(k=N) = R_single^N."""
    N, R_s = 5, 0.8
    expected = R_s ** N
    assert R_kofN(R_s, N, k=N) == pytest.approx(expected, rel=1e-9)


def test_R_kofN_k1_equals_R_at_least_one():
    """k=1 is the 'at least one' system — R = 1 - (1-R)^N."""
    N, R_s = 10, 0.7
    expected = 1.0 - (1.0 - R_s) ** N
    assert R_kofN(R_s, N, k=1) == pytest.approx(expected, rel=1e-9)


def test_R_kofN_matches_scipy_binom():
    """Verify against scipy.stats.binom CDF directly."""
    N, k, R_s = 22, 15, 0.55
    expected = float(1.0 - binom.cdf(k - 1, N, R_s))
    assert R_kofN(R_s, N, k) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("R_s", [0.0, 0.3, 0.5, 0.8, 0.99, 1.0])
def test_R_kofN_bounded(R_s):
    result = R_kofN(R_s, N=10, k=7)
    assert 0.0 <= result <= 1.0


def test_R_kofN_increases_with_R_single():
    """Higher single-turbine reliability → higher k-of-N reliability."""
    R_values = np.linspace(0.1, 0.99, 30)
    results = [R_kofN(r, N=22, k=15) for r in R_values]
    assert all(b >= a - 1e-10 for a, b in zip(results, results[1:]))


def test_R_kofN_decreases_with_k():
    """More stringent k → lower system reliability."""
    R_s, N = 0.6, 10
    results = [R_kofN(R_s, N, k) for k in range(1, N + 1)]
    assert all(b <= a + 1e-10 for a, b in zip(results, results[1:]))


def test_R_kofN_invalid_k_raises():
    with pytest.raises(ValueError):
        R_kofN(0.8, N=10, k=0)
    with pytest.raises(ValueError):
        R_kofN(0.8, N=10, k=11)


def test_R_at_least_one_formula():
    R_s, N = 0.7, 8
    expected = 1.0 - (1.0 - R_s) ** N
    assert R_at_least_one(R_s, N) == pytest.approx(expected, rel=1e-9)


def test_R_at_least_one_is_upper_bound_of_kofN(components):
    """R(k≥1) ≥ R(k≥k_min) for any k_min ≥ 1."""
    from aeolus_rams_phase3.turbine_rbd import lambda_system
    import numpy as np
    lam = lambda_system(components)
    R_turb = float(np.exp(-lam * config.T_5YR))
    R_ub = R_at_least_one(R_turb, config.FARM_N_TURBINES)
    R_kN = R_kofN(R_turb, config.FARM_N_TURBINES, config.FARM_K_MIN_TURBINES)
    assert R_ub >= R_kN


# ---------------------------------------------------------------------------
# BoP: R_substation and R_cable
# ---------------------------------------------------------------------------

def test_bop_R_at_zero_is_one():
    R_sub, R_cab = _bop_R(0.0)
    assert R_sub == pytest.approx(1.0, abs=1e-12)
    assert R_cab == pytest.approx(1.0, abs=1e-12)


def test_bop_R_decreasing_in_time():
    _, R_cab_early = _bop_R(100.0)
    _, R_cab_late = _bop_R(5000.0)
    assert R_cab_late < R_cab_early


def test_bop_R_substation_high_reliability():
    """Substation MTBF=18,000 days → R(5yr)=exp(-1825/18000) ≈ 0.904."""
    R_sub, _ = _bop_R(1825.25)
    expected = math.exp(-1825.25 / 18_000.0)
    assert R_sub == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# R_farm_with_bop
# ---------------------------------------------------------------------------

def test_R_farm_with_bop_requires_t_or_bop_args():
    with pytest.raises(ValueError):
        R_farm_with_bop(R_turbine=0.5)


def test_R_farm_with_bop_uses_t_correctly():
    R_turb = 0.5
    R_sub, R_cab = _bop_R(config.T_5YR)
    R_kN = R_kofN(R_turb, config.FARM_N_TURBINES, config.FARM_K_MIN_TURBINES)
    expected = R_kN * R_sub * R_cab
    result = R_farm_with_bop(R_turb, t_days=config.T_5YR)
    assert result == pytest.approx(expected, rel=1e-9)


def test_R_farm_with_bop_bounded(components):
    from aeolus_rams_phase3.turbine_rbd import lambda_system
    lam = lambda_system(components)
    R_t = float(np.exp(-lam * config.T_5YR))
    result = R_farm_with_bop(R_t, t_days=config.T_5YR)
    assert 0.0 <= result <= 1.0


def test_R_farm_bop_below_kofN_alone(components):
    """Adding BoP in series can only lower or equal k-of-N reliability."""
    from aeolus_rams_phase3.turbine_rbd import lambda_system
    lam = lambda_system(components)
    R_t = float(np.exp(-lam * config.T_5YR))
    R_kN = R_kofN(R_t, config.FARM_N_TURBINES, config.FARM_K_MIN_TURBINES)
    R_farm = R_farm_with_bop(R_t, t_days=config.T_5YR)
    assert R_farm <= R_kN + 1e-12


# ---------------------------------------------------------------------------
# farm_system_table
# ---------------------------------------------------------------------------

def test_farm_system_table_correct_row_count(components):
    tbl = farm_system_table(components)
    assert len(tbl) == len(config.MISSION_TIMES_DAYS)


def test_farm_system_table_no_nan(components):
    tbl = farm_system_table(components)
    num_cols = ["R_single_turbine", "R_substation", "R_export_cable", "R_farm_total"]
    assert not tbl[num_cols].isna().any().any()


def test_farm_table_R_farm_less_than_R_kofN(components):
    tbl = farm_system_table(components)
    kN_col = f"R_{config.FARM_K_MIN_TURBINES}of{config.FARM_N_TURBINES}_turbines"
    assert (tbl["R_farm_total"] <= tbl[kN_col] + 1e-12).all()


def test_farm_table_R_values_monotone_decreasing(components):
    tbl = farm_system_table(components).sort_values("t_days")
    assert np.all(np.diff(tbl["R_farm_total"].to_numpy()) <= 0)


# ---------------------------------------------------------------------------
# bop_sensitivity_table
# ---------------------------------------------------------------------------

def test_bop_sensitivity_table_rows(components):
    from aeolus_rams_phase3.turbine_rbd import lambda_system
    lam = lambda_system(components)
    R_turb_kN = R_kofN(
        float(np.exp(-lam * config.T_5YR)),
        config.FARM_N_TURBINES,
        config.FARM_K_MIN_TURBINES,
    )
    tbl = bop_sensitivity_table(R_turb_kN, n_steps=5)
    assert not tbl.empty
    assert (tbl["R_farm_5yr"] <= R_turb_kN + 1e-12).all()


import numpy as np
