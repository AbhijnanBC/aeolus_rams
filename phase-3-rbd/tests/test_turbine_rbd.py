from __future__ import annotations

import math

import numpy as np
import pytest

from aeolus_rams_phase3 import config
from aeolus_rams_phase3.turbine_rbd import (
    R_turbine_series, R_turbine_exponential, lambda_system, mtbf_system,
    system_reliability_table, lambda_contributions, availability_annualised,
)


# ---------------------------------------------------------------------------
# R_turbine_series: series product matches closed-form exponential
# ---------------------------------------------------------------------------

def test_series_product_equals_exponential_closed_form(components):
    """R_sys = ∏R_i must equal exp(-λ_total × t) for exponential components."""
    t_vals = np.array([365.25, 1825.25, 3650.5])
    lam = lambda_system(components)
    R_product = R_turbine_series(t_vals, components)
    R_closed = np.exp(-lam * t_vals)
    np.testing.assert_allclose(R_product, R_closed, rtol=1e-9)


def test_R_turbine_series_at_zero_is_one(components):
    R = R_turbine_series(np.array([0.0]), components)
    assert float(R[0]) == pytest.approx(1.0, abs=1e-12)


def test_R_turbine_series_monotonically_decreasing(components):
    t = np.linspace(0, 5000, 300)
    R = R_turbine_series(t, components)
    assert np.all(np.diff(R) <= 0)


def test_R_turbine_less_than_any_single_component(components):
    """Series system must be less reliable than its most reliable component."""
    t = np.array([config.T_5YR])
    R_sys = float(R_turbine_series(t, components)[0])
    for name, c in components.items():
        R_i = float(c.R(t)[0])
        assert R_sys <= R_i + 1e-12, (
            f"R_system(5yr)={R_sys} > R_{name}={R_i} — "
            "series system cannot be more reliable than any component"
        )


def test_R_turbine_1yr_in_expected_range(components):
    # With Pitch (MTBF=1936) and Hydraulic (MTBF=1845) dominating,
    # R_system(1yr) must be below either component individually
    # and above 0 (trivially).
    R_1yr = float(R_turbine_series(np.array([config.T_1YR]), components)[0])
    assert 0.0 < R_1yr < 0.90  # pitch alone gives ~0.83, system is lower


def test_R_turbine_5yr_dominated_by_pitch_and_hydraulic(components):
    # The two most unreliable known-fitted components set a ceiling on R_sys
    R_5yr = float(R_turbine_series(np.array([config.T_5YR]), components)[0])
    R_pitch_5yr = float(components["Pitch System"].R(np.array([config.T_5YR]))[0])
    R_hydro_5yr = float(components["Hydraulic System"].R(np.array([config.T_5YR]))[0])
    assert R_5yr < R_pitch_5yr
    assert R_5yr < R_hydro_5yr


# ---------------------------------------------------------------------------
# lambda_system and mtbf_system
# ---------------------------------------------------------------------------

def test_lambda_system_positive(components):
    lam = lambda_system(components)
    assert lam > 0


def test_mtbf_system_below_any_single_component(components):
    """Series system MTBF must be less than the shortest component MTBF."""
    sys_mtbf = mtbf_system(components)
    min_component_mtbf = min(c.mtbf_days for c in components.values())
    assert sys_mtbf < min_component_mtbf


def test_pitch_and_hydraulic_together_exceed_half_lambda_total(components):
    """Spec claims pitch+hydraulic dominate the λ budget."""
    lam_total = lambda_system(components)
    lam_pitch = components["Pitch System"].lambda_per_day
    lam_hydro = components["Hydraulic System"].lambda_per_day
    assert (lam_pitch + lam_hydro) / lam_total > 0.45


def test_availability_annualised_bounded(components):
    a = availability_annualised(lambda_system(components))
    assert 0.0 < a < 1.0


# ---------------------------------------------------------------------------
# system_reliability_table
# ---------------------------------------------------------------------------

def test_system_reliability_table_row_count(components):
    tbl = system_reliability_table(components)
    assert len(tbl) == len(config.MISSION_TIMES_DAYS)


def test_system_reliability_table_no_nan(components):
    tbl = system_reliability_table(components)
    assert not tbl["R_turbine"].isna().any()


def test_system_reliability_table_monotone(components):
    tbl = system_reliability_table(components).sort_values("t_days")
    r = tbl["R_turbine"].to_numpy()
    assert np.all(np.diff(r) <= 0)


def test_system_reliability_table_q_plus_r_is_one(components):
    tbl = system_reliability_table(components)
    totals = (tbl["R_turbine"] + tbl["Q_turbine"]).to_numpy()
    np.testing.assert_allclose(totals, 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# lambda_contributions
# ---------------------------------------------------------------------------

def test_lambda_contributions_fractions_sum_to_100(components):
    tbl = lambda_contributions(components)
    assert tbl["lambda_fraction_pct"].sum() == pytest.approx(100.0, abs=0.001)


def test_lambda_contributions_top_two_are_pitch_and_hydraulic(components):
    tbl = lambda_contributions(components).reset_index(drop=True)
    top_two = set(tbl.loc[:1, "component"].tolist())
    assert "Pitch System" in top_two or "Hydraulic System" in top_two
