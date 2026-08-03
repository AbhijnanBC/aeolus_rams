"""
Tests for sensitivity.py — Layer 5.

Key checks:
  1. test_sensitivity_monotonic_in_mtbf
     Higher MTBF → equal or higher A_farm. Violation = simulation bug.
  2. test_sensitivity_monotonic_in_mttr_mult
     Higher MTTR multiplier → equal or lower A_farm.
  3. test_sweep_output_shape
     Correct number of rows and required columns.
  4. test_tornado_data_sorted
     tornado_df must be sorted by impact_total descending.
"""
from __future__ import annotations

import numpy as np
import pytest

from aeolus_rams_phase4.sensitivity import (
    sweep_component_mtbf,
    sweep_mttr_multiplier,
    build_tornado_data,
)
from aeolus_rams_phase4 import config


@pytest.fixture
def small_pitch_sweep(two_component_params, perfect_bop_params):
    """5-point pitch MTBF sweep at n=200 for speed."""
    sweep_vals = np.array([500.0, 1000.0, 2000.0, 4000.0, 8000.0])
    return sweep_component_mtbf(
        two_component_params, perfect_bop_params,
        "Pitch System", sweep_vals,
        N_turbines=5, k_min=3, T_days=365.25,
        n_simulations=200, seed=42,
    )


def test_sensitivity_monotonic_in_mtbf(small_pitch_sweep):
    """A_farm_mean must be non-decreasing as Pitch MTBF increases."""
    vals = small_pitch_sweep["A_farm_mean"].values
    for i in range(len(vals) - 1):
        assert vals[i] <= vals[i + 1] + 0.01, (
            f"A_farm dropped from {vals[i]:.4f} to {vals[i+1]:.4f} "
            f"as MTBF increased — monotonicity violated"
        )


def test_sweep_output_shape(small_pitch_sweep):
    """Sweep must have 5 rows and required columns."""
    assert len(small_pitch_sweep) == 5
    for col in ["component", "param_varied", "param_value",
                "A_farm_mean", "A_farm_std", "A_farm_p5", "A_farm_p95"]:
        assert col in small_pitch_sweep.columns


def test_sensitivity_monotonic_in_mttr_mult(two_component_params, perfect_bop_params):
    """A_farm_mean must be non-increasing as MTTR multiplier increases."""
    mult_vals = np.array([0.5, 1.0, 2.0, 4.0])
    df = sweep_mttr_multiplier(
        two_component_params, perfect_bop_params, mult_vals,
        N_turbines=5, k_min=3, T_days=365.25,
        n_simulations=200, seed=42,
    )
    vals = df["A_farm_mean"].values
    for i in range(len(vals) - 1):
        assert vals[i] >= vals[i + 1] - 0.01, (
            f"A_farm INCREASED from {vals[i]:.4f} to {vals[i+1]:.4f} "
            f"as MTTR multiplier increased — monotonicity violated"
        )


def test_tornado_data_sorted(small_pitch_sweep, two_component_params, perfect_bop_params):
    """build_tornado_data must produce DataFrame sorted by impact_total descending."""
    mttr_df = sweep_mttr_multiplier(
        two_component_params, perfect_bop_params,
        np.array([0.5, 1.0, 2.0]),
        N_turbines=5, k_min=3, T_days=365.25,
        n_simulations=200, seed=42,
    )
    sweep_dfs = {"pitch": small_pitch_sweep, "mttr": mttr_df}
    tornado = build_tornado_data(sweep_dfs, baseline_A_farm=0.95)
    impacts = tornado["impact_total"].values
    for i in range(len(impacts) - 1):
        assert impacts[i] >= impacts[i + 1] - 1e-9, "Tornado not sorted by impact"