"""
Tests for farm_state.py — Layer 3.

Key checks:
  1. test_farm_kofn_with_perfect_bop
     When BoP never fails, A_farm ≈ A_kofN_exact.
  2. test_farm_dominated_by_failed_bop
     When BoP always fails, A_farm ≈ 0.
  3. test_farm_availability_between_zero_and_one
     Basic range check.
  4. test_n_turbine_up_bounded
     Mean turbines UP must be in [0, N_turbines].
  5. test_farm_monotone_in_k
     Increasing k_min decreases A_farm (monotone in threshold).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binom

from aeolus_rams_phase4.component_sampler import ComponentParams
from aeolus_rams_phase4.farm_state import simulate_farm
from aeolus_rams_phase4 import config


# ── Test 1: Perfect BoP isolates k-of-N ──────────────────────────────────────

def test_farm_kofn_with_perfect_bop(two_component_params, perfect_bop_params, rng):
    """When BoP MTBF → ∞, A_farm should equal A_kofN_exact."""
    N = 5  # small N for speed
    k = 3
    T = 365.25

    result = simulate_farm(
        two_component_params, perfect_bop_params, N, k, T, rng
    )
    # With perfect BoP, A_farm must equal A_kofN (within float precision)
    assert abs(result.A_farm - result.A_kofN_exact) < 1e-9, (
        f"A_farm={result.A_farm:.6f} ≠ A_kofN_exact={result.A_kofN_exact:.6f} "
        "when BoP is perfect"
    )
    assert result.A_bop > 0.9999  # verify BoP did stay up


# ── Test 2: Always-failed BoP → zero farm availability ───────────────────────

def test_farm_dominated_by_failed_bop(two_component_params):
    """BoP with MTBF=0 (fails at t=0) → A_farm ≈ 0."""
    always_failed_bop = [
        ComponentParams(
            name="Always Failed", mtbf_days=1e-9, mttr_raw_days=1e9,
            access_fraction=1.0, confidence="test_only", is_bop=True,
        )
    ]
    rng = np.random.default_rng(0)
    result = simulate_farm(two_component_params, always_failed_bop, 3, 2, 365.25, rng)
    # BoP fails immediately and is never repaired — A_farm should be ~0
    assert result.A_farm < 0.01, (
        f"A_farm={result.A_farm:.4f} should be ~0 when BoP always fails"
    )


# ── Test 3: Range ─────────────────────────────────────────────────────────────

def test_farm_availability_between_zero_and_one(
    two_component_params, perfect_bop_params, rng
):
    """A_farm, A_turbine_mean, A_bop, A_kofN_exact all in [0, 1]."""
    for _ in range(50):
        result = simulate_farm(two_component_params, perfect_bop_params, 5, 3, 365.25, rng)
        assert 0.0 <= result.A_farm <= 1.0 + 1e-10
        assert 0.0 <= result.A_turbine_mean <= 1.0 + 1e-10
        assert 0.0 <= result.A_bop <= 1.0 + 1e-10
        assert 0.0 <= result.A_kofN_exact <= 1.0 + 1e-10


# ── Test 4: Turbines-UP bounded ───────────────────────────────────────────────

def test_n_turbine_up_bounded(two_component_params, perfect_bop_params):
    """0 ≤ n_turbine_up_mean ≤ N_turbines."""
    N = 8
    rng = np.random.default_rng(1)
    result = simulate_farm(two_component_params, perfect_bop_params, N, 4, 365.25, rng)
    assert 0.0 <= result.n_turbine_up_mean <= N + 1e-9


# ── Test 5: A_farm monotone decreasing in k_min ───────────────────────────────

def test_farm_monotone_in_k(two_component_params, perfect_bop_params):
    """Higher k_min → lower or equal A_farm."""
    N = 10
    T = 365.25
    rng_seed = 42
    prev_A = 1.1

    for k in [1, 3, 6, 9, 10]:
        rng = np.random.default_rng(rng_seed)
        result = simulate_farm(two_component_params, perfect_bop_params, N, k, T, rng)
        assert result.A_farm <= prev_A + 1e-6, (
            f"A_farm={result.A_farm:.4f} at k={k} is HIGHER than previous {prev_A:.4f} — "
            "A_farm must be non-increasing in k"
        )
        prev_A = result.A_farm