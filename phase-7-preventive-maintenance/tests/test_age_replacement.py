"""
Tests for age_replacement.py.

Every test verifies a mathematical property that must hold regardless of
which component or cost inputs are used. Numerical values are verified
against independent pre-calculations.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import gamma as _gamma

from aeolus_rams_phase7.age_replacement import (
    cost_rate, optimal_replacement_age, cost_ratio_sensitivity,
)
from aeolus_rams_phase7.config import (
    WeibullParams, ReplacementCosts, DESIGN_LIFE_DAYS,
)


# ── Test 1: β ≤ 1 guard ────────────────────────────────────────────────────

def test_beta_lt1_returns_no_T_star(beta_lt1_params, beta_lt1_costs):
    """Phase 2: Pitch System β=0.728 < 1 → T*=None, warning issued."""
    result = optimal_replacement_age(beta_lt1_params, beta_lt1_costs)
    assert result.T_star_days is None, "T* must be None for β < 1"
    assert result.beta_supports_PM is False
    assert result.T_star_warning != ""
    # Warning must mention β ≤ 1
    assert "1" in result.T_star_warning or "β" in result.T_star_warning


def test_beta_eq1_returns_no_T_star():
    """β = 1.0 exactly → no PM benefit (exponential = constant hazard)."""
    params = WeibullParams("Test", beta=1.0, eta=5000.0,
                           source="test", citation="")
    costs = ReplacementCosts("Test", Cp=1_000, Cf=10_000)
    result = optimal_replacement_age(params, costs)
    assert result.T_star_days is None
    assert result.beta_supports_PM is False


# ── Test 2: β > 1 produces valid T* ──────────────────────────────────────

def test_bearing_T_star_exists_and_finite(bearing_params, bearing_costs):
    """β=2.0 > 1 → T* must be finite and positive."""
    result = optimal_replacement_age(bearing_params, bearing_costs)
    assert result.beta_supports_PM is True
    assert result.T_star_days is not None
    assert result.T_star_days > 0.0
    assert math.isfinite(result.T_star_days)


def test_T_star_below_mttf_times_factor(bearing_params, bearing_costs):
    """T* for β=2 must be less than 3×MTTF (rough upper bound)."""
    result = optimal_replacement_age(bearing_params, bearing_costs)
    assert result.T_star_days < 3.0 * result.mttf_days


def test_C_star_less_than_C_no_PM(bearing_params, bearing_costs):
    """C(T*) must be strictly less than C_no_PM (PM must save money)."""
    result = optimal_replacement_age(bearing_params, bearing_costs)
    assert result.C_star < result.C_no_PM - 1e-6, (
        f"C* = {result.C_star:.4f} must be < C_no_PM = {result.C_no_PM:.4f}"
    )


def test_bearing_T_star_approx_23yr(bearing_params, bearing_costs):
    """Pre-verified: Bearing T* ≈ 23.0yr at Cp/Cf=0.10 (β=2.0, η=25000d)."""
    result = optimal_replacement_age(bearing_params, bearing_costs)
    T_star_yr = result.T_star_years
    assert abs(T_star_yr - 23.0) < 0.5, (
        f"Bearing T* = {T_star_yr:.2f}yr, expected ≈ 23.0yr"
    )


def test_gearbox_T_star_approx_22yr(gearbox_params, gearbox_costs):
    """Pre-verified: Gearbox T* ≈ 22.3yr at Cp/Cf=0.10 (β=1.8, η=24000d)."""
    result = optimal_replacement_age(gearbox_params, gearbox_costs)
    T_star_yr = result.T_star_years
    assert abs(T_star_yr - 22.3) < 0.5, (
        f"Gearbox T* = {T_star_yr:.2f}yr, expected ≈ 22.3yr"
    )


def test_generator_T_star_exceeds_design_life(generator_params):
    """Generator (β=1.5, η=38000d) T* >> 25yr → warning issued."""
    costs = ReplacementCosts("Generator", Cp=120_000, Cf=800_000)
    result = optimal_replacement_age(generator_params, costs)
    assert result.T_star_days is not None
    assert result.T_star_days > DESIGN_LIFE_DAYS
    assert result.T_star_warning != "", "Must warn when T* > design life"


# ── Test 3: T* monotonicity in Cp/Cf ─────────────────────────────────────

def test_T_star_increases_with_Cp_Cf_ratio(bearing_params):
    """Higher Cp/Cf → T* must increase (or stay constant). Monotone property."""
    base_Cf = 1_500_000.0
    ratios = [0.01, 0.05, 0.10, 0.20, 0.40, 0.50]
    T_stars = []
    for ratio in ratios:
        costs = ReplacementCosts("Bearing", Cp=ratio * base_Cf, Cf=base_Cf)
        res = optimal_replacement_age(bearing_params, costs)
        assert res.T_star_days is not None
        T_stars.append(res.T_star_days)
    for i in range(len(T_stars) - 1):
        assert T_stars[i] <= T_stars[i + 1] + 1.0, (
            f"T* not monotone at ratio step {i}: "
            f"{T_stars[i]:.0f}d > {T_stars[i+1]:.0f}d"
        )


# ── Test 4: Cost-rate formula ────────────────────────────────────────────

def test_cost_rate_at_zero_returns_inf(bearing_params, bearing_costs):
    """C(0) must return inf (undefined at T=0)."""
    assert cost_rate(0.0, bearing_params, bearing_costs) == float("inf")


def test_cost_rate_at_T_star_is_minimum(bearing_params, bearing_costs):
    """C(T*±1d) must be ≥ C(T*)."""
    result = optimal_replacement_age(bearing_params, bearing_costs)
    T_star = result.T_star_days
    C_star = result.C_star
    C_minus = cost_rate(T_star - 1.0, bearing_params, bearing_costs)
    C_plus  = cost_rate(T_star + 1.0, bearing_params, bearing_costs)
    assert C_minus >= C_star - 1e-4, f"C(T*-1) = {C_minus:.6f} < C* = {C_star:.6f}"
    assert C_plus  >= C_star - 1e-4, f"C(T*+1) = {C_plus:.6f} < C* = {C_star:.6f}"


# ── Test 5: Savings fraction ──────────────────────────────────────────────

def test_savings_fraction_between_0_and_1(bearing_params, bearing_costs):
    """Savings fraction must be in [0, 1]."""
    result = optimal_replacement_age(bearing_params, bearing_costs)
    assert 0.0 <= result.savings_fraction <= 1.0


def test_bearing_savings_approx_46pct(bearing_params, bearing_costs):
    """Pre-verified: Bearing savings ≈ 46.3% at Cp/Cf=0.10."""
    result = optimal_replacement_age(bearing_params, bearing_costs)
    assert abs(result.savings_fraction - 0.463) < 0.02, (
        f"Savings = {result.savings_fraction*100:.1f}%, expected ≈ 46.3%"
    )


# ── Test 6: Weibull MTTF formula ─────────────────────────────────────────

def test_mttf_formula(bearing_params):
    """MTTF = η × Γ(1 + 1/β), verified against pre-computed value."""
    from scipy.special import gamma as _gamma
    expected_mttf = 25000.0 * _gamma(1.0 + 1.0 / 2.0)  # = 25000 × 0.8862 = 22156d
    assert abs(bearing_params.mttf_days - expected_mttf) < 1.0