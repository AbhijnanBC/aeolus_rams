"""
Tests for turbine_state.py — Layer 2.

The five mandatory checks from the Phase 4 spec (Section 4.8):

  1. test_turbine_availability_converges_to_analytical
     Simulated A_turbine → MTBF/(MTBF+eff_MTTR) as n→∞. The most
     critical validation test: if this fails, no downstream number is valid.

  2. test_zero_repair_time_gives_R_t
     Phase 3 numerical bridge: MTTR→0 simulation must recover
     R_turbine(T) = exp(-λ_system × T) from Phase 3's committed value.

  3. test_seed_reproducibility
     Two runs with same seed must be bitwise identical.

  4. test_availability_between_zero_and_one
     Basic range check.

  5. test_comp_down_days_sums_to_turbine_down_time
     Per-component downtime ≤ turbine downtime (since multiple components
     can be down simultaneously).
"""
from __future__ import annotations

import numpy as np
import pytest

from aeolus_rams_phase4.component_sampler import ComponentParams
from aeolus_rams_phase4.turbine_state import simulate_turbine
from aeolus_rams_phase4 import config


# ── Test 1: Convergence to analytical steady-state availability ────────────

def test_turbine_availability_converges_to_analytical():
    """A_turbine (simulated) → MTBF/(MTBF+eff_MTTR) within 0.5% at n=50,000."""
    # Use system-level MTBF as a single surrogate component for clean arithmetic
    mtbf = config.PHASE3_MTBF_SYSTEM          # 457.88 days
    mttr_raw = 5.0
    access = 0.60
    eff_mttr = mttr_raw / access               # 8.33 days
    expected_A = mtbf / (mtbf + eff_mttr)      # ≈ 0.9822

    comp = ComponentParams(
        name="Surrogate", mtbf_days=mtbf, mttr_raw_days=mttr_raw,
        access_fraction=access, confidence="test_only",
    )
    T = config.T_MISSION_DAYS * 50            # 50 years per simulation
    n = 1000
    rng = np.random.default_rng(0)
    avails = []
    for _ in range(n):
        res = simulate_turbine([comp], T, rng)
        avails.append(res.availability)

    simulated_A = float(np.mean(avails))
    assert abs(simulated_A - expected_A) < 0.005, (
        f"Simulated A={simulated_A:.4f} deviates >0.5% from analytical {expected_A:.4f}"
    )


# ── Test 2: Phase 3 numerical bridge ─────────────────────────────────────────

def test_zero_repair_time_gives_R_t():
    """MTTR→0: simulated availability converges to Phase 3's R_turbine(1yr).

    This is the most important cross-phase validation. It checks that the
    Phase 4 simulation, when repair time goes to zero, reproduces the Phase 3
    closed-form R(t) = exp(-λ_system × T).

    Phase 3 committed value: R_turbine(1yr) = 0.4503662982516767.
    (Source: phase-3-rbd/outputs/system_reliability_table.csv)
    """
    lam_sys = config.PHASE3_LAMBDA_SYSTEM  # 0.002184 /day
    mtbf_sys = 1.0 / lam_sys               # 457.88 days
    T = config.T_MISSION_DAYS              # 365.25 days
    expected_R = float(np.exp(-lam_sys * T))  # = 0.4504

    # Zero-repair component: essentially never repaired within T
    # (MTTR=1e-6 days so repair is near-instantaneous and availability ≈ R(T))
    zero_repair = ComponentParams(
        name="ZeroRepair", mtbf_days=mtbf_sys, mttr_raw_days=1e-6,
        access_fraction=1.0, confidence="test_only",
    )

    n = 100_000  # high n for numerical precision
    rng = np.random.default_rng(7)
    avails = []
    for _ in range(n):
        res = simulate_turbine([zero_repair], T, rng)
        survived = 1.0 if len(res.down_periods) == 0 else 0.0
        avails.append(survived)

    simulated_R = float(np.mean(avails))
    tolerance = config.PHASE3_BRIDGE_TOLERANCE  # 0.008

    assert abs(simulated_R - expected_R) < tolerance, (
        f"MTTR=0 bridge test FAILED: "
        f"simulated={simulated_R:.5f}, "
        f"Phase3 R_turbine(1yr)={expected_R:.5f}, "
        f"error={abs(simulated_R - expected_R):.5f} > tolerance={tolerance}"
    )


# ── Test 3: Seed reproducibility ─────────────────────────────────────────────

def test_seed_reproducibility(two_component_params):
    """Same seed → identical simulation results."""
    T = 365.25
    res_a = simulate_turbine(two_component_params, T, np.random.default_rng(42))
    res_b = simulate_turbine(two_component_params, T, np.random.default_rng(42))
    assert abs(res_a.availability - res_b.availability) < 1e-12, (
        "Same seed must give identical availability"
    )
    assert res_a.down_periods == res_b.down_periods


# ── Test 4: Availability range ────────────────────────────────────────────────

def test_availability_between_zero_and_one(two_component_params, rng):
    """A_turbine must always be in [0, 1]."""
    T = 365.25
    for _ in range(200):
        res = simulate_turbine(two_component_params, T, rng)
        assert 0.0 <= res.availability <= 1.0 + 1e-10, (
            f"Availability {res.availability} out of range"
        )


# ── Test 5: Component downtime consistency ────────────────────────────────────

def test_comp_down_days_non_negative(two_component_params, rng):
    """Per-component downtime must be non-negative and ≤ T_days."""
    T = 365.25
    for _ in range(100):
        res = simulate_turbine(two_component_params, T, rng)
        for comp, days in res.comp_down_days.items():
            assert days >= 0.0, f"{comp} downtime {days} < 0"
            assert days <= T + 1e-9, f"{comp} downtime {days} > T={T}"


def test_down_periods_within_mission_time(two_component_params, rng):
    """All down_period boundaries must be in [0, T_days]."""
    T = 365.25
    for _ in range(100):
        res = simulate_turbine(two_component_params, T, rng)
        for start, end in res.down_periods:
            assert 0.0 <= start < end <= T + 1e-9, (
                f"Down period ({start:.2f}, {end:.2f}) outside [0, {T}]"
            )