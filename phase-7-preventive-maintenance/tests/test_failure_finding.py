"""
Tests for failure_finding.py.

Every test verifies a mathematical property of the PFDavg formula or
the derived τ_SIL2 value. The PFDavg formula is verified against an
independent numerical integration before any other test runs.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate

from aeolus_rams_phase7.failure_finding import (
    PFDavg_exact, PFDavg_approx,
    required_proof_test_interval, derive_lambda_from_Q,
    proof_test_interval_table,
)
from aeolus_rams_phase7.config import (
    LAMBDA_G3b, PHASE5_Q_G3b, TAU_IMPLIED_G3b_DAYS,
    SIL2_PFDavg_TARGET,
)


# ── Test 0: Formula correctness (most important test in Phase 7) ──────────

def test_PFDavg_correct_formula_via_integration():
    """Verify PFDavg_exact against direct numerical integration of the definition.

    PFDavg = (1/τ) ∫₀^τ [1 - exp(-λt)] dt

    Any mismatch here means the formula is wrong — all downstream
    Phase 7 safety results would be invalid.
    """
    for lam, tau in [(1e-4, 100), (7.5e-5, 730), (1e-3, 30), (5e-5, 365)]:
        # Direct numerical integration of the IEC definition
        integral, _ = integrate.quad(lambda t: 1.0 - math.exp(-lam * t), 0.0, tau)
        pfd_integration = integral / tau

        # Our closed-form formula
        pfd_formula = PFDavg_exact(lam, tau)

        assert abs(pfd_formula - pfd_integration) < 1e-9, (
            f"PFDavg formula mismatch at λ={lam}, τ={tau}: "
            f"formula={pfd_formula:.10f}, integration={pfd_integration:.10f}"
        )


def test_PFDavg_NOT_1_minus_formula():
    """Explicit test that the plan document's wrong formula is NOT used.

    Wrong formula: PFDavg = (1-exp(-λτ))/(λτ)  → This is (1 - PFDavg), not PFDavg.
    Correct formula: PFDavg = 1 - (1-exp(-λτ))/(λτ)
    """
    lam, tau = 7.5e-5, 730.0
    pfd_correct = PFDavg_exact(lam, tau)
    wrong_formula = (1.0 - math.exp(-lam * tau)) / (lam * tau)

    # Correct PFDavg should be small (~0.027), not large (~0.973)
    assert pfd_correct < 0.1, (
        f"PFDavg_exact returned {pfd_correct:.4f} — should be ~0.027"
    )
    assert wrong_formula > 0.9, (
        f"wrong_formula returned {wrong_formula:.4f} — should be ~0.97 (the complement)"
    )
    assert abs(pfd_correct + wrong_formula - 1.0) < 1e-10, (
        "pfd_correct and wrong_formula should sum to ~1.0 — confirms they are complements"
    )


# ── Test 1: Phase 5 bridge ────────────────────────────────────────────────

def test_PFDavg_correct_recovers_Q_G3b(lambda_G3b):
    """PFDavg_exact(λ_G3b, 730d) must equal Q_G3b from Phase 5.

    This is the Phase 5→7 numerical bridge. If it fails, either the
    formula is wrong or λ_G3b was derived incorrectly.
    """
    pfd = PFDavg_exact(lambda_G3b, TAU_IMPLIED_G3b_DAYS)
    assert abs(pfd - PHASE5_Q_G3b) < 1e-5, (
        f"PFDavg_exact(λ_G3b, {TAU_IMPLIED_G3b_DAYS}d) = {pfd:.6f}, "
        f"expected Q_G3b = {PHASE5_Q_G3b:.6f}"
    )


def test_lambda_G3b_approximately_correct():
    """λ_G3b ≈ 7.57×10⁻⁵ /day (pre-verified by Python before coding)."""
    assert abs(LAMBDA_G3b - 7.571e-5) < 0.01e-5, (
        f"λ_G3b = {LAMBDA_G3b:.4e}, expected ≈ 7.571e-5"
    )


# ── Test 2: τ_SIL2 correctness ───────────────────────────────────────────

def test_tau_SIL2_achieves_target(lambda_G3b):
    """Plugging τ_SIL2 back into PFDavg_exact must give PFDavg ≤ SIL2_target."""
    tau_sil2 = required_proof_test_interval(lambda_G3b, SIL2_PFDavg_TARGET)
    pfd = PFDavg_exact(lambda_G3b, tau_sil2)
    assert pfd <= SIL2_PFDavg_TARGET + 1e-10, (
        f"τ_SIL2={tau_sil2:.2f}d gives PFDavg={pfd:.6f} > target {SIL2_PFDavg_TARGET:.0e}"
    )


def test_tau_SIL2_approximately_27d(lambda_G3b):
    """Pre-verified: τ_SIL2 ≈ 26.4 days at λ_G3b = 7.571e-5/d."""
    tau_sil2 = required_proof_test_interval(lambda_G3b, SIL2_PFDavg_TARGET)
    assert abs(tau_sil2 - 26.4) < 1.0, (
        f"τ_SIL2 = {tau_sil2:.2f}d, expected ≈ 26.4d"
    )


def test_approx_formula_close_to_exact_for_small_lambda_tau(lambda_G3b):
    """For λτ < 0.1, exact and approx should agree within 5%."""
    tau = 50.0   # λτ = 7.571e-5 × 50 ≈ 0.0038 << 0.1
    pfd_ex = PFDavg_exact(lambda_G3b, tau)
    pfd_ap = PFDavg_approx(lambda_G3b, tau)
    rel_err = abs(pfd_ex - pfd_ap) / pfd_ex
    assert rel_err < 0.05, (
        f"Exact={pfd_ex:.6f}, Approx={pfd_ap:.6f}, rel_err={rel_err:.3%}"
    )


# ── Test 3: Monotonicity ─────────────────────────────────────────────────

def test_PFDavg_monotone_increasing_in_tau(lambda_G3b):
    """PFDavg must increase as τ increases (more time → more latent failures)."""
    taus = [7, 14, 27, 30, 60, 90, 180, 365, 730]
    pdfs = [PFDavg_exact(lambda_G3b, t) for t in taus]
    for i in range(len(pdfs) - 1):
        assert pdfs[i] < pdfs[i + 1], (
            f"PFDavg not monotone: PFD({taus[i]}d)={pdfs[i]:.6f} "
            f">= PFD({taus[i+1]}d)={pdfs[i+1]:.6f}"
        )


def test_tau_required_monotone_in_lambda():
    """Higher λ → shorter τ required for same SIL target."""
    target = SIL2_PFDavg_TARGET
    lambdas = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    taus = [required_proof_test_interval(lam, target) for lam in lambdas]
    for i in range(len(taus) - 1):
        assert taus[i] > taus[i + 1] - 0.01, (
            f"τ not monotone in λ: τ({lambdas[i]:.0e})={taus[i]:.2f}d "
            f"<= τ({lambdas[i+1]:.0e})={taus[i+1]:.2f}d"
        )


# ── Test 4: derive_lambda_from_Q round-trip ───────────────────────────────

def test_derive_lambda_round_trip():
    """Derive λ from (Q, τ), plug back in, should recover Q."""
    Q_test = 0.027132
    tau_test = 730.0
    lam = derive_lambda_from_Q(Q_test, tau_test)
    Q_recovered = PFDavg_exact(lam, tau_test)
    assert abs(Q_recovered - Q_test) < 1e-8, (
        f"Round-trip error: Q_recovered={Q_recovered:.8f}, Q_test={Q_test:.8f}"
    )


# ── Test 5: Edge cases ────────────────────────────────────────────────────

def test_PFDavg_zero_at_zero_tau():
    """PFDavg(λ, τ=0) = 0 (no time → no accumulated PFD)."""
    pfd = PFDavg_exact(1e-3, 0.0)
    assert pfd == 0.0


def test_PFDavg_approaches_half_lambda_tau_for_small_x():
    """For very small λτ, PFDavg ≈ λτ/2 to high accuracy (Taylor: remainder O(x²))."""
    lam, tau = 1e-6, 1.0   # λτ = 1e-6, extremely small
    pfd = PFDavg_exact(lam, tau)
    expected = lam * tau / 2.0
    assert abs(pfd - expected) / expected < 1e-5


def test_SIL_classification_in_table(lambda_G3b):
    """At τ=27d, the relay should achieve SIL-2; at τ=730d, it should not."""
    df = proof_test_interval_table(lambda_G3b, "G3b Safety Relay")
    row_27  = df[df["tau_days"].between(26, 28)].iloc[0]
    row_730 = df[df["tau_days"].between(700, 800)].iloc[0]
    assert row_27["SIL_achieved"] == "SIL-2"
    assert row_730["SIL_achieved"] != "SIL-2"   # Should be BELOW SIL-1