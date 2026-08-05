"""
aeolus_rams_phase7.failure_finding
=====================================
Proof-test interval derivation for dormant protective devices.

Background (IEC 61511:2016, Section 6.7; IEC 61508-6:2010 Annex D):

A "dormant" protective device (e.g. a safety relay in a de-energised trip
circuit) can fail silently — it enters a failed state but this is only
revealed when the device is demanded or explicitly tested. Failures
accumulate monotonically between tests. The relevant safety metric is:

    PFDavg  =  Probability of Failure on Demand (averaged over test cycle)
            =  time-averaged probability that the device is in a failed
               state at a random demand instant within the test interval τ.

CORRECT FORMULA (IEC 61508-6 Annex B, Eq. B.4):

    PFDavg = (1/τ) ∫₀^τ [1 - exp(-λt)] dt
           = 1 - [1 - exp(-λτ)] / (λτ)
           ≈ λτ / 2                           [first-order, valid for λτ < 0.1]

⚠ PLAN DOCUMENT CORRECTION:
The Phase 7 pipeline plan document incorrectly wrote:
    PFDavg = [1 - exp(-λτ)] / (λτ)           ← WRONG (this is 1 - PFDavg)
All functions in this module use the CORRECT formula above.
Verification: PFDavg_correct(7.571e-5 /d, 730d) = 0.027132 = Q_G3b ✓

Phase 7 finding (derivation below, committed to failure_finding_table.csv):
    Q_G3b = 0.027132  (Phase 5 gate_Q_table.csv)
    λ_G3b = 7.571×10⁻⁵ /day  (inferred at τ_implied = 730d)
    τ_SIL2 ≈ 26.4 days (≈ monthly proof-testing required for PFDavg ≤ 10⁻³)

This is the direct, quantified follow-through on Phase 6's ALARP Option 1:
"replace G3b with SIL-2 relay" is meaningless without a proof-test interval
that maintains the SIL-2 PFDavg ≤ 10⁻³ rating over the device's service life.

References:
    IEC 61508-6:2010 Annex B — Reliability data for E/E/PE elements.
    IEC 61511:2016 Section 6.7 — Proof test interval calculation.
    Smith, D.J. (2011) "Reliability, Maintainability and Risk." 8th ed. Elsevier.
    Rausand, M. (2014) "Reliability of Safety-Critical Systems." Wiley. Ch. 5.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Core PFDavg formulae
# ---------------------------------------------------------------------------

def PFDavg_exact(lambda_per_day: float, tau_days: float) -> float:
    """Exact PFDavg for a single-channel dormant device.

    PFDavg = 1 - [1 - exp(-λτ)] / (λτ)

    Parameters
    ----------
    lambda_per_day : float
        Constant failure rate (1/MTBF in days). Must be > 0.
    tau_days : float
        Proof-test interval (days). Must be > 0.

    Returns
    -------
    float
        PFDavg in [0, 1].

    Notes
    -----
    For λτ → 0: PFDavg → 0 (continuous testing → perfect safety).
    For λτ → ∞: PFDavg → 1 (never tested → always in unknown state).
    The function is strictly increasing in both λ and τ.

    Assumption: 100% proof-test effectiveness. The test reveals and repairs
    all latent faults. For imperfect testing, multiply λ by (1-coverage)
    where coverage is the fraction of faults detected — not implemented here
    but straightforward to add for Phase 8+ if required.
    """
    if lambda_per_day <= 0.0 or tau_days <= 0.0:
        return 0.0

    x = lambda_per_day * tau_days

    # Numerically stable evaluation: for very small x, use Taylor series
    # to avoid catastrophic cancellation in 1 - (1-exp(-x))/x
    if x <= 1e-6:
        # Taylor: 1 - (1-x+x²/2-x³/6+...)/x = 1 - (1/x - 1 + x/2 - ...) → x/2 - x²/6 + ...
        return x / 2.0 - x ** 2 / 6.0 + x ** 3 / 24.0

    return 1.0 - (1.0 - math.exp(-x)) / x


def PFDavg_approx(lambda_per_day: float, tau_days: float) -> float:
    """First-order approximation: PFDavg ≈ λτ/2.

    Valid to within 1% for λτ < 0.05, within 4% for λτ < 0.1.
    For G3b: λτ = 7.571e-5 × 730 = 0.05527 → approx error < 3%.
    Included for comparison and in the PFD table.
    """
    return lambda_per_day * tau_days / 2.0


# ---------------------------------------------------------------------------
# Lambda derivation from observed PFD
# ---------------------------------------------------------------------------

def derive_lambda_from_Q(
    Q_observed: float,
    tau_implied_days: float,
) -> float:
    """Recover failure rate λ from an observed PFDavg and implied test interval.

    Inverts PFDavg_exact(λ, τ) = Q numerically using Brent's method.

    Used to convert Phase 5 gate Q values (which are PFDavg values under the
    failure-finding model) into per-day failure rates suitable for deriving
    the required proof-test interval for SIL-2 compliance.

    Parameters
    ----------
    Q_observed : float
        Observed PFDavg (= Q from Phase 5 gate_Q_table.csv for G3b).
    tau_implied_days : float
        The proof-test interval that produced this Q. For G3b, this is
        the current (implicit) inspection cycle = 730d (2yr major O&M).

    Returns
    -------
    float
        Failure rate λ (1/day) such that PFDavg_exact(λ, tau_implied) = Q_observed.

    Raises
    ------
    ValueError
        If Q_observed ≥ 1 or ≤ 0, or if no root is found in the search range.
    """
    if not (0.0 < Q_observed < 1.0):
        raise ValueError(f"Q_observed={Q_observed} must be in (0, 1)")
    if tau_implied_days <= 0.0:
        raise ValueError(f"tau_implied_days={tau_implied_days} must be > 0")

    def objective(lam: float) -> float:
        return PFDavg_exact(lam, tau_implied_days) - Q_observed

    # Upper bound: even at λτ = 100, PFDavg ≈ 1; search up to 100/τ
    lam_upper = 100.0 / tau_implied_days
    return float(brentq(objective, 1e-12, lam_upper, xtol=1e-14))


# ---------------------------------------------------------------------------
# Required proof-test interval for a SIL target
# ---------------------------------------------------------------------------

def required_proof_test_interval(
    lambda_per_day: float,
    PFDavg_target: float,
    use_exact: bool = True,
) -> float:
    """Find the maximum proof-test interval τ_max that maintains PFDavg ≤ target.

    Parameters
    ----------
    lambda_per_day : float
        Constant failure rate (1/day).
    PFDavg_target : float
        Target PFDavg (e.g. 1e-3 for SIL-2).
    use_exact : bool
        If True, solve PFDavg_exact(λ, τ) = target numerically.
        If False, use the approximation τ ≈ 2 × target / λ.

    Returns
    -------
    float
        τ_max (days). To maintain SIL, proof-test at intervals ≤ τ_max.

    Notes
    -----
    The approximation is accurate to within 1% for λτ < 0.05:
        τ_approx = 2 × PFDavg_target / λ
    For G3b at SIL-2:
        τ_approx = 2 × 1e-3 / 7.571e-5 = 26.42d
        τ_exact  = 26.43d (difference < 0.1%)
    Both are reported in the failure_finding_table.csv.
    """
    tau_approx = 2.0 * PFDavg_target / lambda_per_day

    if not use_exact:
        return tau_approx

    def objective(tau: float) -> float:
        return PFDavg_exact(lambda_per_day, tau) - PFDavg_target

    # The function PFDavg_exact is monotonically increasing in tau.
    # At τ_approx×2, PFDavg > target (upper bracket).
    # At τ_approx×0.1, PFDavg < target (lower bracket).
    tau_lo = tau_approx * 0.05
    tau_hi = tau_approx * 5.0
    # Ensure bounds bracket the root
    if objective(tau_lo) > 0.0:
        tau_lo = 1e-6
    if objective(tau_hi) < 0.0:
        tau_hi *= 10.0

    return float(brentq(objective, tau_lo, tau_hi, xtol=1e-6))


# ---------------------------------------------------------------------------
# Proof-test interval table (for plotting and CSV output)
# ---------------------------------------------------------------------------

def proof_test_interval_table(
    lambda_per_day: float,
    component: str,
    tau_range_days: np.ndarray | None = None,
    SIL_targets: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build PFDavg vs. τ table with SIL classification at each τ.

    Parameters
    ----------
    lambda_per_day : float
        Component failure rate (1/day).
    component : str
        Component name (for labelling).
    tau_range_days : np.ndarray, optional
        Proof-test intervals to evaluate. Defaults to config.TAU_SWEEP_DAYS.
    SIL_targets : dict, optional
        Maps SIL label → PFDavg upper bound. Defaults to config.SIL_TARGETS.

    Returns
    -------
    pd.DataFrame
        Columns: component, tau_days, tau_months, PFDavg_exact, PFDavg_approx,
                 lambda_tau, SIL_achieved.
        One row per τ value.
    """
    from .config import TAU_SWEEP_DAYS, SIL_TARGETS as _DEFAULT_SIL

    if tau_range_days is None:
        tau_range_days = TAU_SWEEP_DAYS
    if SIL_targets is None:
        SIL_targets = _DEFAULT_SIL

    # Sorted by PFDavg threshold ascending (strictest SIL first for classification)
    sil_sorted = sorted(SIL_targets.items(), key=lambda kv: kv[1])

    rows = []
    for tau in tau_range_days:
        pfd_ex  = PFDavg_exact(lambda_per_day, tau)
        pfd_app = PFDavg_approx(lambda_per_day, tau)
        lam_tau = lambda_per_day * tau

        # Classify: highest SIL achieved = most restrictive target still met
        sil_achieved = "BELOW SIL-1"
        for sil_label, threshold in sil_sorted:
            if pfd_ex <= threshold:
                sil_achieved = sil_label
                break

        rows.append({
            "component":      component,
            "tau_days":       tau,
            "tau_months":     tau / 30.44,
            "lambda_tau":     lam_tau,
            "PFDavg_exact":   pfd_ex,
            "PFDavg_approx":  pfd_app,
            "SIL_achieved":   sil_achieved,
        })

    return pd.DataFrame(rows)