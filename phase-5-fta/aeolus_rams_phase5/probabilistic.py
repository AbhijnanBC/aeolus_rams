"""
aeolus_rams_phase5.probabilistic
==================================
Probabilistic FTA — propagating MTBF uncertainty through the fault tree
to produce a distribution on Q_top rather than a point estimate.

Uncertainty sources:
  1. Pitch System MTBF: Phase 2 bootstrap 95% CI ∈ [1017, 2659] days
     (only Tier A component; CI derived from Weibull β bootstrap).
     Approximated by a log-normal distribution fit to the CI bounds.

  2. Hydraulic System MTBF: Phase 2 Tier B fitted; no explicit CI in
     mtbf_table.csv. Approximate ±40% (conservative for fitted component).

  3. Mechanical Brake, Electrical Safety System: assumed_placeholder MTBFs.
     Use ±50% range (config.PLACEHOLDER_MTBF_UNCERTAINTY_FRACTION).

  4. SCADA/Communication: posterior_informed. Use ±30% (better informed).

Uncertainty in the sub-cause fractions is NOT propagated here — that is
addressed by ccf.py::subcause_fraction_sensitivity(). These are kept
separate because they are different types of uncertainty:
  - MTBF uncertainty = aleatory/epistemic on failure rates
  - Fraction uncertainty = structural uncertainty on the fault tree topology

Reference:
    Aven, T. (2012) "Foundations of Risk Analysis." 2nd ed. Wiley. Ch. 5.
    Vaurio, J.K. (1994) "Treatment of general dependencies in system failure
    and risk analysis." IEEE Trans. Reliability, 43(3):460–466.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .fault_tree import build_overspeed_fault_tree, Gate


def Q_uncertainty_from_mtbf_ci(
    mtbf_point: float,
    mtbf_ci_low: float,
    mtbf_ci_high: float,
    t: float = 365.25,
    n_samples: int = 50_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample Q(t) from a log-normal MTBF distribution fit to a 95% CI.

    Parameters
    ----------
    mtbf_point : float
        Point estimate of MTBF (days).
    mtbf_ci_low, mtbf_ci_high : float
        Lower and upper bounds of the 95% CI (days).
        Assumed to span 2×1.96 standard deviations on the log scale.
    t : float
        Mission time (days). Default: 365.25 (1 year).
    n_samples : int
        Number of Monte Carlo samples.
    rng : np.random.Generator, optional
        Seeded generator for reproducibility.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Q(t) = 1 - exp(-t / MTBF_sample) for each sample.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Fit log-normal to CI: μ = ln(MTBF_point), σ from CI width
    log_mu = np.log(mtbf_point)
    log_sigma = (np.log(mtbf_ci_high) - np.log(mtbf_ci_low)) / (2 * 1.96)
    log_sigma = max(log_sigma, 1e-6)  # guard against zero width

    mtbf_samples = rng.lognormal(mean=log_mu, sigma=log_sigma, size=n_samples)
    mtbf_samples = np.maximum(mtbf_samples, 1.0)  # MTBF must be ≥ 1 day

    Q_samples = 1.0 - np.exp(-t / mtbf_samples)
    return Q_samples


def _build_uncertainty_params(
    t: float = 365.25,
) -> dict[str, tuple[float, float, float]]:
    """Return (mtbf_point, ci_low, ci_high) for each uncertain component.

    Sources for CI bounds:
      Pitch System:       Phase 2 bootstrap 95% CI [1017, 2659] days
      Hydraulic System:   Tier B fitted — ±40% symmetric on log scale
      Mechanical Brake:   assumed_placeholder — ±50%
      SCADA/Communication: posterior_informed — ±30%
      Electrical Safety:  assumed_placeholder — ±50%
    """
    from . import config as _cfg

    Q_1yr, _ = _cfg.load_Q_values()

    def _mtbf_from_Q(Q: float, t: float) -> float:
        """Recover MTBF from Q(t) = 1 - exp(-t/MTBF)."""
        if Q >= 1.0:
            return 1.0
        return -t / np.log(1.0 - Q)

    params = {}

    # Pitch System — Phase 2 bootstrap CI
    params["Pitch System"] = (
        _cfg.PITCH_MTBF_POINT,
        _cfg.PITCH_MTBF_CI_LOW,
        _cfg.PITCH_MTBF_CI_HIGH,
    )

    # Hydraulic System — fitted, use ±40%
    hyd_mtbf = _mtbf_from_Q(Q_1yr["Hydraulic System"], t)
    params["Hydraulic System"] = (hyd_mtbf, hyd_mtbf * 0.60, hyd_mtbf * 1.40)

    # Mechanical Brake — placeholder, use ±50%
    brake_mtbf = _mtbf_from_Q(Q_1yr["Mechanical Brake"], t)
    params["Mechanical Brake"] = (brake_mtbf, brake_mtbf * 0.50, brake_mtbf * 1.50)

    # SCADA/Communication — posterior_informed, use ±30%
    scada_mtbf = _mtbf_from_Q(Q_1yr["SCADA/Communication"], t)
    params["SCADA/Communication"] = (scada_mtbf, scada_mtbf * 0.70, scada_mtbf * 1.30)

    # Electrical Safety System — placeholder, use ±50%
    esys_mtbf = _mtbf_from_Q(Q_1yr["Electrical Safety System"], t)
    params["Electrical Safety System"] = (esys_mtbf, esys_mtbf * 0.50, esys_mtbf * 1.50)

    return params


def probabilistic_fta(
    t: float = 365.25,
    n_samples: int = 50_000,
    seed: int = 42,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Run probabilistic FTA: propagate MTBF uncertainty to Q_top distribution.

    For each of n_samples:
      1. Sample MTBF from log-normal distribution for each uncertain component.
      2. Compute Q(t) = 1 - exp(-t/MTBF_sample).
      3. Build the fault tree with these Q values.
      4. Compute Q_top (AND × three OR gates).
      5. Record Q_top sample.

    Returns
    -------
    (Q_top_samples, summary_df) : tuple
        Q_top_samples: np.ndarray of shape (n_samples,)
        summary_df: DataFrame with percentiles, mean, CI, return period CI.
    """
    from . import config as _cfg

    rng = np.random.default_rng(seed)
    uncertainty_params = _build_uncertainty_params(t)

    # Pre-sample all component MTBFs
    sampled_Q: dict[str, np.ndarray] = {}
    for comp, (mtbf_p, mtbf_lo, mtbf_hi) in uncertainty_params.items():
        sampled_Q[comp] = Q_uncertainty_from_mtbf_ci(
            mtbf_p, mtbf_lo, mtbf_hi, t=t, n_samples=n_samples, rng=rng
        )

    # Load fallback values for components not in uncertainty_params
    Q_1yr_base = _cfg.FALLBACK_Q_1YR

    Q_top_samples = np.empty(n_samples)
    for i in range(n_samples):
        # Build Q_1yr dict for this sample
        Q_sample: dict[str, float] = dict(Q_1yr_base)
        for comp, q_arr in sampled_Q.items():
            Q_sample[comp] = float(q_arr[i])

        tree = build_overspeed_fault_tree(Q_sample)
        Q_top_samples[i] = tree.Q

    # Summary statistics
    pct = [5, 25, 50, 75, 95]
    summary = pd.DataFrame([{
        "n_samples":    n_samples,
        "t_days":       t,
        "mean_Q_top":   Q_top_samples.mean(),
        "std_Q_top":    Q_top_samples.std(),
        **{f"p{p}_Q_top": np.percentile(Q_top_samples, p) for p in pct},
        "mean_return_period_yr": 1.0 / Q_top_samples.mean() if Q_top_samples.mean() > 0 else np.inf,
        "p5_return_period_yr":   1.0 / np.percentile(Q_top_samples, 95),  # high Q → short RP
        "p95_return_period_yr":  1.0 / np.percentile(Q_top_samples, 5),   # low Q → long RP
    }])

    return Q_top_samples, summary