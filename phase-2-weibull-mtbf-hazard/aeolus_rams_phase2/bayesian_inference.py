"""
aeolus_rams_phase2.bayesian_inference
======================================
Section 2.4 — Bayesian Tier B↔C Bridge

**Problem:** Tier C components with 3-4 farm observations are discarded entirely.
**Solution:** Use Bayesian inference to blend literature priors with small observed samples.

When a component has too few observations for direct fitting (< MIN_TBF_FOR_EXPONENTIAL),
instead of dropping it to pure Tier C, we use a conjugate gamma-Poisson model:

  Prior: λ ~ Gamma(α₀, β₀)  where E[λ]=α₀/β₀ from literature MTBF
  Likelihood: n_failures ~ Poisson(λ·T_obs)  where T_obs = total observation time
  Posterior: λ ~ Gamma(α₀+n, β₀+T_obs)

The posterior mean λ̂ = (α₀+n) / (β₀+T_obs) produces a Bayesian MTBF estimate that
respects both the literature ("Gearbox typically ~2372 days") and the farm evidence
("but we saw 4 failures in 800 days = higher than expected").

This is particularly powerful when observations suggest a failure rate HIGHER than
literature — it identifies components that deserve more attention or maintenance
adjustments for YOUR operating environment.

References:
  - Gelman et al. (2013) "Bayesian Data Analysis" §3.2 (Poisson conjugate prior)
  - Meeker & Escobar (1998) "Statistical Methods for Reliability Data" §8.3
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BayesianPosterior:
    """Bayesian posterior estimate for a component's failure rate."""
    component: str
    n_observed_failures: int
    t_observation_days: float
    prior_mtbf_days: float
    posterior_mtbf_days: float
    posterior_lambda: float
    prior_lambda: float
    likelihood_lambda: float
    confidence: str  # "prior_only" | "weak_likelihood" | "posterior_informed"
    alpha_prior: float
    beta_prior: float
    alpha_posterior: float
    beta_posterior: float

    @property
    def prior_lambda_to_posterior_ratio(self) -> float:
        """How much did the prior shift due to observations?
        ratio > 1 means observed data suggests HIGHER failure rate than literature.
        ratio < 1 means observed data suggests LOWER failure rate than literature.
        ratio ~= 1 means data and prior are consistent."""
        if self.prior_lambda < 1e-10:
            return 1.0
        return self.posterior_lambda / self.prior_lambda


def derive_gamma_prior_from_mtbf(mtbf_days: float, confidence_weeks: float = 52.0) -> tuple[float, float]:
    """Convert a literature MTBF (days) into gamma prior parameters α, β.

    Strategy: Assume the literature MTBF is the prior's MEAN, and choose a spread
    based on how confident we are in that literature value.

    Parameters
    ----------
    mtbf_days : float
        Literature prior MTBF in days (e.g., 2372 for Gearbox).
    confidence_weeks : float, default=52
        Pseudo-observation period used to calibrate the prior's "strength".
        If literature says "MTBF ~2372 days" with moderate confidence,
        we model that as: "if we ran the component for confidence_weeks,
        we'd expect ~(confidence_weeks*7)/mtbf_days failures."

    Returns
    -------
    alpha, beta : float
        Gamma(α, β) parameters such that E[λ] = α/β = 1/mtbf_days.
        Higher confidence_weeks → lower variance → more "stubborn" prior.
    """
    if mtbf_days <= 0:
        raise ValueError(f"mtbf_days must be positive, got {mtbf_days}")

    prior_lambda = 1.0 / mtbf_days
    confidence_days = confidence_weeks * 7
    
    # Confidence interpretation: expected failures in confidence_period
    # α = n_expected_failures, β = 1 / (λ * confidence_days)
    # This makes α/β = n_expected_failures / confidence_days = λ
    alpha = confidence_days / mtbf_days  # E[n_failures in confidence_period]
    beta = confidence_days  # Scaling to recover E[λ] = α/β = 1/mtbf_days

    return alpha, beta


def bayesian_posterior_poisson(
    component: str,
    n_failures: int,
    t_observation_days: float,
    prior_mtbf_days: float,
    prior_confidence_weeks: float = 52.0,
) -> BayesianPosterior:
    """Compute posterior belief about component's failure rate using Poisson likelihood.

    Parameters
    ----------
    component : str
        Component name for logging/identification.
    n_failures : int
        Number of failures observed in farm data (e.g., 3 or 4).
    t_observation_days : float
        Total observation time (days) across all assets for this component.
        E.g., if 5 turbines each operated 400 days, this would be 2000.
    prior_mtbf_days : float
        Literature-informed prior MTBF (days).
    prior_confidence_weeks : float, default=52
        Pseudo-observations encoded into prior. Increase for more literature confidence,
        decrease if literature is uncertain or applies to different turbine size/vintage.

    Returns
    -------
    BayesianPosterior
        Posterior inference, including posterior_mtbf_days and confidence assessment.
    """
    alpha_prior, beta_prior = derive_gamma_prior_from_mtbf(
        prior_mtbf_days, prior_confidence_weeks
    )
    
    # Likelihood: observed failures under constant rate model
    prior_lambda = alpha_prior / beta_prior
    likelihood_lambda = n_failures / t_observation_days if t_observation_days > 0 else 0.0
    
    # Conjugate posterior: Gamma(α + n, β + T)
    alpha_posterior = alpha_prior + n_failures
    beta_posterior = beta_prior + t_observation_days
    posterior_lambda = alpha_posterior / beta_posterior
    posterior_mtbf_days = 1.0 / posterior_lambda if posterior_lambda > 0 else float("inf")
    
    # Assess confidence in the posterior
    if n_failures == 0:
        confidence = "prior_only"
        confidence_reason = "no failures observed — posterior equals prior"
    elif n_failures < 2:
        confidence = "weak_likelihood"
        confidence_reason = f"only {n_failures} failure(s) — prior remains strong"
    else:
        confidence = "posterior_informed"
        confidence_reason = f"{n_failures} failures provides meaningful update"
    
    logger.info(
        "%s: Bayesian posterior. Prior MTBF %.0f days → Posterior MTBF %.0f days. "
        "%s. (Prior λ=%.4e, Likelihood λ=%.4e, Posterior λ=%.4e)",
        component, prior_mtbf_days, posterior_mtbf_days, confidence_reason,
        prior_lambda, likelihood_lambda, posterior_lambda,
    )
    
    return BayesianPosterior(
        component=component,
        n_observed_failures=n_failures,
        t_observation_days=t_observation_days,
        prior_mtbf_days=prior_mtbf_days,
        posterior_mtbf_days=posterior_mtbf_days,
        posterior_lambda=posterior_lambda,
        prior_lambda=prior_lambda,
        likelihood_lambda=likelihood_lambda,
        confidence=confidence,
        alpha_prior=alpha_prior,
        beta_prior=beta_prior,
        alpha_posterior=alpha_posterior,
        beta_posterior=beta_posterior,
    )


@dataclass(frozen=True)
class BayesianWeibullResult:
    """Bayesian update using literature Weibull shape, farm-observed scale."""
    component: str
    beta_from_literature: float
    eta_posterior_days: float
    mtbf_posterior_days: float
    n_observed_failures: int
    confidence: str
    source: str  # "posterior_weibull" | "posterior_exponential_only"


def bayesian_weibull_with_literature_shape(
    component: str,
    literature_beta: float,
    literature_mtbf_days: float,
    n_failures_observed: int,
    t_observation_days: float,
    allow_exponential_fallback: bool = True,
) -> BayesianWeibullResult:
    """Advanced: Use literature shape parameter (β) with farm-observed scale (η).

    When literature says "Gearbox failures typically have β≈1.6 (wear-out)",
    but we only have 3-4 farm failures, we can:
      1. Fix β = 1.6 (from literature or meta-analysis)
      2. Fit only η (scale) from observations
      3. Get a Weibull curve tuned to our specific environment

    This is more powerful than Bayesian exponential (which forces β≈1) because
    it respects the known failure MODE (wear-out vs random) while adapting
    to local conditions.

    Parameters
    ----------
    component : str
        Component name.
    literature_beta : float
        Shape parameter from literature or industry source (e.g., 1.6 for Gearbox).
    literature_mtbf_days : float
        Literature MTBF for calibration.
    n_failures_observed : int
        Number of failures observed locally.
    t_observation_days : float
        Total observation time (days).
    allow_exponential_fallback : bool, default=True
        If n_failures < 2, fall back to Bayesian exponential instead of failing.

    Returns
    -------
    BayesianWeibullResult
        Posterior Weibull with fixed β and fitted η.
    """
    from scipy.special import gamma as gamma_func
    
    if n_failures_observed < 1:
        if allow_exponential_fallback:
            logger.warning(
                "%s: no observed failures — falling back to Bayesian exponential",
                component,
            )
            # Use Bayesian exponential as fallback
            posterior = bayesian_posterior_poisson(
                component, 0, t_observation_days, literature_mtbf_days
            )
            return BayesianWeibullResult(
                component=component,
                beta_from_literature=1.0,  # Exponential
                eta_posterior_days=posterior.posterior_mtbf_days,
                mtbf_posterior_days=posterior.posterior_mtbf_days,
                n_observed_failures=0,
                confidence="prior_only",
                source="posterior_exponential_only",
            )
        else:
            raise ValueError(f"{component}: no observed failures and fallback disabled")
    
    # For a Weibull with fixed β and observed failures in time T:
    # Simple MLE for η: η̂ = (T / n)^(1/β) * Γ(1 + 1/β)
    # (Exact formula depends on right-censoring; this is uncensored approximation)
    
    eta_posterior = (t_observation_days / n_failures_observed) ** (1.0 / literature_beta) * gamma_func(1 + 1.0 / literature_beta)
    mtbf_posterior = eta_posterior * gamma_func(1 + 1.0 / literature_beta)
    
    logger.info(
        "%s: Bayesian Weibull (fixed β=%.2f from literature). "
        "n_failures=%d in %.0f days → η̂=%.0f days, MTBF=%.0f days.",
        component, literature_beta, n_failures_observed, t_observation_days,
        eta_posterior, mtbf_posterior,
    )
    
    return BayesianWeibullResult(
        component=component,
        beta_from_literature=literature_beta,
        eta_posterior_days=eta_posterior,
        mtbf_posterior_days=mtbf_posterior,
        n_observed_failures=n_failures_observed,
        confidence="posterior_informed" if n_failures_observed >= 2 else "weak_likelihood",
        source="posterior_weibull",
    )
