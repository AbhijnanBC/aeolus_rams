"""
aeolus_rams_phase2.distributions
====================================
Section 2.4 — Distribution Fitting.

The central design decision in this module: the small-sample danger
Section 2.4 warns about in prose ("a 2-parameter Weibull MLE fit through
2 data points will always 'succeed' numerically and always be
meaningless") is enforced here as a HARD GUARD in code, not just a
comment. `fit_weibull` and `fit_exponential_only` raise
`InsufficientDataError` below their respective minimums unless the
caller explicitly passes `allow_unsafe=True` — and even then, the result
carries a loud `warnings` field so it can never be silently mistaken for
a trustworthy fit downstream (in a report table, a CSV export, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.special import gamma

from . import config


class InsufficientDataError(ValueError):
    """Raised when a fit is requested on too few usable TBF points.
    This is the code-level enforcement of Section 2.4's core warning —
    catching it (or passing allow_unsafe=True) is a deliberate choice,
    never an accident of scipy silently returning a fit anyway."""


@dataclass(frozen=True)
class WeibullFitResult:
    beta: float
    eta: float
    mtbf_days: float
    aic_weibull: float
    aic_exponential: float
    preferred: str            # "weibull" | "exponential"
    lambda_exp: float
    mtbf_exp_days: float
    n_used: int
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def wearout_interpretation(self) -> str:
        if self.beta > 1.05:
            return "beta > 1: wear-out (hazard rises with age)"
        if self.beta < 0.95:
            return "beta < 1: infant mortality (hazard falls with age)"
        return "beta ~= 1: effectively random (age-independent failures)"


@dataclass(frozen=True)
class ExponentialFitResult:
    lambda_rate: float
    mtbf_days: float
    n_used: int
    warnings: tuple[str, ...] = field(default_factory=tuple)


def fit_weibull(
    tbf_days: np.ndarray,
    min_n: int = config.MIN_TBF_FOR_WEIBULL,
    allow_unsafe: bool = False,
) -> WeibullFitResult:
    """Tier A — full 2-parameter Weibull fit, always compared against the
    simpler exponential fit via AIC (never assume 2 parameters is
    automatically the better model — Section 2.4's own instruction).

    Raises `InsufficientDataError` if `len(tbf_days) < min_n` unless
    `allow_unsafe=True`. When forced through anyway, the result's
    `.warnings` tuple documents exactly why it shouldn't be trusted —
    see `demonstrate_degenerate_fit` below for the textbook example of
    why this guard exists.
    """
    tbf_days = np.asarray(tbf_days, dtype=float)
    n = len(tbf_days)
    warnings: list[str] = []

    if n < min_n:
        msg = (
            f"fit_weibull called with only {n} usable TBF point(s); "
            f"Section 2.4 requires >= {min_n} for a trustworthy "
            f"2-parameter fit. A fit through this few points has "
            f"essentially zero degrees of freedom to validate anything "
            f"and will look plausible while being meaningless."
        )
        if not allow_unsafe:
            raise InsufficientDataError(msg)
        warnings.append("UNSAFE FIT: " + msg)

    if n < 2:
        raise InsufficientDataError(
            f"fit_weibull cannot fit anything with n={n} points, even "
            f"with allow_unsafe=True — scipy needs at least 2."
        )

    beta_hat, _, eta_hat = stats.weibull_min.fit(tbf_days, floc=0)
    mtbf = float(eta_hat * gamma(1 + 1 / beta_hat))

    _, scale_exp = stats.expon.fit(tbf_days, floc=0)
    ll_weibull = float(np.sum(stats.weibull_min.logpdf(tbf_days, beta_hat, loc=0, scale=eta_hat)))
    ll_exp = float(np.sum(stats.expon.logpdf(tbf_days, loc=0, scale=scale_exp)))
    aic_weibull = 2 * 2 - 2 * ll_weibull
    aic_exp = 2 * 1 - 2 * ll_exp

    return WeibullFitResult(
        beta=float(beta_hat),
        eta=float(eta_hat),
        mtbf_days=mtbf,
        aic_weibull=aic_weibull,
        aic_exponential=aic_exp,
        preferred="weibull" if aic_weibull < aic_exp else "exponential",
        lambda_exp=float(1 / scale_exp),
        mtbf_exp_days=float(scale_exp),
        n_used=n,
        warnings=tuple(warnings),
    )


def fit_exponential_only(
    tbf_days: np.ndarray,
    min_n: int = config.MIN_TBF_FOR_EXPONENTIAL,
    allow_unsafe: bool = False,
) -> ExponentialFitResult:
    """Tier B — deliberately NOT fitting a shape parameter. A handful of
    points can support a single-parameter (rate-only) estimate far more
    reliably than a 2-parameter shape+scale fit — see the n=2 degenerate
    case in `demonstrate_degenerate_fit`."""
    tbf_days = np.asarray(tbf_days, dtype=float)
    n = len(tbf_days)
    warnings: list[str] = []

    if n < min_n:
        msg = (
            f"fit_exponential_only called with only {n} usable TBF "
            f"point(s); Section 2.4 requires >= {min_n} even for a "
            f"1-parameter fit."
        )
        if not allow_unsafe:
            raise InsufficientDataError(msg)
        warnings.append("UNSAFE FIT: " + msg)

    if n < 1:
        raise InsufficientDataError("fit_exponential_only needs at least 1 point.")

    _, scale = stats.expon.fit(tbf_days, floc=0)
    return ExponentialFitResult(
        lambda_rate=float(1 / scale),
        mtbf_days=float(scale),
        n_used=n,
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Section 2.4 (Advanced) — Right-Censoring Aware Fitting
# ---------------------------------------------------------------------------

def fit_weibull_with_censoring(
    tbf_complete: np.ndarray,
    tbf_censored: np.ndarray | None = None,
    min_n: int = config.MIN_TBF_FOR_WEIBULL,
    allow_unsafe: bool = False,
) -> WeibullFitResult:
    """Tier A — Weibull fit with explicit right-censoring support.

    This is the correct MLE approach when some TBF intervals are right-censored
    (component hasn't failed yet; we only know it lasted at least this long).

    Parameters
    ----------
    tbf_complete : np.ndarray
        Observed failure times (uncensored intervals).
    tbf_censored : np.ndarray, optional
        Right-censored times (last observation, component still running).
        If None, no censoring (same as vanilla fit_weibull).
    min_n : int
        Minimum uncensored observations for trust.
    allow_unsafe : bool
        Allow fit on too few samples with warnings.

    Returns
    -------
    WeibullFitResult
        Same as fit_weibull; warnings note if censoring was incorporated.

    Notes
    -----
    Implementation: scipy's `weibull_min.fit` doesn't handle censoring natively.
    For now, we use a simplified approach:
      1. Fit to uncensored data (current behavior)
      2. Log how many censored obs are present
      3. Document that a production system should use lifelines.KaplanMeierFitter
         or lifelines.WeibullAFTFitter for full Kaplan-Meier + MLE integration
    """
    tbf_complete = np.asarray(tbf_complete, dtype=float)
    n_complete = len(tbf_complete)
    n_censored = len(tbf_censored) if tbf_censored is not None else 0
    warnings: list[str] = []

    if n_censored > 0:
        warnings.append(
            f"CENSORING PRESENT: {n_censored} right-censored obs; fit uses only "
            f"{n_complete} complete intervals. Production fitting should use "
            f"lifelines.WeibullAFTFitter or Kaplan-Meier MLE for full integration."
        )

    # Fall back to standard uncensored fit (with warning)
    if n_censored > 0 or True:  # Always note availability
        result = fit_weibull(tbf_complete, min_n=min_n, allow_unsafe=allow_unsafe)
        return WeibullFitResult(
            beta=result.beta,
            eta=result.eta,
            mtbf_days=result.mtbf_days,
            aic_weibull=result.aic_weibull,
            aic_exponential=result.aic_exponential,
            preferred=result.preferred,
            lambda_exp=result.lambda_exp,
            mtbf_exp_days=result.mtbf_exp_days,
            n_used=n_complete,
            warnings=tuple(list(result.warnings) + warnings),
        )


def fit_exponential_with_censoring(
    tbf_complete: np.ndarray,
    tbf_censored: np.ndarray | None = None,
    min_n: int = config.MIN_TBF_FOR_EXPONENTIAL,
    allow_unsafe: bool = False,
) -> ExponentialFitResult:
    """Tier B — Exponential fit with explicit right-censoring support.

    For exponential, the censoring adjustment is simpler than Weibull:
    λ̂ = (# failures) / (total observation time), which naturally handles censoring.

    Parameters
    ----------
    tbf_complete : np.ndarray
        Observed failure times (uncensored intervals).
    tbf_censored : np.ndarray, optional
        Right-censored times (component hadn't failed yet).

    Returns
    -------
    ExponentialFitResult
        With warnings if censoring affects interpretation.
    """
    tbf_complete = np.asarray(tbf_complete, dtype=float)
    tbf_censored = np.asarray(tbf_censored, dtype=float) if tbf_censored is not None else np.array([])
    
    n_complete = len(tbf_complete)
    n_censored = len(tbf_censored)
    warnings: list[str] = []

    if n_censored == 0:
        # No censoring; use standard fit
        return fit_exponential_only(tbf_complete, min_n=min_n, allow_unsafe=allow_unsafe)

    # For exponential with censoring: λ̂ = n_failures / T_total
    # where T_total = sum(complete) + sum(censored)
    total_time = np.sum(tbf_complete) + np.sum(tbf_censored)
    lambda_hat = n_complete / total_time if total_time > 0 else 0.0

    if n_complete < min_n:
        msg = (
            f"fit_exponential_with_censoring: only {n_complete} complete failures "
            f"(+ {n_censored} censored obs); < {min_n} minimum."
        )
        if not allow_unsafe:
            raise InsufficientDataError(msg)
        warnings.append("UNSAFE FIT: " + msg)

    warnings.append(
        f"CENSORING INCORPORATED: {n_complete} complete failures over "
        f"{total_time:.0f} days total (incl. {n_censored} censored intervals). "
        f"λ̂ = {lambda_hat:.6f}/day."
    )

    return ExponentialFitResult(
        lambda_rate=float(lambda_hat),
        mtbf_days=float(1.0 / lambda_hat) if lambda_hat > 0 else float("inf"),
        n_used=n_complete,
        warnings=tuple(warnings),
    )


def demonstrate_degenerate_fit(
    tbf_days: np.ndarray = np.array([56.9, 217.3]),
) -> WeibullFitResult:
    """Reproduces Section 2.4's own cautionary example: a 2-parameter
    Weibull "fit" through only 2 points runs without error and produces
    plausible-looking numbers (beta~1.79, eta~154.9 for the spec's own
    sample) — and is meaningless, because there are zero degrees of
    freedom left to validate anything. Exists so this danger is something
    you can run and see, not just read about; `tests/test_distributions.py`
    asserts this reproduces the documented numbers."""
    return fit_weibull(tbf_days, allow_unsafe=True)
