"""
aeolus_rams_phase2.bootstrap
================================
Section 2.7 — Uncertainty Quantification via Bootstrap.

Same guarding philosophy as `distributions.py`: bootstrapping 2-3 points
just resamples the same 2-3 values in different combinations and returns
an interval that *looks* scientific while encoding almost no real
information (Section 2.7's own warning). `bootstrap_weibull_ci` and
`bootstrap_exponential_ci` refuse to run below their tier's minimum
unless explicitly forced, exactly like the fitting functions they wrap.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from . import config
from .distributions import InsufficientDataError


@dataclass(frozen=True)
class BootstrapResult:
    beta_ci: tuple[float, float] | None
    eta_ci: tuple[float, float] | None
    lambda_ci: tuple[float, float] | None
    mtbf_ci_days: tuple[float, float] | None
    n_successful: int
    n_requested: int
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_meaningful(self) -> bool:
        """False for anything bootstrapped from a Tier-C-sized sample, or
        for a run where every resample failed to fit — use this to gate
        whether a CI is even worth printing in a report, rather than
        trusting fabricated-looking precision at face value."""
        if any("UNSAFE" in w for w in self.warnings):
            return False
        return self.mtbf_ci_days is not None


def bootstrap_weibull_ci(
    tbf_days: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    min_n: int = config.MIN_TBF_FOR_WEIBULL,
    allow_unsafe: bool = False,
) -> BootstrapResult:
    """Tier A — percentile bootstrap CI for (beta, eta, MTBF).

    Do NOT run this on Tier C components and report the resulting
    interval as meaningful (Section 2.7). If you want an uncertainty
    statement for Tier C, it should be the qualitative one
    `literature_priors` attaches to every prior, not a numeric interval
    from here.
    """
    tbf_days = np.asarray(tbf_days, dtype=float)
    n = len(tbf_days)
    warning_messages: list[str] = []

    if n < min_n:
        msg = (
            f"bootstrap_weibull_ci called with only {n} usable TBF "
            f"point(s); Section 2.7 requires >= {min_n}. Resampling this "
            f"few values produces an interval that looks scientific but "
            f"encodes almost no real information."
        )
        if not allow_unsafe:
            raise InsufficientDataError(msg)
        warning_messages.append("UNSAFE BOOTSTRAP: " + msg)

    if n < 2:
        raise InsufficientDataError("Cannot bootstrap a Weibull fit with fewer than 2 points.")

    rng = np.random.default_rng(seed)
    betas, etas = [], []
    for _ in range(n_boot):
        resample = rng.choice(tbf_days, size=n, replace=True)
        try:
            with warnings.catch_warnings():
                # Resampling with replacement from a tiny/degenerate
                # sample can produce near-duplicate values, which trips
                # scipy's internal precision-loss RuntimeWarning — expected
                # noise on the allow_unsafe path, not a real failure.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                b, _, e = stats.weibull_min.fit(resample, floc=0)
        except Exception:
            continue
        betas.append(b)
        etas.append(e)

    betas_arr = np.array(betas)
    etas_arr = np.array(etas)
    n_successful = len(betas_arr)

    if n_successful == 0:
        return BootstrapResult(
            beta_ci=None, eta_ci=None, lambda_ci=None, mtbf_ci_days=None,
            n_successful=0, n_requested=n_boot,
            warnings=tuple(warning_messages + ["All bootstrap resamples failed to fit."]),
        )

    from scipy.special import gamma as gamma_fn
    mtbf_samples = etas_arr * gamma_fn(1 + 1 / betas_arr)

    return BootstrapResult(
        beta_ci=(float(np.percentile(betas_arr, 2.5)), float(np.percentile(betas_arr, 97.5))),
        eta_ci=(float(np.percentile(etas_arr, 2.5)), float(np.percentile(etas_arr, 97.5))),
        lambda_ci=None,
        mtbf_ci_days=(float(np.percentile(mtbf_samples, 2.5)), float(np.percentile(mtbf_samples, 97.5))),
        n_successful=n_successful,
        n_requested=n_boot,
        warnings=tuple(warning_messages),
    )


def bootstrap_exponential_ci(
    tbf_days: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    min_n: int = config.MIN_TBF_FOR_EXPONENTIAL,
    allow_unsafe: bool = False,
) -> BootstrapResult:
    """Tier B — percentile bootstrap CI for (lambda, MTBF). Not required
    by Section 2.9's Definition of Done (only Tier A explicitly asks for
    a bootstrap CI), but implemented symmetrically since a 1-parameter
    fit's uncertainty is just as easy to quantify honestly, and Tier B
    components sit right at the data-sufficiency boundary where knowing
    the CI width matters most.
    """
    tbf_days = np.asarray(tbf_days, dtype=float)
    n = len(tbf_days)
    warning_messages: list[str] = []

    if n < min_n:
        msg = (
            f"bootstrap_exponential_ci called with only {n} usable TBF "
            f"point(s); Section 2.7's spirit (if not letter, since it "
            f"only specifies Tier A explicitly) suggests >= {min_n}."
        )
        if not allow_unsafe:
            raise InsufficientDataError(msg)
        warning_messages.append("UNSAFE BOOTSTRAP: " + msg)

    if n < 1:
        raise InsufficientDataError("Cannot bootstrap an exponential fit with 0 points.")

    rng = np.random.default_rng(seed)
    lambdas, mtbfs = [], []
    for _ in range(n_boot):
        resample = rng.choice(tbf_days, size=n, replace=True)
        try:
            _, scale = stats.expon.fit(resample, floc=0)
        except Exception:
            continue
        if scale <= 0:
            continue
        lambdas.append(1 / scale)
        mtbfs.append(scale)

    lambdas_arr = np.array(lambdas)
    mtbfs_arr = np.array(mtbfs)
    n_successful = len(lambdas_arr)

    if n_successful == 0:
        return BootstrapResult(
            beta_ci=None, eta_ci=None, lambda_ci=None, mtbf_ci_days=None,
            n_successful=0, n_requested=n_boot,
            warnings=tuple(warning_messages + ["All bootstrap resamples failed to fit."]),
        )

    return BootstrapResult(
        beta_ci=None,
        eta_ci=None,
        lambda_ci=(float(np.percentile(lambdas_arr, 2.5)), float(np.percentile(lambdas_arr, 97.5))),
        mtbf_ci_days=(float(np.percentile(mtbfs_arr, 2.5)), float(np.percentile(mtbfs_arr, 97.5))),
        n_successful=n_successful,
        n_requested=n_boot,
        warnings=tuple(warning_messages),
    )
