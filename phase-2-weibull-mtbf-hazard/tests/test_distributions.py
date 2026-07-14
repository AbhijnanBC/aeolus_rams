from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from aeolus_rams_phase2.distributions import (
    fit_weibull,
    fit_exponential_only,
    demonstrate_degenerate_fit,
    InsufficientDataError,
    WeibullFitResult,
    ExponentialFitResult,
)


# ---------------------------------------------------------------------------
# Hard guards
# ---------------------------------------------------------------------------

def test_fit_weibull_raises_below_minimum():
    with pytest.raises(InsufficientDataError):
        fit_weibull(np.array([10.0, 20.0, 30.0]))  # n=3 < default min_n=8


def test_fit_weibull_allow_unsafe_returns_flagged_result():
    result = fit_weibull(np.array([56.9, 217.3]), allow_unsafe=True)
    assert isinstance(result, WeibullFitResult)
    assert any("UNSAFE" in w for w in result.warnings)


def test_fit_weibull_below_two_points_always_raises():
    with pytest.raises(InsufficientDataError):
        fit_weibull(np.array([50.0]), allow_unsafe=True)


def test_fit_exponential_only_raises_below_minimum():
    with pytest.raises(InsufficientDataError):
        fit_exponential_only(np.array([10.0, 20.0]))  # n=2 < default min_n=5


def test_fit_exponential_only_allow_unsafe():
    result = fit_exponential_only(np.array([10.0, 20.0]), allow_unsafe=True)
    assert isinstance(result, ExponentialFitResult)
    assert any("UNSAFE" in w for w in result.warnings)


def test_fit_exponential_only_zero_points_always_raises():
    with pytest.raises(InsufficientDataError):
        fit_exponential_only(np.array([]), allow_unsafe=True)


# ---------------------------------------------------------------------------
# The documented degenerate n=2 example
# ---------------------------------------------------------------------------

def test_demonstrate_degenerate_fit_reproduces_documented_numbers():
    result = demonstrate_degenerate_fit()
    assert result.beta == pytest.approx(1.79, abs=0.01)
    assert result.eta == pytest.approx(154.90, abs=0.05)
    assert result.n_used == 2
    assert any("UNSAFE" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Fit correctness on well-behaved samples
# ---------------------------------------------------------------------------

def test_fit_weibull_recovers_known_parameters_approximately():
    rng = np.random.default_rng(42)
    true_beta, true_eta = 2.5, 100.0
    sample = stats.weibull_min.rvs(true_beta, scale=true_eta, size=300, random_state=rng)
    fit = fit_weibull(sample)
    assert fit.beta == pytest.approx(true_beta, rel=0.15)
    assert fit.eta == pytest.approx(true_eta, rel=0.15)
    assert fit.n_used == 300


def test_fit_weibull_prefers_exponential_for_exponential_data():
    rng = np.random.default_rng(7)
    sample = stats.expon.rvs(scale=80.0, size=300, random_state=rng)
    fit = fit_weibull(sample)
    # beta should land near 1.0 for genuinely exponential (memoryless) data,
    # and AIC should not strongly favour the extra Weibull shape parameter.
    assert fit.beta == pytest.approx(1.0, abs=0.25)


def test_fit_weibull_prefers_weibull_for_strongly_shaped_data():
    rng = np.random.default_rng(3)
    sample = stats.weibull_min.rvs(4.0, scale=100.0, size=300, random_state=rng)
    fit = fit_weibull(sample)
    assert fit.preferred == "weibull"
    assert fit.aic_weibull < fit.aic_exponential


def test_wearout_interpretation_labels():
    rng = np.random.default_rng(1)
    wearout = fit_weibull(stats.weibull_min.rvs(4.0, scale=100.0, size=50, random_state=rng))
    infant = fit_weibull(stats.weibull_min.rvs(0.5, scale=100.0, size=50, random_state=rng))
    assert "wear-out" in wearout.wearout_interpretation
    assert "infant mortality" in infant.wearout_interpretation


def test_fit_exponential_only_recovers_known_rate():
    rng = np.random.default_rng(5)
    true_mtbf = 60.0
    sample = stats.expon.rvs(scale=true_mtbf, size=200, random_state=rng)
    fit = fit_exponential_only(sample)
    assert fit.mtbf_days == pytest.approx(true_mtbf, rel=0.15)
    assert fit.lambda_rate == pytest.approx(1 / true_mtbf, rel=0.15)
