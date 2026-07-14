from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from aeolus_rams_phase2.bootstrap import (
    bootstrap_weibull_ci, bootstrap_exponential_ci, BootstrapResult,
)
from aeolus_rams_phase2.distributions import InsufficientDataError


def test_bootstrap_weibull_ci_raises_below_minimum():
    with pytest.raises(InsufficientDataError):
        bootstrap_weibull_ci(np.array([10.0, 20.0, 30.0]), n_boot=50)


def test_bootstrap_weibull_ci_allow_unsafe_flagged():
    result = bootstrap_weibull_ci(np.array([56.9, 217.3]), n_boot=200, allow_unsafe=True)
    assert isinstance(result, BootstrapResult)
    assert not result.is_meaningful
    assert any("UNSAFE" in w for w in result.warnings)


def test_bootstrap_weibull_ci_meaningful_result_has_no_warnings():
    rng = np.random.default_rng(11)
    sample = stats.weibull_min.rvs(2.0, scale=100.0, size=20, random_state=rng)
    result = bootstrap_weibull_ci(sample, n_boot=300, seed=1)
    assert result.is_meaningful
    assert result.warnings == ()
    assert result.n_successful <= result.n_requested
    assert result.beta_ci[0] < result.beta_ci[1]
    assert result.eta_ci[0] < result.eta_ci[1]
    assert result.mtbf_ci_days[0] < result.mtbf_ci_days[1]


def test_bootstrap_weibull_ci_is_reproducible_with_seed():
    sample = stats.weibull_min.rvs(2.0, scale=100.0, size=20, random_state=np.random.default_rng(0))
    r1 = bootstrap_weibull_ci(sample, n_boot=200, seed=99)
    r2 = bootstrap_weibull_ci(sample, n_boot=200, seed=99)
    assert r1.beta_ci == r2.beta_ci
    assert r1.mtbf_ci_days == r2.mtbf_ci_days


def test_bootstrap_exponential_ci_raises_below_minimum():
    with pytest.raises(InsufficientDataError):
        bootstrap_exponential_ci(np.array([10.0, 20.0]), n_boot=50)


def test_bootstrap_exponential_ci_meaningful():
    rng = np.random.default_rng(2)
    sample = stats.expon.rvs(scale=60.0, size=15, random_state=rng)
    result = bootstrap_exponential_ci(sample, n_boot=300, seed=1)
    assert result.is_meaningful
    assert result.lambda_ci[0] < result.lambda_ci[1]
    assert result.mtbf_ci_days[0] < result.mtbf_ci_days[1]


def test_bootstrap_exponential_ci_zero_points_raises():
    with pytest.raises(InsufficientDataError):
        bootstrap_exponential_ci(np.array([]), n_boot=50, allow_unsafe=True)
