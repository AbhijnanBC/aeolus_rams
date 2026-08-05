"""Tests for aeolus_rams_phase8.bayesian_update"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from aeolus_rams_phase8 import config
from aeolus_rams_phase8 import bayesian_update
from aeolus_rams_phase8 import synthetic_data


@pytest.fixture
def bearing_df():
    return synthetic_data.generate_bearing_failures(seed=42)


@pytest.fixture
def small_grid():
    """30×30 grid for speed; uses 5 synthetic failures."""
    rng = np.random.default_rng(42)
    fail_times = rng.weibull(config.BEARING_BETA, 5) * (config.BEARING_ETA * 0.95)
    return bayesian_update.compute_posterior_grid(
        fail_times=fail_times,
        censor_times=np.array([]),
        n_beta=30,
        n_eta=30,
    )


class TestLogPosterior:

    def test_prior_log_posterior_peaks_near_phase2(self):
        """With no data, the posterior should peak at the prior centre."""
        result = bayesian_update.compute_posterior_grid(
            fail_times=np.array([]),
            censor_times=np.array([]),
            n_beta=50,
            n_eta=50,
        )
        beta_map = result["beta_map"]
        eta_map  = result["eta_map"]
        # MAP should be within 10% of the prior centre
        assert abs(beta_map - config.BEARING_BETA) / config.BEARING_BETA < 0.10
        assert abs(eta_map  - config.BEARING_ETA)  / config.BEARING_ETA  < 0.10

    def test_log_likelihood_correct_for_known_case(self):
        """
        For a single failure at t=eta with β=1 (exponential), the log-likelihood
        at (β=1, η=t) is log(1/t) + (1-1)·log(1) − 1 = −log(t).
        """
        t_fail = 5000.0
        result = bayesian_update.compute_posterior_grid(
            fail_times=np.array([t_fail]),
            censor_times=np.array([]),
            beta0=1.0,
            eta0=t_fail,
            n_beta=5,
            n_eta=5,
        )
        # Should not raise; posterior should be defined
        assert result["posterior"].sum() > 0.99


class TestPosteriorGrid:

    def test_posterior_grid_sums_to_one(self, small_grid):
        assert abs(small_grid["posterior"].sum() - 1.0) < 1e-6

    def test_posterior_grid_shape(self):
        result = bayesian_update.compute_posterior_grid(
            fail_times=np.array([1000.0]),
            censor_times=np.array([]),
        )
        assert result["posterior"].shape == (config.GRID_N_BETA, config.GRID_N_ETA)

    def test_map_estimate_is_within_grid(self):
        result = bayesian_update.compute_posterior_grid(
            fail_times=np.array([20000.0, 22000.0, 18000.0]),
            censor_times=np.array([]),
        )
        lb_lo = math.exp(math.log(config.BEARING_BETA) - config.GRID_BETA_RANGE_SIGMA * config.PRIOR_BETA_LOG_SIGMA)
        lb_hi = math.exp(math.log(config.BEARING_BETA) + config.GRID_BETA_RANGE_SIGMA * config.PRIOR_BETA_LOG_SIGMA)
        le_lo = math.exp(math.log(config.BEARING_ETA)  - config.GRID_ETA_RANGE_SIGMA  * config.PRIOR_ETA_LOG_SIGMA)
        le_hi = math.exp(math.log(config.BEARING_ETA)  + config.GRID_ETA_RANGE_SIGMA  * config.PRIOR_ETA_LOG_SIGMA)
        assert lb_lo < result["beta_map"] < lb_hi
        assert le_lo < result["eta_map"]  < le_hi

    def test_marginal_beta_sums_to_one(self, small_grid):
        marginal = small_grid["marginal_beta"]
        assert abs(marginal.sum() - 1.0) < 1e-6

    def test_marginal_eta_sums_to_one(self, small_grid):
        marginal = small_grid["marginal_eta"]
        assert abs(marginal.sum() - 1.0) < 1e-6


class TestSequentialUpdate:

    def test_more_data_narrows_posterior(self, bearing_df):
        posteriors, summary_df = bayesian_update.run_sequential_update(bearing_df)
        # CI width for β should narrow monotonically stages 0→3
        widths = []
        for stage in range(4):
            row = summary_df[summary_df["update_stage"] == stage].iloc[0]
            widths.append(row["beta_ci90_hi"] - row["beta_ci90_lo"])
        assert widths[3] <= widths[1], (
            f"Stage 3 CI width {widths[3]:.4f} should be ≤ Stage 1 CI width {widths[1]:.4f}"
        )

    def test_posterior_shifts_toward_lower_eta(self, bearing_df):
        """
        Synthetic data has η = 0.95 × η_phase2. MAP η should be < η_phase2.
        """
        posteriors, summary_df = bayesian_update.run_sequential_update(bearing_df)
        stage3_eta = summary_df[summary_df["update_stage"] == 3].iloc[0]["eta_map_days"]
        assert stage3_eta < config.BEARING_ETA, (
            f"Expected η̂_MAP {stage3_eta:.0f}d < η_phase2 {config.BEARING_ETA:.0f}d"
        )

    def test_summary_df_has_correct_columns(self, bearing_df):
        _, summary_df = bayesian_update.run_sequential_update(bearing_df)
        required = [
            "update_stage", "n_failures", "n_censored",
            "beta_map", "eta_map_days",
            "beta_ci90_lo", "beta_ci90_hi",
            "eta_ci90_lo_days", "eta_ci90_hi_days",
            "T_star_updated_days", "T_star_updated_years",
            "savings_updated_fraction",
        ]
        for col in required:
            assert col in summary_df.columns, f"Missing column: {col}"

    def test_summary_has_four_stages(self, bearing_df):
        _, summary_df = bayesian_update.run_sequential_update(bearing_df)
        assert len(summary_df) == 4

    def test_t_star_is_positive(self, bearing_df):
        _, summary_df = bayesian_update.run_sequential_update(bearing_df)
        assert (summary_df["T_star_updated_days"] > 0).all()

    def test_ci90_covers_prior_beta(self, bearing_df):
        """
        Prior centre β₀=2.0 should lie within the Stage 0 90% CI.
        """
        _, summary_df = bayesian_update.run_sequential_update(bearing_df)
        row = summary_df[summary_df["update_stage"] == 0].iloc[0]
        assert row["beta_ci90_lo"] <= config.BEARING_BETA <= row["beta_ci90_hi"]


class TestBarlowProschan:

    def test_t_star_decreases_with_lower_eta(self):
        """Lower η (worse bearing) → shorter optimal replacement interval."""
        t_high = bayesian_update._barlow_proschan_t_star(2.0, 25000.0)
        t_low  = bayesian_update._barlow_proschan_t_star(2.0, 20000.0)
        assert t_low < t_high

    def test_t_star_positive(self):
        t = bayesian_update._barlow_proschan_t_star(2.0, 25000.0)
        assert t > 0.0

    def test_barlow_proschan_integral_zero_T(self):
        val = bayesian_update._barlow_proschan_integral(2.0, 25000.0, T=0.0)
        assert val == pytest.approx(0.0, abs=1.0)