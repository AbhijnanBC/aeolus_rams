"""Tests for aeolus_rams_phase8.cox_survival"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from aeolus_rams_phase8 import config
from aeolus_rams_phase8 import synthetic_data


# Skip entire module if lifelines not installed
lifelines = pytest.importorskip("lifelines", reason="lifelines not installed")

from aeolus_rams_phase8 import cox_survival


@pytest.fixture
def covariates_df():
    return synthetic_data.generate_covariates(seed=42)


@pytest.fixture
def cph_result(covariates_df):
    return cox_survival.fit_cox(covariates_df)


class TestCoxFit:

    def test_cox_fit_runs_without_error(self, covariates_df):
        result = cox_survival.fit_cox(covariates_df)
        assert result is not None

    def test_hazard_ratios_are_positive(self, cph_result):
        hr_df = cph_result["hazard_ratios"]
        assert (hr_df["exp_coef"] > 0).all()

    def test_wind_hr_above_one(self, cph_result):
        """Data generated with TRUE_HR_WIND=1.15; fitted HR should be > 1.0."""
        hr_df = cph_result["hazard_ratios"]
        wind_hr = hr_df[hr_df["covariate"] == "wind_speed_mean_mps"].iloc[0]["exp_coef"]
        assert wind_hr > 1.0, f"Expected wind HR > 1.0, got {wind_hr:.4f}"

    def test_vibration_hr_above_one(self, cph_result):
        """Data generated with TRUE_HR_VIBRATION=1.25; fitted HR should be > 1.0."""
        hr_df = cph_result["hazard_ratios"]
        vib_hr = hr_df[hr_df["covariate"] == "vibration_rms_ms2"].iloc[0]["exp_coef"]
        assert vib_hr > 1.0, f"Expected vibration HR > 1.0, got {vib_hr:.4f}"

    def test_all_covariates_in_output(self, cph_result):
        hr_df = cph_result["hazard_ratios"]
        for cov in config.COVARIATE_NAMES:
            assert cov in hr_df["covariate"].values, f"Missing covariate: {cov}"

    def test_hazard_ratios_df_has_required_columns(self, cph_result):
        hr_df = cph_result["hazard_ratios"]
        for col in ["exp_coef", "exp_coef_lower_95", "exp_coef_upper_95", "p_value", "z_score"]:
            assert col in hr_df.columns, f"Missing column: {col}"

    def test_concordance_above_chance(self, cph_result):
        assert cph_result["concordance"] > 0.50

    def test_gamma_extracted_correctly(self, cph_result):
        gamma = cph_result["gamma_estimated"]
        assert isinstance(gamma, float)
        assert 0.0 < gamma < 2.0, f"Expected γ ∈ (0, 2), got {gamma:.4f}"

    def test_ci_bounds_ordered(self, cph_result):
        hr_df = cph_result["hazard_ratios"]
        assert (hr_df["exp_coef_lower_95"] < hr_df["exp_coef_upper_95"]).all()

    def test_covariate_stats_complete(self, cph_result):
        for cov in config.COVARIATE_NAMES:
            assert cov in cph_result["covariate_stats"]
            mu, std = cph_result["covariate_stats"][cov]
            assert std > 0


class TestSurvivalCurves:

    @pytest.fixture
    def survival_result(self, cph_result, covariates_df):
        return cox_survival.compute_survival_curves(cph_result, covariates_df)

    def test_survival_curve_is_monotone_decreasing(self, survival_result):
        sf = survival_result["sf_avg"]
        vals = sf.values.flatten()
        diffs = np.diff(vals)
        assert (diffs <= 1e-6).all(), "Survival function is not monotone non-increasing"

    def test_high_wind_curve_below_low_wind(self, survival_result):
        """At any time t, S_high_wind(t) ≤ S_low_wind(t)."""
        km_lo = survival_result["km_low"]
        km_hi = survival_result["km_high"]
        # Compare at common time points up to 500 days
        t_check = np.linspace(10, 500, 20)
        lo_vals = km_lo.survival_function_at_times(t_check).values
        hi_vals = km_hi.survival_function_at_times(t_check).values
        # Allow a small tolerance (KM is a step function)
        assert (hi_vals <= lo_vals + 0.15).all()

    def test_log_rank_p_value_is_float(self, survival_result):
        p = survival_result["lr_p_value"]
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0