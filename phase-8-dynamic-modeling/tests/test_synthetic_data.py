"""Tests for aeolus_rams_phase8.synthetic_data"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aeolus_rams_phase8 import config
from aeolus_rams_phase8 import synthetic_data


class TestBearingFailures:

    def test_bearing_failures_count(self):
        df = synthetic_data.generate_bearing_failures(seed=42)
        expected_total = config.N_NEW_BEARING_FAILURES + config.N_CENSORED_BEARING
        assert len(df) == expected_total

    def test_bearing_failures_all_positive(self):
        df = synthetic_data.generate_bearing_failures(seed=42)
        assert (df["event_time_days"] > 0).all()

    def test_bearing_failures_batch_structure(self):
        df = synthetic_data.generate_bearing_failures(seed=42)
        fail_df = df[df["event_observed"] == 1]
        batch_vals = set(fail_df["batch"].unique())
        assert batch_vals == {1, 2, 3}

    def test_censored_observations_have_event_zero(self):
        df = synthetic_data.generate_bearing_failures(seed=42)
        cens = df[df["batch"] == 0]
        assert (cens["event_observed"] == 0).all()
        assert len(cens) == config.N_CENSORED_BEARING

    def test_batch_sizes_equal(self):
        df = synthetic_data.generate_bearing_failures(seed=42)
        fail_df = df[df["event_observed"] == 1]
        for batch in [1, 2, 3]:
            n = (fail_df["batch"] == batch).sum()
            assert n == config.N_NEW_BEARING_FAILURES // 3

    def test_synthetic_flag_is_true(self):
        df = synthetic_data.generate_bearing_failures(seed=42)
        assert df["synthetic"].all()

    def test_reproducible_with_same_seed(self):
        df1 = synthetic_data.generate_bearing_failures(seed=42)
        df2 = synthetic_data.generate_bearing_failures(seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = synthetic_data.generate_bearing_failures(seed=42)
        df2 = synthetic_data.generate_bearing_failures(seed=99)
        assert not df1["event_time_days"].equals(df2["event_time_days"])


class TestCovariates:

    def test_covariates_df_has_required_columns(self):
        df = synthetic_data.generate_covariates(seed=42)
        for col in ["duration_days", "event_observed",
                    "wind_speed_mean_mps", "vibration_rms_ms2", "ambient_temp_C"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_covariates_wind_range(self):
        df = synthetic_data.generate_covariates(seed=42)
        assert df["wind_speed_mean_mps"].min() >= config.WIND_CLIP_LO
        assert df["wind_speed_mean_mps"].max() <= config.WIND_CLIP_HI

    def test_covariates_event_rate(self):
        df = synthetic_data.generate_covariates(seed=42)
        event_rate = df["event_observed"].mean()
        assert 0.50 < event_rate < 0.90

    def test_vibration_correlated_with_wind(self):
        df = synthetic_data.generate_covariates(seed=42)
        r = df["wind_speed_mean_mps"].corr(df["vibration_rms_ms2"])
        assert r > 0.30, f"Expected r > 0.30, got {r:.3f}"

    def test_covariates_row_count(self):
        df = synthetic_data.generate_covariates(seed=42)
        assert len(df) == config.N_SYNTHETIC_PERIODS

    def test_duration_positive(self):
        df = synthetic_data.generate_covariates(seed=42)
        assert (df["duration_days"] > 0).all()

    def test_covariates_reproducible(self):
        df1 = synthetic_data.generate_covariates(seed=42)
        df2 = synthetic_data.generate_covariates(seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_synthetic_flag_is_true(self):
        df = synthetic_data.generate_covariates(seed=42)
        assert df["synthetic"].all()


class TestAnomalyTrajectory:

    def test_anomaly_score_has_degradation_signal(self):
        """Mean S in last 10% of trajectory > mean in first 30%."""
        t_arr, s_arr = synthetic_data.generate_anomaly_trajectory(t_failure=500.0, seed=42)
        n = len(s_arr)
        mean_early = s_arr[:int(0.30 * n)].mean()
        mean_late  = s_arr[int(0.90 * n):].mean()
        assert mean_late > mean_early, (
            f"Expected late mean {mean_late:.3f} > early mean {mean_early:.3f}"
        )

    def test_anomaly_trajectory_length(self):
        import math
        t_f = 200.0
        t_arr, s_arr = synthetic_data.generate_anomaly_trajectory(t_failure=t_f, seed=42)
        assert len(t_arr) == math.ceil(t_f) + 1
        assert len(s_arr) == len(t_arr)


class TestGenerateAll:

    def test_generate_all_returns_expected_keys(self, tmp_path):
        result = synthetic_data.generate_all(seed=42, output_dir=tmp_path)
        assert "bearing_failures" in result
        assert "covariates" in result

    def test_generate_all_writes_csvs(self, tmp_path):
        synthetic_data.generate_all(seed=42, output_dir=tmp_path)
        assert (tmp_path / "synthetic_failures.csv").exists()
        assert (tmp_path / "synthetic_covariates.csv").exists()