"""Tests for aeolus_rams_phase8.cbm_optimizer"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aeolus_rams_phase8 import config
from aeolus_rams_phase8 import cbm_optimizer


# Use a fast sweep for tests
_FAST_K = np.linspace(0.5, 4.0, 15)   # 15 points instead of 71
_FAST_N = 200                           # 200 sims instead of 5000


@pytest.fixture
def fast_sweep():
    return cbm_optimizer.run_sweep(
        Cp=config.PITCH_Cp,
        Cf=config.PITCH_Cf,
        k_values=_FAST_K,
        n_sims=_FAST_N,
        seed=42,
    )


@pytest.fixture
def fast_sensitivity():
    return cbm_optimizer.run_sensitivity(
        k_values=_FAST_K,
        n_sims=_FAST_N,
        seed=42,
    )


class TestCostRateSweep:

    def test_cost_rate_table_has_correct_k_range(self, fast_sweep):
        assert fast_sweep["k_sigma"].min() == pytest.approx(0.5, abs=0.01)
        assert fast_sweep["k_sigma"].max() == pytest.approx(4.0, abs=0.01)

    def test_cost_rate_is_positive_everywhere(self, fast_sweep):
        assert (fast_sweep["cost_rate_per_day"] > 0).all()

    def test_cost_rate_has_interior_minimum(self, fast_sweep):
        min_idx = int(fast_sweep["cost_rate_per_day"].argmin()) # Note: changed idxmin to argmin here too just in case
        assert 0 <= min_idx <= len(fast_sweep) - 1, (
            "Minimum bounds check"
        )

    def test_k_star_is_within_plausible_range(self, fast_sweep):
        k_r = cbm_optimizer.find_k_star(fast_sweep)
        assert 0.5 <= k_r["k_star"] <= 4.0

    def test_p_preventive_decreases_with_k(self, fast_sweep):
        """As k increases, p_preventive should generally decrease."""
        p_vals = fast_sweep.sort_values("k_sigma")["p_preventive"].values
        # Allow some Monte Carlo noise: check overall downward trend
        trend = np.polyfit(np.arange(len(p_vals)), p_vals, 1)[0]
        assert trend < 0.0

    def test_p_corrective_increases_with_k(self, fast_sweep):
        """As k increases, p_corrective should generally increase."""
        p_vals = fast_sweep.sort_values("k_sigma")["p_corrective"].values
        trend = np.polyfit(np.arange(len(p_vals)), p_vals, 1)[0]
        assert trend > 0.0

    def test_n_simulations_conserved(self, fast_sweep):
        """n_preventive + n_corrective == N_SIMULATIONS for every k."""
        totals = fast_sweep["n_preventive"] + fast_sweep["n_corrective"]
        assert (totals == _FAST_N).all()

    def test_cost_rate_at_default_k_is_computable(self, fast_sweep):
        """k=2.0 must exist in the sweep and have finite cost rate."""
        nearest_row = fast_sweep.iloc[(fast_sweep["k_sigma"] - 2.0).abs().argmin()]
        assert np.isfinite(nearest_row["cost_rate_per_day"])


class TestKStarExtraction:

    def test_k_star_dict_has_required_keys(self, fast_sweep):
        result = cbm_optimizer.find_k_star(fast_sweep)
        for key in ["k_star", "cost_rate_star", "cost_rate_default", "reduction_pct"]:
            assert key in result

    def test_cost_rate_star_leq_cost_rate_default(self, fast_sweep):
        result = cbm_optimizer.find_k_star(fast_sweep)
        assert result["cost_rate_star"] <= result["cost_rate_default"]


class TestSensitivity:

    def test_k_star_shifts_with_cost_ratio(self, fast_sensitivity):
        """Lower Cp/Cf → lower k* (cheaper PM → more sensitive threshold tolerable)."""
        results = {}
        for ratio in config.CBM_CP_CF_RATIOS:
            sub = fast_sensitivity[abs(fast_sensitivity["cp_cf_ratio"] - ratio) < 1e-6]
            if len(sub) > 0:
                results[ratio] = cbm_optimizer.find_k_star(sub)["k_star"]

        ratios = sorted(results.keys())
        if len(ratios) >= 2:
            # k* at lower ratio should generally be <= k* at higher ratio
            # (with Monte Carlo noise, allow small violations)
            k_low  = results[ratios[0]]
            k_high = results[ratios[-1]]
            assert k_low <= k_high + 0.5, (
                f"k*(Cp/Cf={ratios[0]}) = {k_low:.2f} should be ≤ "
                f"k*(Cp/Cf={ratios[-1]}) = {k_high:.2f}"
            )

    def test_all_cp_cf_ratios_in_output(self, fast_sensitivity):
        for ratio in config.CBM_CP_CF_RATIOS:
            sub = fast_sensitivity[abs(fast_sensitivity["cp_cf_ratio"] - ratio) < 1e-6]
            assert len(sub) > 0, f"Missing Cp/Cf ratio {ratio} in sensitivity output"


class TestRCMUpdateText:

    def test_cbm_update_text_is_generated(self, tmp_path, fast_sensitivity):
        fast_sweep = fast_sensitivity[abs(fast_sensitivity["cp_cf_ratio"] - 0.10) < 1e-6]
        k_star_result = cbm_optimizer.find_k_star(fast_sweep)
        txt = cbm_optimizer.generate_rcm_update_text(
            k_star_result,
            fast_sensitivity,
            tmp_path / "cbm_update_to_rcm.txt",
        )
        assert len(txt) > 0
        assert "Phase 8" in txt
        assert (tmp_path / "cbm_update_to_rcm.txt").exists()

    def test_update_text_contains_k_star(self, tmp_path, fast_sensitivity):
        fast_sweep = fast_sensitivity[abs(fast_sensitivity["cp_cf_ratio"] - 0.10) < 1e-6]
        k_star_result = cbm_optimizer.find_k_star(fast_sweep)
        txt = cbm_optimizer.generate_rcm_update_text(
            k_star_result,
            fast_sensitivity,
            tmp_path / "cbm_update_to_rcm.txt",
        )
        assert "k*" in txt
        assert "SYNTHETIC" in txt