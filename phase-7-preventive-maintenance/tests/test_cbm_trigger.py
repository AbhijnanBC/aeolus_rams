"""Tests for cbm_trigger.py."""
from __future__ import annotations

import numpy as np
import pytest

from aeolus_rams_phase7.cbm_trigger import cbm_trigger_rule, trigger_threshold_sensitivity


def test_trigger_output_shape():
    """Output DataFrame must have same length as input."""
    scores = np.random.default_rng(42).normal(0.1, 0.03, 200)
    timestamps = np.arange(200)
    df = cbm_trigger_rule(scores, timestamps)
    assert len(df) == 200
    for col in ["timestamp", "score", "rolling_mean", "rolling_std",
                "threshold", "maintenance_triggered"]:
        assert col in df.columns


def test_high_score_triggers_maintenance():
    """A sustained score spike well above baseline must trigger maintenance."""
    rng = np.random.default_rng(0)
    scores = rng.normal(0.1, 0.01, size=200)
    scores[180:] = 0.8   # large sustained spike
    timestamps = np.arange(200)
    df = cbm_trigger_rule(scores, timestamps, k_sigma=2.0)
    # Must trigger at least once in the spike window
    assert df.loc[180:, "maintenance_triggered"].any()


def test_no_false_triggers_on_flat_signal():
    """A completely flat signal should produce no triggers after warm-up."""
    scores = np.full(200, 0.1)
    timestamps = np.arange(200)
    df = cbm_trigger_rule(scores, timestamps, k_sigma=2.0)
    # After rolling window has warmed up (std=0), threshold = mean (no variance)
    # All flat signals: score == rolling_mean → no trigger (not >=)
    assert not df.loc[40:, "maintenance_triggered"].any()


def test_higher_k_sigma_fewer_triggers():
    """Higher k_sigma threshold must produce ≤ triggers than lower k_sigma."""
    rng = np.random.default_rng(42)
    scores = rng.normal(0.1, 0.05, 500)
    timestamps = np.arange(500)
    n_k2 = cbm_trigger_rule(scores, timestamps, k_sigma=2.0)["maintenance_triggered"].sum()
    n_k3 = cbm_trigger_rule(scores, timestamps, k_sigma=3.0)["maintenance_triggered"].sum()
    assert n_k3 <= n_k2, f"k=3 gave more triggers ({n_k3}) than k=2 ({n_k2})"


def test_sensitivity_output_shape():
    """trigger_threshold_sensitivity should return one row per k value."""
    rng = np.random.default_rng(0)
    scores = rng.normal(0.1, 0.05, 200)
    timestamps = np.arange(200)
    k_vals = np.array([1.5, 2.0, 2.5, 3.0])
    df = trigger_threshold_sensitivity(scores, timestamps, k_values=k_vals)
    assert len(df) == 4
    assert "n_triggers" in df.columns