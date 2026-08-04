"""Tests for ccf.py."""
from __future__ import annotations

import numpy as np
import pytest

from aeolus_rams_phase5.ccf import (
    apply_beta_factor_ccf,
    ccf_sensitivity_sweep,
)
from aeolus_rams_phase5 import config as _cfg


def test_beta_zero_recovers_independent(top_gate_1yr, Q_1yr):
    """With β=0, CCF-adjusted Q_top must equal the independent Q_top."""
    result = apply_beta_factor_ccf(top_gate_1yr, Q_1yr, beta_G1G2=0.0)
    assert abs(result["Q_top_ccf_adjusted"] - result["Q_top_independent"]) < 1e-12, (
        "β=0 must give Q_top_ccf = Q_top_independent"
    )
    assert abs(result["ccf_risk_uplift"] - 1.0) < 1e-10


def test_positive_beta_increases_risk(top_gate_1yr, Q_1yr):
    """Positive β must increase Q_top (CCF always increases risk)."""
    for beta in [0.05, 0.10, 0.15, 0.20]:
        result = apply_beta_factor_ccf(top_gate_1yr, Q_1yr, beta_G1G2=beta)
        assert result["Q_top_ccf_adjusted"] >= result["Q_top_independent"] - 1e-12, (
            f"β={beta}: CCF-adjusted Q_top < independent Q_top — CCF must increase risk"
        )
        assert result["ccf_risk_uplift"] >= 1.0 - 1e-10


def test_uplift_monotone_in_beta(top_gate_1yr, Q_1yr):
    """Higher β → higher uplift (monotone relationship)."""
    betas = [0.0, 0.05, 0.10, 0.15, 0.20]
    uplifts = [
        apply_beta_factor_ccf(top_gate_1yr, Q_1yr, b)["ccf_risk_uplift"]
        for b in betas
    ]
    for i in range(len(uplifts) - 1):
        assert uplifts[i] <= uplifts[i + 1] + 1e-9, (
            f"Uplift not monotone: uplift[{i}]={uplifts[i]:.4f} > uplift[{i+1}]={uplifts[i+1]:.4f}"
        )


def test_sweep_output_shape(top_gate_1yr, Q_1yr):
    """CCF sweep must have correct shape and required columns."""
    df = ccf_sensitivity_sweep(top_gate_1yr, Q_1yr, beta_range=np.linspace(0, 0.20, 5))
    assert len(df) == 5
    for col in ["beta_G1G2", "Q_top_independent", "Q_top_ccf_adjusted",
                "ccf_risk_uplift", "return_period_ccf"]:
        assert col in df.columns


def test_return_period_decreases_with_beta(top_gate_1yr, Q_1yr):
    """Higher β → shorter return period."""
    df = ccf_sensitivity_sweep(top_gate_1yr, Q_1yr, beta_range=np.array([0.0, 0.10, 0.20]))
    rps = df["return_period_ccf"].values
    assert rps[0] >= rps[1] >= rps[2] - 1e-6, (
        f"Return period should decrease with β: {rps}"
    )


def test_central_uplift_in_expected_range(top_gate_1yr, Q_1yr):
    """β=0.10 uplift should be in [1.1, 3.0] (literature range for this topology)."""
    result = apply_beta_factor_ccf(top_gate_1yr, Q_1yr, beta_G1G2=0.10)
    uplift = result["ccf_risk_uplift"]
    assert 1.1 <= uplift <= 3.0, (
        f"Uplift at β=0.10 = {uplift:.3f} outside expected range [1.1, 3.0]"
    )