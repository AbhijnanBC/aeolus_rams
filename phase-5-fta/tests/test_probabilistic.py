"""Tests for probabilistic.py — Q_top uncertainty propagation."""
from __future__ import annotations

import numpy as np
import pytest

from aeolus_rams_phase5.probabilistic import (
    Q_uncertainty_from_mtbf_ci,
    probabilistic_fta,
)
from aeolus_rams_phase5 import config as _cfg


def test_Q_samples_in_range():
    """All Q samples from log-normal MTBF must be in [0, 1]."""
    Q_samples = Q_uncertainty_from_mtbf_ci(
        mtbf_point=1936.0, mtbf_ci_low=1017.0, mtbf_ci_high=2659.0,
        t=365.25, n_samples=5000,
    )
    assert Q_samples.shape == (5000,)
    assert (Q_samples >= 0.0).all(), "Q must be ≥ 0"
    assert (Q_samples <= 1.0).all(), "Q must be ≤ 1"


def test_Q_sample_mean_near_point_estimate():
    """Mean of Q samples should be close to Q from the point estimate MTBF.

    Not exact because log-normal is asymmetric, but within 20%.
    """
    mtbf = 1936.0
    t = 365.25
    Q_point = 1 - np.exp(-t / mtbf)
    Q_samples = Q_uncertainty_from_mtbf_ci(
        mtbf, 1017.0, 2659.0, t=t, n_samples=20_000
    )
    Q_mean = Q_samples.mean()
    assert abs(Q_mean - Q_point) / Q_point < 0.20, (
        f"Mean Q sample {Q_mean:.4f} deviates >20% from point estimate {Q_point:.4f}"
    )


def test_wider_CI_gives_wider_Q_distribution():
    """A wider MTBF CI should produce a wider Q distribution (larger std)."""
    rng = np.random.default_rng(0)
    Q_narrow = Q_uncertainty_from_mtbf_ci(
        1936.0, 1500.0, 2400.0, n_samples=5000, rng=np.random.default_rng(0)
    )
    Q_wide = Q_uncertainty_from_mtbf_ci(
        1936.0, 500.0, 5000.0, n_samples=5000, rng=np.random.default_rng(0)
    )
    assert Q_wide.std() > Q_narrow.std(), (
        "Wider CI must give wider Q distribution"
    )


def test_probabilistic_fta_small():
    """Smoke test: probabilistic_fta with 1,000 samples must complete and return sane results."""
    Q_top_samples, summary = probabilistic_fta(t=365.25, n_samples=1000, seed=0)
    assert Q_top_samples.shape == (1000,)
    assert (Q_top_samples > 0.0).all()
    assert (Q_top_samples <= 1.0).all()
    # Mean Q_top should be in a reasonable range (order 10⁻⁴ to 10⁻³)
    mean_Q = Q_top_samples.mean()
    assert 1e-5 < mean_Q < 1e-2, f"Mean Q_top = {mean_Q:.3e} outside expected range"
    assert "p5_Q_top" in summary.columns
    assert "mean_return_period_yr" in summary.columns


def test_Q_top_CI_straddles_point_estimate():
    """The 90% CI on Q_top must straddle the point-estimate Q_top."""
    from aeolus_rams_phase5.fault_tree import build_overspeed_fault_tree
    Q_point = build_overspeed_fault_tree(_cfg.FALLBACK_Q_1YR).Q

    Q_top_samples, summary = probabilistic_fta(t=365.25, n_samples=5000, seed=99)
    p5  = float(summary["p5_Q_top"].iloc[0])
    p95 = float(summary["p95_Q_top"].iloc[0])

    assert p5 <= Q_point <= p95, (
        f"Point estimate Q_top={Q_point:.4e} not in 90% CI [{p5:.4e}, {p95:.4e}]. "
        "Uncertainty model may be miscalibrated."
    )