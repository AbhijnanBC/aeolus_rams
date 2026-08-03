"""
Tests for component_sampler.py — Layer 1.

Key checks:
  1. mttr_effective_days formula is correctly applied.
  2. failure_repair_sequence yields non-negative TBF and TTR.
  3. Over many draws, sample mean converges to theoretical mean.
  4. Two sequences from the same seed are identical (reproducibility).
"""
from __future__ import annotations

import numpy as np
import pytest

from aeolus_rams_phase4.component_sampler import ComponentParams, failure_repair_sequence


def test_effective_mttr_formula():
    """eff_MTTR = raw_MTTR / access_fraction."""
    p = ComponentParams(
        name="Test", mtbf_days=1000.0, mttr_raw_days=5.0,
        access_fraction=0.60, confidence="test",
    )
    assert abs(p.mttr_effective_days - 5.0 / 0.60) < 1e-9


def test_lambda_failure():
    """λ = 1/MTBF."""
    p = ComponentParams(
        name="Test", mtbf_days=2000.0, mttr_raw_days=1.0,
        access_fraction=1.0, confidence="test",
    )
    assert abs(p.lambda_failure - 1.0 / 2000.0) < 1e-12


def test_sequence_yields_positive_values(rng):
    """TBF and TTR must be strictly positive (Exponential ≥ 0, =0 measure-zero)."""
    p = ComponentParams(
        name="Test", mtbf_days=1000.0, mttr_raw_days=10.0,
        access_fraction=0.5, confidence="test",
    )
    seq = failure_repair_sequence(p, rng)
    for _ in range(500):
        tbf, ttr = next(seq)
        assert tbf > 0.0, "TBF must be positive"
        assert ttr > 0.0, "TTR must be positive"


def test_tbf_mean_converges_to_mtbf(rng):
    """Sample mean TBF must converge to MTBF at n=10,000 (within 3%)."""
    mtbf = 1936.377
    p = ComponentParams(
        name="Test", mtbf_days=mtbf, mttr_raw_days=1.0,
        access_fraction=1.0, confidence="test",
    )
    seq = failure_repair_sequence(p, rng)
    n = 10_000
    tbfs = [next(seq)[0] for _ in range(n)]
    sample_mean = np.mean(tbfs)
    assert abs(sample_mean - mtbf) / mtbf < 0.03, (
        f"TBF sample mean {sample_mean:.1f} deviates >3% from MTBF {mtbf}"
    )


def test_ttr_mean_converges_to_effective_mttr(rng):
    """Sample mean TTR must converge to eff_MTTR at n=10,000 (within 3%)."""
    p = ComponentParams(
        name="Test", mtbf_days=1000.0, mttr_raw_days=6.0,
        access_fraction=0.60, confidence="test",
    )
    expected_eff = 6.0 / 0.60  # 10.0 days
    seq = failure_repair_sequence(p, rng)
    n = 10_000
    ttrs = [next(seq)[1] for _ in range(n)]
    sample_mean = np.mean(ttrs)
    assert abs(sample_mean - expected_eff) / expected_eff < 0.03, (
        f"TTR sample mean {sample_mean:.2f} deviates >3% from eff_MTTR {expected_eff}"
    )


def test_reproducibility():
    """Same seed → identical first 100 draws."""
    p = ComponentParams(
        name="Test", mtbf_days=500.0, mttr_raw_days=5.0,
        access_fraction=0.5, confidence="test",
    )
    rng_a = np.random.default_rng(99)
    rng_b = np.random.default_rng(99)
    seq_a = failure_repair_sequence(p, rng_a)
    seq_b = failure_repair_sequence(p, rng_b)
    for _ in range(100):
        a = next(seq_a)
        b = next(seq_b)
        assert a == b, "Same seed must give identical sequences"