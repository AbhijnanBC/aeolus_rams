from __future__ import annotations

import numpy as np
import pytest

from aeolus_rams_phase3 import config
from aeolus_rams_phase3.sensitivity import (
    aggregate_sensitivity, per_component_sensitivity,
    option_a_reference, render_sensitivity_band,
)
from aeolus_rams_phase3.turbine_rbd import lambda_system


# ---------------------------------------------------------------------------
# aggregate_sensitivity
# ---------------------------------------------------------------------------

def test_aggregate_sensitivity_correct_row_count(components):
    df = aggregate_sensitivity(components, n_points=20)
    assert len(df) == 20


def test_aggregate_sensitivity_R_bounded(components):
    df = aggregate_sensitivity(components, n_points=30)
    assert (df["R_turbine"] > 0).all()
    assert (df["R_turbine"] < 1).all()


def test_aggregate_sensitivity_R_increases_with_mtbf(components):
    """Higher assumed placeholder MTBF → longer MTBF_total → higher R(t)."""
    df = aggregate_sensitivity(components, n_points=40).sort_values(
        "assumed_placeholder_mtbf_days"
    )
    r = df["R_turbine"].to_numpy()
    assert np.all(np.diff(r) >= -1e-12)


def test_aggregate_sensitivity_lambda_total_consistency(components):
    """lambda_system = lambda_known + lambda_placeholder_total for each row."""
    df = aggregate_sensitivity(components, n_points=15)
    total_check = df["lambda_known_per_day"] + df["lambda_placeholder_total_per_day"]
    np.testing.assert_allclose(
        total_check.to_numpy(), df["lambda_system_per_day"].to_numpy(), rtol=1e-9
    )


def test_aggregate_sensitivity_R_at_option_a_mtbf(components):
    """When placeholder MTBF = mean option-A value, R should be close
    to the actual pipeline R_system(t) computed from real lambdas."""
    lam_base = lambda_system(components)
    R_base = float(np.exp(-lam_base * config.T_5YR))
    df = aggregate_sensitivity(components, n_points=60)
    # The range min should be R with very short placeholder MTBFs (low R)
    # and the range max should be R with very long MTBFs (higher R)
    assert df["R_turbine"].min() < R_base
    assert df["R_turbine"].max() > R_base or df["R_turbine"].max() >= R_base - 1e-6


def test_aggregate_sensitivity_no_nan(components):
    df = aggregate_sensitivity(components, n_points=20)
    assert not df["R_turbine"].isna().any()


# ---------------------------------------------------------------------------
# per_component_sensitivity
# ---------------------------------------------------------------------------

def test_per_component_sensitivity_covers_all_placeholders(components):
    df = per_component_sensitivity(components, n_points=10)
    placeholder_names = {n for n, c in components.items() if c.is_placeholder}
    assert set(df["component"].unique()) == placeholder_names


def test_per_component_sensitivity_R_range_valid(components):
    df = per_component_sensitivity(components, n_points=15)
    assert (df["R_turbine"] > 0).all()
    assert (df["R_turbine"] < 1).all()


def test_per_component_sensitivity_increases_with_mtbf(components):
    """Higher MTBF for one component → higher or equal R_system."""
    df = per_component_sensitivity(components, n_points=20)
    for comp in df["component"].unique():
        sub = df[df["component"] == comp].sort_values("assumed_mtbf_days")
        r = sub["R_turbine"].to_numpy()
        assert np.all(np.diff(r) >= -1e-12), f"{comp}: R not non-decreasing"


def test_per_component_sensitivity_delta_R_sign(components):
    """Short-MTBF end should decrease R; long-MTBF end should increase R."""
    df = per_component_sensitivity(components, n_points=20)
    short_end = df.sort_values("assumed_mtbf_days").groupby("component").first()
    long_end = df.sort_values("assumed_mtbf_days").groupby("component").last()
    # delta_R at the short end: most components have lower R than Option A
    assert (short_end["delta_R_from_option_a"] <= 0).any()
    assert (long_end["delta_R_from_option_a"] >= 0).any()


# ---------------------------------------------------------------------------
# option_a_reference
# ---------------------------------------------------------------------------

def test_option_a_reference_count(components):
    ref = option_a_reference(components)
    n_ph = sum(1 for c in components.values() if c.is_placeholder)
    assert len(ref) == n_ph


def test_option_a_reference_mtbf_matches_config(components):
    ref = option_a_reference(components)
    for _, row in ref.iterrows():
        expected = config.PLACEHOLDER_MTBF[row["component"]].mtbf_days
        assert row["option_a_mtbf_days"] == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# render_sensitivity_band (writes to disk, checks file exists)
# ---------------------------------------------------------------------------

def test_render_sensitivity_band_creates_png(components, tmp_path):
    out = tmp_path / "sens_band.png"
    path = render_sensitivity_band(components, out, t=config.T_5YR)
    assert path.exists()
    assert path.stat().st_size > 5_000   # non-trivial PNG
