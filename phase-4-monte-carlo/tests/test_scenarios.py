"""
Tests for scenarios.py.

Key checks:
  1. All three scenarios build valid parameter lists.
  2. Optimised has lower effective MTTR than baseline.
  3. Degraded has higher effective MTTR than baseline.
  4. MTBF values are unchanged across scenarios (scenarios only touch MTTR).
  5. expected_turbine_availability is monotone: optimised > baseline > degraded.
"""
from __future__ import annotations

import pytest

from aeolus_rams_phase4.scenarios import (
    build_scenario_params,
    expected_turbine_availability,
    expected_bop_availability,
)
from aeolus_rams_phase4 import config


def test_all_scenarios_build():
    """All three scenarios must build without errors."""
    for key in ["baseline", "optimised", "degraded"]:
        t, b = build_scenario_params(key)
        assert len(t) == len(config.TURBINE_COMPONENTS)
        assert len(b) == len(config.BOP_COMPONENTS)


def test_unknown_scenario_raises():
    """Invalid scenario key must raise ValueError."""
    with pytest.raises(ValueError, match="Unknown scenario"):
        build_scenario_params("nonexistent")


def test_mtbf_unchanged_across_scenarios():
    """MTBF must be the same across all scenarios (only MTTR changes)."""
    t_bl, _ = build_scenario_params("baseline")
    t_opt, _ = build_scenario_params("optimised")
    t_deg, _ = build_scenario_params("degraded")

    for bl, opt, deg in zip(t_bl, t_opt, t_deg):
        assert bl.mtbf_days == opt.mtbf_days == deg.mtbf_days, (
            f"MTBF for {bl.name} changed across scenarios — must not"
        )


def test_mttr_ordered_across_scenarios():
    """Optimised MTTR < Baseline MTTR < Degraded MTTR."""
    t_bl, _ = build_scenario_params("baseline")
    t_opt, _ = build_scenario_params("optimised")
    t_deg, _ = build_scenario_params("degraded")

    for bl, opt, deg in zip(t_bl, t_opt, t_deg):
        assert opt.mttr_raw_days < bl.mttr_raw_days < deg.mttr_raw_days, (
            f"{bl.name}: MTTR order violation — optimised={opt.mttr_raw_days:.2f}, "
            f"baseline={bl.mttr_raw_days:.2f}, degraded={deg.mttr_raw_days:.2f}"
        )


def test_expected_availability_ordered():
    """A_turbine (analytical): optimised > baseline > degraded."""
    a_bl = expected_turbine_availability("baseline")
    a_opt = expected_turbine_availability("optimised")
    a_deg = expected_turbine_availability("degraded")
    assert a_opt > a_bl > a_deg, (
        f"Availability ordering violated: "
        f"optimised={a_opt:.4f}, baseline={a_bl:.4f}, degraded={a_deg:.4f}"
    )


def test_expected_bop_availability_in_range():
    """BoP analytical availability must be in (0.5, 1.0) for all scenarios."""
    for key in ["baseline", "optimised", "degraded"]:
        a_bop = expected_bop_availability(key)
        assert 0.5 < a_bop < 1.0, (
            f"Scenario {key}: A_bop={a_bop:.4f} out of expected range (0.5, 1.0)"
        )