"""
aeolus_rams_phase4.scenarios
================================
Builds the three scenario parameter lists from the baseline config
by applying the scenario MTTR multiplier defined in config.SCENARIOS.

Usage in pipeline.py:
    from .scenarios import build_scenario_params

    turbine_params, bop_params = build_scenario_params("optimised")
    mc_df = run_monte_carlo(turbine_params, bop_params, ...)
"""

from __future__ import annotations

import dataclasses

from . import config
from .component_sampler import ComponentParams


def build_scenario_params(
    scenario_key: str,
) -> tuple[list[ComponentParams], list[ComponentParams]]:
    """Build turbine and BoP parameter lists for a named scenario.

    Parameters
    ----------
    scenario_key : str
        One of 'baseline', 'optimised', 'degraded'
        (must be a key in config.SCENARIOS).

    Returns
    -------
    (turbine_params, bop_params) : tuple of lists
        Component parameter lists with mttr_raw_days scaled by the
        scenario's MTTR multiplier. mtbf_days and access_fraction
        are unchanged (those are component physics, not logistics).
    """
    if scenario_key not in config.SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario_key}'. "
            f"Valid keys: {list(config.SCENARIOS.keys())}"
        )

    mult = config.SCENARIOS[scenario_key]["mttr_multiplier"]

    turbine_params = [
        dataclasses.replace(p, mttr_raw_days=p.mttr_raw_days * mult)
        for p in config.TURBINE_COMPONENTS
    ]
    bop_params = [
        dataclasses.replace(p, mttr_raw_days=p.mttr_raw_days * mult)
        for p in config.BOP_COMPONENTS
    ]

    return turbine_params, bop_params


def expected_turbine_availability(scenario_key: str) -> float:
    """Pre-simulation steady-state availability estimate.

    Uses the analytical formula A = MTBF / (MTBF + eff_MTTR) for a
    single two-state component, then takes the series-system product.

    A_turbine_series = ∏ᵢ [ MTBF_i / (MTBF_i + eff_MTTR_i) ]

    This is the Phase 3 → Phase 4 conceptual bridge value — the Monte
    Carlo should converge to this when n → ∞. Used to sanity-check
    simulation results before running the full 10,000-simulation batch.
    """
    turbine_params, _ = build_scenario_params(scenario_key)
    A = 1.0
    for p in turbine_params:
        eff_mttr = p.mttr_raw_days / p.access_fraction
        A *= p.mtbf_days / (p.mtbf_days + eff_mttr)
    return A


def expected_bop_availability(scenario_key: str) -> float:
    """Pre-simulation BoP availability estimate (substation × cable)."""
    _, bop_params = build_scenario_params(scenario_key)
    A = 1.0
    for p in bop_params:
        eff_mttr = p.mttr_raw_days / p.access_fraction
        A *= p.mtbf_days / (p.mtbf_days + eff_mttr)
    return A


def print_scenario_preview() -> None:
    """Print the pre-simulation availability estimates for all scenarios.
    Call before the full Monte Carlo to verify parameter sanity."""
    print("\n── Pre-simulation availability estimates ──────────────────────────")
    print(f"{'Scenario':<15}  {'A_turbine':>12}  {'A_bop':>10}  {'A_farm_est':>12}")
    print("-" * 58)
    from scipy.stats import binom
    for key, sc in config.SCENARIOS.items():
        A_t = expected_turbine_availability(key)
        A_b = expected_bop_availability(key)
        # Farm estimate: P(≥k_min turbines up) × A_bop
        A_kofN = 1.0 - binom.cdf(
            config.FARM_K_MIN_TURBINES - 1,
            config.FARM_N_TURBINES,
            A_t,
        )
        A_farm_est = A_kofN * A_b
        print(f"{sc['label']:<15}  {A_t:>12.4f}  {A_b:>10.4f}  {A_farm_est:>12.4f}")
    print()