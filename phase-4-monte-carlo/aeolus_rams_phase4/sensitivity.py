"""
aeolus_rams_phase4.sensitivity
=================================
Layer 5 — single-parameter sweep engine.

Three sweep types (Section 4.4.5):
  1. Component MTBF sweep — replaces one component's MTBF with each
     value in sweep_values, runs a Monte Carlo at n_simulations per
     point, records mean A_farm.
  2. MTTR multiplier sweep — multiplies ALL components' mttr_raw_days
     by each value in sweep_values.
  3. BoP MTBF sweep — replaces a BoP component's MTBF (separate call
     using bop_params as the target list).

All sweeps run at SWEEP_N_SIMULATIONS = 2,000 per point × ~22 points
= ~44,000 simulations total per sweep. Runtime ≈ 4 minutes per sweep.

Output: one CSV per sweep with columns:
  component, param_varied, param_value,
  A_farm_mean, A_farm_std, A_farm_p5, A_farm_p95

These four CSVs feed directly into the tornado chart (Figure 3).
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pandas as pd

from .component_sampler import ComponentParams
from .monte_carlo import run_monte_carlo

logger = logging.getLogger("aeolus_rams_phase4.sensitivity")


def sweep_component_mtbf(
    baseline_turbine_params: list[ComponentParams],
    bop_params: list[ComponentParams],
    component_name: str,
    sweep_values: np.ndarray,
    N_turbines: int,
    k_min: int,
    T_days: float,
    n_simulations: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Sweep one turbine component's MTBF and record A_farm distribution.

    Parameters
    ----------
    baseline_turbine_params : list[ComponentParams]
        The 13-component baseline list. The named component will be replaced.
    component_name : str
        Must exactly match a ComponentParams.name in baseline_turbine_params.
    sweep_values : np.ndarray
        MTBF values (days) to sweep over.
    n_simulations : int
        Simulations per sweep point (default: SWEEP_N_SIMULATIONS = 2000).

    Returns
    -------
    pd.DataFrame
        One row per sweep value. Columns:
        component, param_varied, param_value,
        A_farm_mean, A_farm_std, A_farm_p5, A_farm_p95.
    """
    if not any(p.name == component_name for p in baseline_turbine_params):
        raise ValueError(
            f"Component '{component_name}' not found in baseline_turbine_params. "
            f"Available: {[p.name for p in baseline_turbine_params]}"
        )

    rows = []
    for val in sweep_values:
        modified_params = [
            dataclasses.replace(p, mtbf_days=val)
            if p.name == component_name
            else p
            for p in baseline_turbine_params
        ]
        mc = run_monte_carlo(
            modified_params, bop_params,
            N_turbines, k_min, T_days, n_simulations, seed,
            scenario_label=f"{component_name}_mtbf_{val:.0f}",
        )
        rows.append({
            "component": component_name,
            "param_varied": "mtbf_days",
            "param_value": val,
            "A_farm_mean": mc["A_farm"].mean(),
            "A_farm_std": mc["A_farm"].std(),
            "A_farm_p5": mc["A_farm"].quantile(0.05),
            "A_farm_p95": mc["A_farm"].quantile(0.95),
        })
        logger.info(
            "Sweep %s MTBF=%.0fd → mean A_farm=%.4f",
            component_name, val, rows[-1]["A_farm_mean"]
        )

    return pd.DataFrame(rows)


def sweep_bop_mtbf(
    turbine_params: list[ComponentParams],
    baseline_bop_params: list[ComponentParams],
    bop_component_name: str,
    sweep_values: np.ndarray,
    N_turbines: int,
    k_min: int,
    T_days: float,
    n_simulations: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Sweep a BoP component's MTBF and record A_farm distribution.

    Same as sweep_component_mtbf but targets baseline_bop_params.
    """
    if not any(p.name == bop_component_name for p in baseline_bop_params):
        raise ValueError(
            f"BoP component '{bop_component_name}' not found. "
            f"Available: {[p.name for p in baseline_bop_params]}"
        )

    rows = []
    for val in sweep_values:
        modified_bop = [
            dataclasses.replace(p, mtbf_days=val)
            if p.name == bop_component_name
            else p
            for p in baseline_bop_params
        ]
        mc = run_monte_carlo(
            turbine_params, modified_bop,
            N_turbines, k_min, T_days, n_simulations, seed,
            scenario_label=f"{bop_component_name}_mtbf_{val:.0f}",
        )
        rows.append({
            "component": bop_component_name,
            "param_varied": "mtbf_days",
            "param_value": val,
            "A_farm_mean": mc["A_farm"].mean(),
            "A_farm_std": mc["A_farm"].std(),
            "A_farm_p5": mc["A_farm"].quantile(0.05),
            "A_farm_p95": mc["A_farm"].quantile(0.95),
        })
        logger.info(
            "BoP sweep %s MTBF=%.0fd → mean A_farm=%.4f",
            bop_component_name, val, rows[-1]["A_farm_mean"]
        )

    return pd.DataFrame(rows)


def sweep_mttr_multiplier(
    baseline_turbine_params: list[ComponentParams],
    baseline_bop_params: list[ComponentParams],
    multiplier_values: np.ndarray,
    N_turbines: int,
    k_min: int,
    T_days: float,
    n_simulations: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Sweep a global MTTR multiplier applied to ALL components.

    Models logistics capability: multiplier < 1 = dedicated SOV,
    multiplier > 1 = severe access constraints/shared vessels.

    Note: the multiplier is applied to mttr_raw_days (before the
    access_fraction division), which effectively scales both the
    raw repair duration and the expected weather-waiting time
    proportionally — consistent with the scenario definition in
    Section 4.5.
    """
    rows = []
    for mult in multiplier_values:
        modified_turbine = [
            dataclasses.replace(p, mttr_raw_days=p.mttr_raw_days * mult)
            for p in baseline_turbine_params
        ]
        modified_bop = [
            dataclasses.replace(p, mttr_raw_days=p.mttr_raw_days * mult)
            for p in baseline_bop_params
        ]
        mc = run_monte_carlo(
            modified_turbine, modified_bop,
            N_turbines, k_min, T_days, n_simulations, seed,
            scenario_label=f"mttr_mult_{mult:.2f}",
        )
        rows.append({
            "component": "ALL",
            "param_varied": "mttr_multiplier",
            "param_value": mult,
            "A_farm_mean": mc["A_farm"].mean(),
            "A_farm_std": mc["A_farm"].std(),
            "A_farm_p5": mc["A_farm"].quantile(0.05),
            "A_farm_p95": mc["A_farm"].quantile(0.95),
        })
        logger.info(
            "MTTR multiplier=%.2f → mean A_farm=%.4f",
            mult, rows[-1]["A_farm_mean"]
        )

    return pd.DataFrame(rows)


def build_tornado_data(
    sweep_dfs: dict[str, pd.DataFrame],
    baseline_A_farm: float,
) -> pd.DataFrame:
    """Compute tornado chart inputs from all four sweep DataFrames.

    For each sweep, compute: change in mean A_farm from worst to best
    swept value. Returns a DataFrame sorted by impact (widest bar first).

    Parameters
    ----------
    sweep_dfs : dict[str, pd.DataFrame]
        Keys: 'pitch', 'hydraulic', 'cable', 'mttr'.
        Values: DataFrames from sweep_* functions.
    baseline_A_farm : float
        Mean A_farm from the baseline Monte Carlo (for centring the bars).

    Returns
    -------
    pd.DataFrame
        Columns: sweep_name, A_farm_at_worst, A_farm_at_best,
                 impact_total, baseline_A_farm.
        Sorted by impact_total descending.
    """
    rows = []
    for name, df in sweep_dfs.items():
        a_min = df["A_farm_mean"].min()
        a_max = df["A_farm_mean"].max()
        rows.append({
            "sweep_name": name,
            "A_farm_at_worst": a_min,
            "A_farm_at_best": a_max,
            "impact_total": a_max - a_min,
            "baseline_A_farm": baseline_A_farm,
        })
    return (
        pd.DataFrame(rows)
        .sort_values("impact_total", ascending=False)
        .reset_index(drop=True)
    )