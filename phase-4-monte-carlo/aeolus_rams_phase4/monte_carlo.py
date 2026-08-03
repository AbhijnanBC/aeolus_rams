"""
aeolus_rams_phase4.monte_carlo
================================
Layer 4 of the simulation architecture — the M-simulation orchestrator.

Runs n_simulations independent farm-year simulations, each with a
different random draw of failure times, repair durations, and
weather-access waiting times. Returns a DataFrame with one row per
simulation — the raw material for all four output figures.

Simulation count guidance:
  n=1,000   — smoke test (~30s), ±0.7% CI on mean A_farm
  n=10,000  — standard run (~5min), ±0.22% CI on mean A_farm
  n=50,000  — high-precision (~25min), ±0.1% CI on mean A_farm

The 95% CI on mean A_farm at n simulations:
    CI_95 ≈ ±1.96 × σ_A / √n
where σ_A ≈ 0.035 (estimated from typical single-turbine availability
variance ÷ effective farm size ≈ 22/3).
For n=10,000: ±1.96 × 0.035 / √10,000 = ±0.0007 = ±0.07%.
This is sufficient precision for the 3-scenario comparison.

Performance note:
  The exact joint-timeline farm simulation is O(N_turbines × events)
  where events ≈ 40–60 per farm-year. At n=10,000 that is ~600,000
  event-time iterations — sub-second in numpy. Total runtime is dominated
  by Python loop overhead in simulate_turbine; use tqdm for progress.
"""

from __future__ import annotations

import logging
from dataclasses import asdict

import numpy as np
import pandas as pd

from .farm_state import simulate_farm, FarmSimResult
from .component_sampler import ComponentParams

logger = logging.getLogger("aeolus_rams_phase4.monte_carlo")


def run_monte_carlo(
    turbine_component_params: list[ComponentParams],
    bop_params: list[ComponentParams],
    N_turbines: int,
    k_min: int,
    T_days: float,
    n_simulations: int,
    seed: int = 42,
    scenario_label: str = "baseline",
) -> pd.DataFrame:
    """Run n_simulations independent farm-year simulations.

    Parameters
    ----------
    turbine_component_params : list[ComponentParams]
        13 turbine component parameter rows.
    bop_params : list[ComponentParams]
        2 BoP parameter rows [Offshore Substation, Export Cable].
    N_turbines : int
        Number of turbines (22).
    k_min : int
        Minimum turbines required (15).
    T_days : float
        Mission duration (365.25).
    n_simulations : int
        Number of independent simulations to run.
    seed : int
        RNG seed. All n_simulations share one Generator — each call to
        simulate_farm advances the state deterministically.
    scenario_label : str
        Tag written into the 'scenario' column of the output DataFrame.

    Returns
    -------
    pd.DataFrame
        One row per simulation. Columns:
        sim_id, scenario, A_farm, A_turbine_mean, A_bop, A_kofN_exact,
        n_turbine_up_mean, [comp_down_<name> for each turbine component].
    """
    rng = np.random.default_rng(seed)
    comp_names = [p.name for p in turbine_component_params]
    rows: list[dict] = []

    for sim_id in range(n_simulations):
        if sim_id % 1000 == 0 and sim_id > 0:
            logger.info(
                "Scenario %s: simulation %d/%d", scenario_label, sim_id, n_simulations
            )

        result: FarmSimResult = simulate_farm(
            turbine_component_params, bop_params,
            N_turbines, k_min, T_days, rng,
        )

        row: dict = {
            "sim_id": sim_id,
            "scenario": scenario_label,
            "A_farm": result.A_farm,
            "A_turbine_mean": result.A_turbine_mean,
            "A_bop": result.A_bop,
            "A_kofN_exact": result.A_kofN_exact,
            "n_turbine_up_mean": result.n_turbine_up_mean,
        }
        # Per-component downtime (summed over all 22 turbines, in days)
        for comp_name in comp_names:
            col = f"comp_down_{comp_name.lower().replace('/', '_').replace(' ', '_')}"
            row[col] = result.turbine_comp_down_days.get(comp_name, 0.0)

        rows.append(row)

    logger.info(
        "Scenario %s: completed %d simulations. Mean A_farm = %.4f",
        scenario_label, n_simulations, np.mean([r["A_farm"] for r in rows]),
    )
    return pd.DataFrame(rows)


def summarise_mc_results(
    mc_df: pd.DataFrame,
    scenario_label: str | None = None,
) -> pd.DataFrame:
    """Summary statistics from Monte Carlo output — for the report table.

    Computes: mean, std, p5, p25, median, p75, p95 for each key metric.

    Parameters
    ----------
    mc_df : pd.DataFrame
        Output of run_monte_carlo().
    scenario_label : str, optional
        Scenario tag. If None, read from mc_df['scenario'].

    Returns
    -------
    pd.DataFrame
        Columns: scenario, metric, mean, std, p5, p25, median, p75, p95.
    """
    label = scenario_label or (
        mc_df["scenario"].iloc[0] if "scenario" in mc_df.columns else "unknown"
    )
    metrics = ["A_farm", "A_turbine_mean", "A_bop", "A_kofN_exact", "n_turbine_up_mean"]
    rows = []
    for col in metrics:
        if col not in mc_df.columns:
            continue
        s = mc_df[col]
        rows.append({
            "scenario": label,
            "metric": col,
            "mean": s.mean(),
            "std": s.std(),
            "p5": s.quantile(0.05),
            "p25": s.quantile(0.25),
            "median": s.median(),
            "p75": s.quantile(0.75),
            "p95": s.quantile(0.95),
        })
    return pd.DataFrame(rows)


def convergence_check(
    turbine_component_params: list[ComponentParams],
    bop_params: list[ComponentParams],
    N_turbines: int,
    k_min: int,
    T_days: float,
    seed: int = 42,
    n_max: int = 10_000,
) -> pd.DataFrame:
    """Run up to n_max simulations, recording mean A_farm at checkpoints.

    Used to verify that the Monte Carlo has converged before committing to
    the full run. Mean A_farm should stabilise (change < 0.1%) by ~3,000
    simulations. If it hasn't by 5,000, the rare-event check should be
    run (see docstring of run_monte_carlo).

    Returns
    -------
    pd.DataFrame
        Columns: n_simulations, mean_A_farm, std_A_farm, delta_mean
        (change from previous checkpoint).
    """
    checkpoints = [100, 500, 1000, 2000, 5000, n_max]
    checkpoints = sorted(set(n for n in checkpoints if n <= n_max))

    rng = np.random.default_rng(seed)
    all_A_farm: list[float] = []
    rows = []
    prev_mean = None

    cp_iter = iter(checkpoints)
    next_cp = next(cp_iter, None)

    for sim_id in range(n_max):
        result = simulate_farm(
            turbine_component_params, bop_params,
            N_turbines, k_min, T_days, rng,
        )
        all_A_farm.append(result.A_farm)

        if next_cp is not None and (sim_id + 1) == next_cp:
            cur_mean = float(np.mean(all_A_farm))
            cur_std = float(np.std(all_A_farm))
            delta = abs(cur_mean - prev_mean) if prev_mean is not None else float("nan")
            rows.append({
                "n_simulations": sim_id + 1,
                "mean_A_farm": cur_mean,
                "std_A_farm": cur_std,
                "delta_mean": delta,
            })
            prev_mean = cur_mean
            next_cp = next(cp_iter, None)

    return pd.DataFrame(rows)