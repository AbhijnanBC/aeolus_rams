"""
aeolus_rams_phase4.farm_state
================================
Layer 3 of the simulation architecture.

Aggregates N independent turbine simulations and 2 BoP component
simulations into farm-level availability metrics using the EXACT
joint-timeline method — not the binomial approximation in the Phase 4
plan's first draft.

Why exact instead of binomial approximation
-------------------------------------------
The binomial approach (compute mean A_turbine, then use scipy.stats.binom
to find P(k_min or more turbines up)) is mathematically correct *only*
when individual turbine availabilities are all equal and independent.

In a single Monte Carlo draw:
  - All 22 turbines ARE simulated independently (correct).
  - But their individual availabilities vary randomly around the mean
    (because each turbine gets its own random TBF/TTR draws).
  - The binomial at the *mean* systematically understates the probability
    of the farm being below threshold in draws where several turbines
    happen to fail simultaneously.

The exact joint-timeline method:
  1. Build a sorted list of all event times across all turbines + BoP.
  2. For each interval, count turbines UP (= not in any down period).
  3. Check if count ≥ k_min AND BoP is up.
  4. Accumulate time where farm was delivering power.

This is O(N × events_per_turbine) per simulation — fast because each
turbine generates only ~0.8 failures/year → ~37 events across 22 turbines.

Independence assumption
-----------------------
Turbines fail independently. Common-cause failures (grid events, storm
surge, simultaneous icing) are NOT modelled. This is explicitly stated
in the report's assumptions section and consistent with Phase 3's
topology.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .component_sampler import ComponentParams
from .turbine_state import simulate_turbine, TurbineSimResult


@dataclass
class FarmSimResult:
    """Metrics from one simulated farm-year."""

    A_farm: float
    """Fraction of T_days farm delivered ≥ k_min turbines AND BoP up."""

    A_turbine_mean: float
    """Mean turbine availability across all N turbines."""

    A_bop: float
    """BoP availability (substation × cable — both in series)."""

    A_kofN_exact: float
    """Fraction of T_days ≥ k_min turbines were simultaneously up,
    ignoring BoP. = A_farm / A_bop when BoP is independent.
    Provided for decomposition in Figure 4."""

    n_turbine_up_mean: float
    """Time-averaged number of simultaneously UP turbines over T_days."""

    turbine_comp_down_days: dict[str, float]
    """Summed per-component downtime across all N turbines (days)."""


def simulate_farm(
    turbine_component_params: list[ComponentParams],
    bop_params: list[ComponentParams],
    N_turbines: int,
    k_min_turbines: int,
    T_days: float,
    rng: np.random.Generator,
) -> FarmSimResult:
    """Simulate one farm-year using the exact joint-timeline method.

    Parameters
    ----------
    turbine_component_params : list[ComponentParams]
        13-component list for each turbine (identical for all turbines —
        independence assumption means each gets different random draws,
        not different parameters).
    bop_params : list[ComponentParams]
        [Offshore Substation, Export Cable] — both in series with the
        turbine array.
    N_turbines : int
        Number of turbines (22 for Farm C).
    k_min_turbines : int
        Minimum turbines required for contractual output (15 for Farm C).
    T_days : float
        Mission duration in days (365.25 for one simulated year).
    rng : np.random.Generator
        Shared seeded generator — passed through to all turbine/BoP sims.

    Returns
    -------
    FarmSimResult
        A_farm, A_turbine_mean, A_bop, A_kofN_exact, n_turbine_up_mean,
        turbine_comp_down_days.
    """
    # ── Step 1: Simulate all N turbines independently ─────────────────────
    turbine_results: list[TurbineSimResult] = []
    agg_comp_down: dict[str, float] = {p.name: 0.0 for p in turbine_component_params}

    for _ in range(N_turbines):
        res = simulate_turbine(turbine_component_params, T_days, rng)
        turbine_results.append(res)
        for comp_name, down_d in res.comp_down_days.items():
            agg_comp_down[comp_name] += down_d

    A_turbine_mean = float(np.mean([r.availability for r in turbine_results]))

    # ── Step 2: Simulate BoP (substation + cable, series) ─────────────────
    # Each BoP component is simulated as a single-component "turbine".
    bop_down_periods: list[list[tuple[float, float]]] = []
    for bop_comp in bop_params:
        bop_res = simulate_turbine([bop_comp], T_days, rng)
        bop_down_periods.append(bop_res.down_periods)

    # BoP is DOWN if EITHER substation OR cable is down (both in series).
    # Build combined BoP down-period list (union of both BoP component outages).
    all_bop_events: set[float] = {0.0, T_days}
    for periods in bop_down_periods:
        for s, e in periods:
            all_bop_events.add(s)
            all_bop_events.add(e)
    bop_event_times = sorted(all_bop_events)

    bop_down_time = 0.0
    for i in range(len(bop_event_times) - 1):
        t0, t1 = bop_event_times[i], bop_event_times[i + 1]
        t_mid = 0.5 * (t0 + t1)
        if any(
            s <= t_mid < e
            for periods in bop_down_periods
            for s, e in periods
        ):
            bop_down_time += (t1 - t0)

    A_bop = 1.0 - bop_down_time / T_days

    # ── Step 3: Exact joint timeline — count UP turbines at each interval ──
    # Collect all event times from all turbines AND BoP.
    all_events: set[float] = {0.0, T_days}
    for res in turbine_results:
        for s, e in res.down_periods:
            all_events.add(s)
            all_events.add(e)
    for periods in bop_down_periods:
        for s, e in periods:
            all_events.add(s)
            all_events.add(e)
    event_times = sorted(all_events)

    farm_up_time = 0.0
    kofn_up_time = 0.0
    turbine_up_accumulator = 0.0  # for mean turbines UP

    for i in range(len(event_times) - 1):
        t0, t1 = event_times[i], event_times[i + 1]
        dt = t1 - t0
        t_mid = 0.5 * (t0 + t1)

        # Count turbines UP at t_mid
        n_up = 0
        for res in turbine_results:
            turbine_down_at_tmid = any(s <= t_mid < e for s, e in res.down_periods)
            if not turbine_down_at_tmid:
                n_up += 1

        turbine_up_accumulator += n_up * dt

        # Check k-of-N threshold
        k_satisfied = n_up >= k_min_turbines
        if k_satisfied:
            kofn_up_time += dt

        # Check BoP
        bop_down_at_tmid = any(
            s <= t_mid < e
            for periods in bop_down_periods
            for s, e in periods
        )

        if k_satisfied and not bop_down_at_tmid:
            farm_up_time += dt

    A_farm = farm_up_time / T_days
    A_kofN_exact = kofn_up_time / T_days
    n_turbine_up_mean = turbine_up_accumulator / T_days

    return FarmSimResult(
        A_farm=A_farm,
        A_turbine_mean=A_turbine_mean,
        A_bop=A_bop,
        A_kofN_exact=A_kofN_exact,
        n_turbine_up_mean=n_turbine_up_mean,
        turbine_comp_down_days=agg_comp_down,
    )