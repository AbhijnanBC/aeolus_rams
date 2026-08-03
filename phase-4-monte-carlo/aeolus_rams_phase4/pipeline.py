"""
aeolus_rams_phase4.pipeline
================================
End-to-end Phase 4 orchestrator.

Flow
----
1.  Print pre-simulation analytical estimates (sanity check before any compute)
2.  Run baseline Monte Carlo (10,000 simulations)
3.  Run optimised Monte Carlo (10,000 simulations)
4.  Run degraded Monte Carlo (10,000 simulations)
5.  Summarise all three scenarios into mc_summary.csv
6.  Run convergence check on baseline (validates simulation before committing)
7.  Run sensitivity sweep: Pitch System MTBF
8.  Run sensitivity sweep: Hydraulic System MTBF
9.  Run sensitivity sweep: Export Cable MTBF
10. Run sensitivity sweep: MTTR multiplier
11. Build tornado chart data
12. Run Phase 3 numerical bridge validation
13. Generate all four figures
14. Generate Phase 4 report
15. Write all outputs to --output-dir
16. Print summary to stdout

Run as a script:
    python -m aeolus_rams_phase4.pipeline \\
        --output-dir outputs/

or import and call run_phase4(...) directly from a notebook.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .component_sampler import ComponentParams
from .farm_state import simulate_farm
from .monte_carlo import run_monte_carlo, summarise_mc_results, convergence_check
from .sensitivity import (
    sweep_component_mtbf,
    sweep_bop_mtbf,
    sweep_mttr_multiplier,
    build_tornado_data,
)
from .scenarios import build_scenario_params, print_scenario_preview
from .plots import (
    plot_availability_distribution,
    plot_scenario_comparison,
    plot_tornado_chart,
    plot_bop_vs_turbine,
)
from .reporting import render_phase4_report

logger = logging.getLogger("aeolus_rams_phase4.pipeline")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class Phase4Result:
    mc_baseline: pd.DataFrame
    mc_optimised: pd.DataFrame
    mc_degraded: pd.DataFrame
    mc_summary: pd.DataFrame
    sensitivity_pitch: pd.DataFrame
    sensitivity_hydraulic: pd.DataFrame
    sensitivity_cable: pd.DataFrame
    sensitivity_mttr: pd.DataFrame
    tornado_df: pd.DataFrame
    convergence_df: pd.DataFrame | None = None
    plot_paths: dict[str, str] = field(default_factory=dict)
    report_markdown: str = ""
    bridge_test_passed: bool = False
    bridge_test_detail: str = ""


# ---------------------------------------------------------------------------
# Phase 3 numerical bridge test
# ---------------------------------------------------------------------------

def _run_bridge_test() -> tuple[bool, str]:
    """Validate Phase 4 simulation against Phase 3 R(t) at MTTR=0 boundary.

    With MTTR → 0 on all components, the fraction of time a turbine spends UP
    in a simulation over T days should converge to exp(-λ_system × T).

    This test uses a large n (50,000 turbine-years) and a single-component
    simulation to avoid Monte Carlo noise.

    Implementation note: with MTTR=0 set as a very small epsilon (1e-6 days)
    to avoid divide-by-zero in mttr_effective_days = raw/access, the component
    is repaired essentially instantly and the turbine's availability becomes
    the probability of having no failures (= R(T)).

    For a single component with MTBF=MTBF_sys and MTTR→0, run over T=1yr:
    Expected availability = R_system(1yr) = 0.4504 (from Phase 3).
    """
    lam_sys = config.PHASE3_LAMBDA_SYSTEM
    mtbf_sys = 1.0 / lam_sys  # 457.88 days

    # Simulate a single-component "system" with the full system MTBF and MTTR→0
    zero_repair_comp = ComponentParams(
        name="ZeroRepairSystem",
        mtbf_days=mtbf_sys,
        mttr_raw_days=1e-6,          # epsilon repair time
        access_fraction=1.0,         # irrelevant when MTTR→0
        confidence="test_only",
    )

    T = config.T_MISSION_DAYS   # 365.25 days
    n = 50_000
    rng = np.random.default_rng(0)

    from .turbine_state import simulate_turbine
    avail_list = []
    for _ in range(n):
        res = simulate_turbine([zero_repair_comp], T, rng)
        # If the length of down_periods is 0, it survived the whole year (R(t))
        survived = 1.0 if len(res.down_periods) == 0 else 0.0
        avail_list.append(survived)

    sim_A = float(np.mean(avail_list))
    expected_A = float(np.exp(-lam_sys * T))  # Phase 3 R_turbine(1yr) = 0.4504

    error = abs(sim_A - expected_A)
    passed = error < config.PHASE3_BRIDGE_TOLERANCE

    detail = (
        f"simulated={sim_A:.5f}, expected R_turbine(1yr)={expected_A:.5f}, "
        f"error={error:.5f} (tolerance={config.PHASE3_BRIDGE_TOLERANCE})"
    )
    return passed, detail


# ---------------------------------------------------------------------------
# Core pipeline function
# ---------------------------------------------------------------------------

def run_phase4(
    output_dir: str | Path | None = None,
    n_simulations: int = config.MAIN_N_SIMULATIONS,
    write_outputs: bool = True,
    run_sweep: bool = True,
    run_convergence: bool = True,
) -> Phase4Result:
    """Run the complete Phase 4 Monte Carlo pipeline.

    Parameters
    ----------
    output_dir : Path, optional
        Where to write all outputs.
    n_simulations : int
        Simulations per scenario. Default: config.MAIN_N_SIMULATIONS (10,000).
        Set lower (e.g. 1,000) for smoke-test runs.
    write_outputs : bool
        False for in-memory only (used in integration tests).
    run_sweep : bool
        False to skip sensitivity sweeps (for quick testing).
    run_convergence : bool
        False to skip convergence check.
    """
    output_dir = Path(output_dir) if output_dir else config.DEFAULT_OUTPUT_DIR
    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Preview ───────────────────────────────────────────────────
    print_scenario_preview()

    # ── Step 2–4: Three scenario Monte Carlo runs ─────────────────────────
    results_by_scenario: dict[str, pd.DataFrame] = {}
    for key in ["baseline", "optimised", "degraded"]:
        logger.info("Running %s Monte Carlo (%d simulations)...", key, n_simulations)
        turbine_params, bop_params = build_scenario_params(key)
        mc_df = run_monte_carlo(
            turbine_params, bop_params,
            config.FARM_N_TURBINES, config.FARM_K_MIN_TURBINES,
            config.T_MISSION_DAYS, n_simulations,
            seed=config.RANDOM_SEED,
            scenario_label=key,
        )
        results_by_scenario[key] = mc_df

    mc_baseline = results_by_scenario["baseline"]
    mc_optimised = results_by_scenario["optimised"]
    mc_degraded = results_by_scenario["degraded"]

    # ── Step 5: Summary table ─────────────────────────────────────────────
    logger.info("Computing summary statistics...")
    summary_parts = [
        summarise_mc_results(mc_baseline, "baseline"),
        summarise_mc_results(mc_optimised, "optimised"),
        summarise_mc_results(mc_degraded, "degraded"),
    ]
    mc_summary = pd.concat(summary_parts, ignore_index=True)

    # ── Step 6: Convergence check ─────────────────────────────────────────
    convergence_df: pd.DataFrame | None = None
    if run_convergence:
        logger.info("Running convergence check...")
        turbine_params_bl, bop_params_bl = build_scenario_params("baseline")
        convergence_df = convergence_check(
            turbine_params_bl, bop_params_bl,
            config.FARM_N_TURBINES, config.FARM_K_MIN_TURBINES,
            config.T_MISSION_DAYS,
            seed=config.RANDOM_SEED,
        )
        logger.info("Convergence table:\n%s", convergence_df.to_string(index=False))

    # ── Steps 7–11: Sensitivity sweeps ────────────────────────────────────
    sweep_dfs: dict[str, pd.DataFrame] = {}
    if run_sweep:
        turbine_bl, bop_bl = build_scenario_params("baseline")

        logger.info("Sensitivity sweep: Pitch System MTBF...")
        sweep_dfs["pitch"] = sweep_component_mtbf(
            turbine_bl, bop_bl,
            "Pitch System", config.SWEEP_PITCH_MTBF,
            config.FARM_N_TURBINES, config.FARM_K_MIN_TURBINES,
            config.T_MISSION_DAYS, config.SWEEP_N_SIMULATIONS,
            seed=config.RANDOM_SEED,
        )

        logger.info("Sensitivity sweep: Hydraulic System MTBF...")
        sweep_dfs["hydraulic"] = sweep_component_mtbf(
            turbine_bl, bop_bl,
            "Hydraulic System", config.SWEEP_HYDRAULIC_MTBF,
            config.FARM_N_TURBINES, config.FARM_K_MIN_TURBINES,
            config.T_MISSION_DAYS, config.SWEEP_N_SIMULATIONS,
            seed=config.RANDOM_SEED,
        )

        logger.info("Sensitivity sweep: Export Cable MTBF...")
        sweep_dfs["cable"] = sweep_bop_mtbf(
            turbine_bl, bop_bl,
            "Export Cable", config.SWEEP_CABLE_MTBF,
            config.FARM_N_TURBINES, config.FARM_K_MIN_TURBINES,
            config.T_MISSION_DAYS, config.SWEEP_N_SIMULATIONS,
            seed=config.RANDOM_SEED,
        )

        logger.info("Sensitivity sweep: MTTR multiplier...")
        sweep_dfs["mttr"] = sweep_mttr_multiplier(
            turbine_bl, bop_bl,
            config.SWEEP_MTTR_MULT,
            config.FARM_N_TURBINES, config.FARM_K_MIN_TURBINES,
            config.T_MISSION_DAYS, config.SWEEP_N_SIMULATIONS,
            seed=config.RANDOM_SEED,
        )

        baseline_A_farm_mean = float(mc_baseline["A_farm"].mean())
        tornado_df = build_tornado_data(sweep_dfs, baseline_A_farm_mean)
    else:
        logger.warning("Sensitivity sweeps skipped (run_sweep=False).")
        sweep_dfs = {k: pd.DataFrame() for k in ["pitch", "hydraulic", "cable", "mttr"]}
        tornado_df = pd.DataFrame()

    # ── Step 12: Phase 3 bridge test ──────────────────────────────────────
    logger.info("Running Phase 3 numerical bridge test...")
    bridge_passed, bridge_detail = _run_bridge_test()
    status = "PASSED" if bridge_passed else "FAILED"
    logger.info("Bridge test: %s — %s", status, bridge_detail)

    # ── Step 13: Plots ────────────────────────────────────────────────────
    plot_paths: dict[str, str] = {}
    if write_outputs:
        logger.info("Generating figures...")
        p1 = plot_availability_distribution(
            mc_baseline, output_dir / "availability_distribution.png"
        )
        plot_paths["availability_distribution"] = str(p1)

        p2 = plot_scenario_comparison(
            {"baseline": mc_baseline, "optimised": mc_optimised, "degraded": mc_degraded},
            output_dir / "scenario_comparison.png",
        )
        plot_paths["scenario_comparison"] = str(p2)

        if not tornado_df.empty:
            p3 = plot_tornado_chart(
                tornado_df, output_dir / "tornado_chart.png",
                # CHANGE this line:
                baseline_A_farm=float(mc_baseline["A_farm"].mean()),
            )
            plot_paths["tornado_chart"] = str(p3)

        p4 = plot_bop_vs_turbine(mc_summary, output_dir / "bop_vs_turbine.png")
        plot_paths["bop_vs_turbine"] = str(p4)

    # ── Step 14: Report ───────────────────────────────────────────────────
    logger.info("Generating Phase 4 Markdown report...")
    report_md = render_phase4_report(
        mc_summary=mc_summary,
        tornado_df=tornado_df,
        baseline_A_farm_mean=float(mc_baseline["A_farm"].mean()),
        plot_paths=plot_paths,
        bridge_test_passed=bridge_passed,
        bridge_test_detail=bridge_detail,
        convergence_df=convergence_df,
    )

    # ── Step 15: Write outputs ────────────────────────────────────────────
    result = Phase4Result(
        mc_baseline=mc_baseline,
        mc_optimised=mc_optimised,
        mc_degraded=mc_degraded,
        mc_summary=mc_summary,
        sensitivity_pitch=sweep_dfs.get("pitch", pd.DataFrame()),
        sensitivity_hydraulic=sweep_dfs.get("hydraulic", pd.DataFrame()),
        sensitivity_cable=sweep_dfs.get("cable", pd.DataFrame()),
        sensitivity_mttr=sweep_dfs.get("mttr", pd.DataFrame()),
        tornado_df=tornado_df,
        convergence_df=convergence_df,
        plot_paths=plot_paths,
        report_markdown=report_md,
        bridge_test_passed=bridge_passed,
        bridge_test_detail=bridge_detail,
    )

    if write_outputs:
        _write_outputs(result, output_dir)

    return result


def _write_outputs(result: Phase4Result, output_dir: Path) -> None:
    result.mc_baseline.to_csv(output_dir / "mc_results_baseline.csv", index=False)
    result.mc_optimised.to_csv(output_dir / "mc_results_optimised.csv", index=False)
    result.mc_degraded.to_csv(output_dir / "mc_results_degraded.csv", index=False)
    result.mc_summary.to_csv(output_dir / "mc_summary.csv", index=False)

    for name, df in [
        ("sensitivity_pitch", result.sensitivity_pitch),
        ("sensitivity_hydraulic", result.sensitivity_hydraulic),
        ("sensitivity_cable", result.sensitivity_cable),
        ("sensitivity_mttr", result.sensitivity_mttr),
    ]:
        if not df.empty:
            df.to_csv(output_dir / f"{name}.csv", index=False)

    if not result.tornado_df.empty:
        result.tornado_df.to_csv(output_dir / "tornado_data.csv", index=False)

    if result.convergence_df is not None:
        result.convergence_df.to_csv(output_dir / "convergence_check.csv", index=False)

    (output_dir / "phase4_report.md").write_text(result.report_markdown, encoding="utf-8")
    logger.info("All Phase 4 outputs written to %s", output_dir.resolve())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aeolus-rams-phase4",
        description=(
            "AEOLUS-RAMS Phase 4 — Monte Carlo Simulation: farm availability "
            "distribution under repair and offshore weather-gated access."
        ),
    )
    p.add_argument(
        "--output-dir", type=Path, default=config.DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {config.DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--n-simulations", type=int, default=config.MAIN_N_SIMULATIONS,
        help=f"Simulations per scenario (default: {config.MAIN_N_SIMULATIONS}). "
             "Use 1000 for a quick smoke test.",
    )
    p.add_argument(
        "--skip-sweep", action="store_true",
        help="Skip sensitivity sweeps (faster; omits Figures 3 and tornado data).",
    )
    p.add_argument(
        "--skip-convergence", action="store_true",
        help="Skip convergence check (saves ~2 min).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        result = run_phase4(
            output_dir=args.output_dir,
            n_simulations=args.n_simulations,
            run_sweep=not args.skip_sweep,
            run_convergence=not args.skip_convergence,
        )
    except Exception:
        logger.exception("Phase 4 run failed")
        return 1

    baseline_summary = result.mc_summary[result.mc_summary["scenario"] == "baseline"]
    mean_A_farm = float(
        baseline_summary.loc[baseline_summary["metric"] == "A_farm", "mean"].iloc[0]
    )
    p5 = float(
        baseline_summary.loc[baseline_summary["metric"] == "A_farm", "p5"].iloc[0]
    )
    p95 = float(
        baseline_summary.loc[baseline_summary["metric"] == "A_farm", "p95"].iloc[0]
    )

    print()
    print("=" * 70)
    print("PHASE 4 COMPLETE")
    print("=" * 70)
    print(f"  Simulations per scenario : {args.n_simulations:,}")
    print(f"  Baseline mean A_farm     : {mean_A_farm:.4f}")
    print(f"  Baseline 90% CI          : [{p5:.4f}, {p95:.4f}]")
    bridge_status = "PASSED ✓" if result.bridge_test_passed else "FAILED ✗"
    print(f"  Phase 3 bridge test      : {bridge_status}")
    print(f"  {result.bridge_test_detail}")
    print(f"  Outputs written to       : {args.output_dir.resolve()}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())