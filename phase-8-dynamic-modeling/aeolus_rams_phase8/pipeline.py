"""
aeolus_rams_phase8.pipeline
============================
CLI orchestrator — reads Phase 2 and Phase 7 upstream CSVs,
runs all three analytical pillars, writes all outputs.

Usage
-----
    aeolus-rams-phase8 [--phase7-dir PATH] [--phase2-dir PATH]
                       [--output-dir PATH] [--seed INT] [--no-plots]

    python -m aeolus_rams_phase8.pipeline --output-dir outputs/
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from . import synthetic_data
from . import bayesian_update
from . import cbm_optimizer
from . import reporting


def _try_load_upstream(
    csv_path: Path,
    phase_name: str,
) -> pd.DataFrame | None:
    """Attempt to load an upstream CSV; return None and warn if missing."""
    if csv_path.exists():
        return pd.read_csv(csv_path)
    print(
        f"  [WARN] {phase_name} CSV not found at {csv_path}. "
        f"Using config.py fallback constants."
    )
    return None


def _load_phase2_params(phase2_dir: Path) -> dict:
    """
    Read Phase 2 mtbf_table.csv for Pitch and Bearing Weibull parameters.
    Falls back to config.py constants if file absent.
    """
    mtbf_path = phase2_dir / "outputs" / "mtbf_table.csv"
    df = _try_load_upstream(mtbf_path, "Phase 2 MTBF table")

    params = {
        "bearing_beta": config.BEARING_BETA,
        "bearing_eta":  config.BEARING_ETA,
        "pitch_beta":   config.PITCH_BETA,
        "pitch_eta":    config.PITCH_ETA,
        "pitch_mttf":   config.PITCH_MTTF,
        "source":       "config.py fallback (Phase 2 CSV not found)",
    }

    if df is not None:
        # Attempt to read Pitch System row (Tier A)
        pitch_mask   = df["component"].str.contains("Pitch", case=False, na=False)
        bearing_mask = df["component"].str.contains("Bearing", case=False, na=False)

        if pitch_mask.any() and "beta" in df.columns:
            row = df[pitch_mask].iloc[0]
            params["pitch_beta"] = float(row["beta"])
            if "MTTF_days" in df.columns:
                params["pitch_mttf"] = float(row["MTTF_days"])
                params["pitch_eta"]  = config.compute_eta_from_mttf(
                    params["pitch_mttf"], params["pitch_beta"]
                )

        if bearing_mask.any() and "beta" in df.columns:
            row = df[bearing_mask].iloc[0]
            params["bearing_beta"] = float(row["beta"])
            # bearing eta from literature; fall back to config
            if "eta_days" in df.columns:
                params["bearing_eta"] = float(row["eta_days"])

        params["source"] = str(mtbf_path)

    print(f"  Phase 2 params loaded from: {params['source']}")
    print(f"    Bearing: β={params['bearing_beta']}, η={params['bearing_eta']:.0f}d")
    print(f"    Pitch:   β={params['pitch_beta']:.4f}, η={params['pitch_eta']:.1f}d")
    return params


def _load_phase7_params(phase7_dir: Path) -> dict:
    """
    Read Phase 7 age_replacement_table.csv for cost parameters and T*.
    Falls back to config.py constants if file absent.
    """
    age_path = phase7_dir / "outputs" / "age_replacement_table.csv"
    df = _try_load_upstream(age_path, "Phase 7 age replacement table")

    params = {
        "bearing_Cp": config.BEARING_Cp,
        "bearing_Cf": config.BEARING_Cf,
        "bearing_T_star": config.BEARING_T_STAR_PHASE7,
        "pitch_Cp":   config.PITCH_Cp,
        "pitch_Cf":   config.PITCH_Cf,
        "source":     "config.py fallback (Phase 7 CSV not found)",
    }

    if df is not None:
        bearing_mask = df["component"].str.contains("Bearing", case=False, na=False)
        pitch_mask   = df["component"].str.contains("Pitch",   case=False, na=False)

        col_map = {
            "Cp_usd":   "Cp",
            "Cf_usd":   "Cf",
            "T_star_days": "T_star",
        }

        if bearing_mask.any():
            row = df[bearing_mask].iloc[0]
            for csv_col, key in col_map.items():
                if csv_col in df.columns:
                    params[f"bearing_{key}"] = float(row[csv_col])

        if pitch_mask.any():
            row = df[pitch_mask].iloc[0]
            for csv_col in ["Cp_usd", "Cf_usd"]:
                if csv_col in df.columns:
                    key = csv_col.replace("_usd", "")
                    params[f"pitch_{key}"] = float(row[csv_col])

        params["source"] = str(age_path)

    print(f"  Phase 7 params loaded from: {params['source']}")
    print(f"    Bearing: Cp=${params['bearing_Cp']:,.0f}, Cf=${params['bearing_Cf']:,.0f}, T*={params['bearing_T_star']:.0f}d")
    print(f"    Pitch:   Cp=${params['pitch_Cp']:,.0f},  Cf=${params['pitch_Cf']:,.0f}")
    return params


def _separator(char: str = "─", width: int = 60) -> None:
    print(char * width)


def run_phase8(
    phase7_dir: Path = Path("../phase-7-preventive-maintenance"),
    phase2_dir: Path = Path("../phase-2-weibull-mtbf-hazard"),
    output_dir: Path = Path("outputs"),
    seed:       int  = config.SYNTHETIC_SEED,
    no_plots:   bool = False,
) -> dict:
    """
    Full Phase 8 pipeline.

    Returns a dict of all results for programmatic access.
    """
    t0 = time.time()

    print()
    _separator("═")
    print("  AEOLUS-RAMS — Phase 8: Dynamic Reliability & Predictive Modeling")
    _separator("═")
    print()

    # ------------------------------------------------------------------
    # Step 1: Load upstream parameters
    # ------------------------------------------------------------------
    print("Step 1/7  Loading upstream Phase 2 and Phase 7 parameters…")
    phase2_params = _load_phase2_params(phase2_dir)
    phase7_params = _load_phase7_params(phase7_dir)
    print()

    # ------------------------------------------------------------------
    # Step 2: Generate synthetic data
    # ------------------------------------------------------------------
    print("Step 2/7  Generating synthetic data (seed={seed})…".format(seed=seed))
    synthetic_outputs = synthetic_data.generate_all(seed=seed, output_dir=output_dir)
    bearing_df    = synthetic_outputs["bearing_failures"]
    covariates_df = synthetic_outputs["covariates"]
    print(f"    Bearing failures: {len(bearing_df)} rows "
          f"({bearing_df['event_observed'].sum()} failures, "
          f"{(bearing_df['event_observed']==0).sum()} censored)")
    print(f"    Cox PH covariates: {len(covariates_df)} rows "
          f"({covariates_df['event_observed'].sum()} events, "
          f"{(covariates_df['event_observed']==0).sum()} censored)")
    print()

    # ------------------------------------------------------------------
    # Step 3: Bayesian Weibull updating
    # ------------------------------------------------------------------
    print("Step 3/7  Running Bayesian Weibull update (150×150 grid × 4 stages)…")
    t1 = time.time()
    bayes_result = bayesian_update.run(
        bearing_df=bearing_df,
        output_dir=output_dir,
        no_plots=no_plots,
    )
    stage3 = bayes_result["summary_df"][bayes_result["summary_df"]["update_stage"] == 3].iloc[0]
    print(f"    Stage 3 MAP: β̂={stage3['beta_map']:.4f}, η̂={stage3['eta_map_days']:.1f}d")
    print(f"    Updated T*={stage3['T_star_updated_years']:.1f} yr "
          f"(Phase 7: {config.BEARING_T_STAR_PHASE7/365.25:.0f} yr)")
    print(f"    Bayesian update completed in {time.time()-t1:.1f}s")
    print()

    # ------------------------------------------------------------------
    # Step 4: Cox Proportional Hazards
    # ------------------------------------------------------------------
    print("Step 4/7  Fitting Cox Proportional Hazards model (lifelines)…")
    t1 = time.time()
    try:
        from . import cox_survival
        cox_result = cox_survival.run(
            covariates_df=covariates_df,
            output_dir=output_dir,
            no_plots=no_plots,
        )
        gamma_est     = cox_result["gamma_estimated"]
        concordance   = cox_result["cph_result"]["concordance"]
        lr_p          = cox_result["survival_result"]["lr_p_value"]
        wind_q1       = cox_result["survival_result"]["wind_q1"]
        wind_q3       = cox_result["survival_result"]["wind_q3"]
        hazard_ratios = cox_result["cph_result"]["hazard_ratios"]
        print(f"    γ_estimated (vibration) = {gamma_est:.4f}")
        print(f"    Concordance index C = {concordance:.4f}")
        print(f"    Log-rank p (wind quartile) = {lr_p:.4f}")
        print(f"    Cox PH completed in {time.time()-t1:.1f}s")
        cox_available = True
    except ImportError:
        print("    [WARN] lifelines not installed — Cox PH step skipped.")
        print("           Install with: pip install lifelines")
        gamma_est     = float("nan")
        concordance   = float("nan")
        lr_p          = float("nan")
        wind_q1       = float("nan")
        wind_q3       = float("nan")
        hazard_ratios = pd.DataFrame()
        cox_available = False
    print()

    # ------------------------------------------------------------------
    # Step 5: CBM threshold optimisation
    # ------------------------------------------------------------------
    print("Step 5/7  Running CBM threshold optimisation (Monte Carlo sweep)…")
    t1 = time.time()
    cbm_result = cbm_optimizer.run(output_dir=output_dir, no_plots=no_plots)
    k_star = cbm_result["k_star_result"]
    print(f"    k* = {k_star['k_star']:.2f}σ (Cp/Cf=0.10)")
    print(f"    Cost reduction vs Phase 7 default (k=2.0): {k_star['reduction_pct']:.1f}%")
    print(f"    CBM optimisation completed in {time.time()-t1:.1f}s")
    print()

    # ------------------------------------------------------------------
    # Step 6: Generate report
    # ------------------------------------------------------------------
    print("Step 6/7  Generating Phase 8 Markdown report…")
    if not hazard_ratios.empty:
        report_text = reporting.generate_report(
            summary_df      = bayes_result["summary_df"],
            hazard_ratios   = hazard_ratios,
            gamma_estimated = gamma_est,
            concordance     = concordance,
            lr_p_value      = lr_p,
            k_star_result   = k_star,
            wind_q1         = wind_q1,
            wind_q3         = wind_q3,
            output_dir      = output_dir,
        )
    else:
        # Stub report if Cox skipped
        stub = (
            "# AEOLUS-RAMS Phase 8 Report\n\n"
            "Cox PH section skipped (lifelines not installed).\n"
            "Install lifelines and re-run for complete report.\n"
        )
        (output_dir / "phase8_report.md").write_text(stub)
        report_text = stub
    print(f"    Report written to {output_dir / 'phase8_report.md'}")
    print()

    # ------------------------------------------------------------------
    # Step 7: Summary
    # ------------------------------------------------------------------
    print("Step 7/7  Pipeline complete.")
    print()
    _separator()
    print("  OUTPUT FILES")
    _separator()
    for f in sorted(output_dir.glob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"    {f.name:<45}  {size_kb:6.1f} KB")
    _separator()
    elapsed = time.time() - t0
    print(f"\n  Total elapsed time: {elapsed:.1f}s")
    print()
    print("═" * 60)
    print("  PHASE 8 COMPLETE")
    print("═" * 60)
    print()

    return {
        "synthetic_outputs": synthetic_outputs,
        "bayes_result":      bayes_result,
        "cbm_result":        cbm_result,
        "cox_available":     cox_available,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AEOLUS-RAMS Phase 8 — Dynamic Reliability Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default relative paths (from phase-8-dynamic-modeling/):
  aeolus-rams-phase8

  # Specify directories explicitly:
  aeolus-rams-phase8 \\
      --phase7-dir ../phase-7-preventive-maintenance \\
      --phase2-dir ../phase-2-weibull-mtbf-hazard    \\
      --output-dir outputs/
        """,
    )
    parser.add_argument(
        "--phase7-dir", type=Path,
        default=Path("../phase-7-preventive-maintenance"),
        help="Path to phase-7-preventive-maintenance/ directory",
    )
    parser.add_argument(
        "--phase2-dir", type=Path,
        default=Path("../phase-2-weibull-mtbf-hazard"),
        help="Path to phase-2-weibull-mtbf-hazard/ directory",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs"),
        help="Directory to write all Phase 8 outputs (created if absent)",
    )
    parser.add_argument(
        "--seed", type=int, default=config.SYNTHETIC_SEED,
        help="Random seed for synthetic data generation (default: 42)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation (useful for CI / headless environments)",
    )

    args = parser.parse_args()

    run_phase8(
        phase7_dir = args.phase7_dir,
        phase2_dir = args.phase2_dir,
        output_dir = args.output_dir,
        seed       = args.seed,
        no_plots   = args.no_plots,
    )


if __name__ == "__main__":
    main()