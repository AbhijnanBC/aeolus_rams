"""
aeolus_rams_phase6.pipeline
============================
CLI orchestrator for Phase 6 ETA.

Usage
-----
    aeolus-rams-phase6 [--phase5-dir PATH] [--output-dir PATH] [--lambda-ie FLOAT]

Reads Phase 5 outputs, builds event tree, plots risk matrix and bow-tie,
quantifies ALARP options, and writes all deliverables to outputs/.

Exit code: 0 on success, 1 on any error (with traceback to stderr).
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as _cfg
from .event_tree import (
    load_gate_Q_values,
    load_basic_event_Q_values,
    load_ccf_uplift,
    build_event_tree,
    consequence_frequency_table,
)
from .risk_matrix import plot_risk_matrix, build_risk_points
from .bowtie import plot_bowtie
from .alarp import compute_all_options, alarp_table
from .reporting import generate_report


@dataclass
class Phase6Result:
    """Container for all Phase 6 pipeline outputs."""
    gate_Q:         dict[str, float]
    basic_event_Q:  dict[str, dict]
    ccf_uplift:     float
    branches:       list           # list[EventTreeBranch] at λ_IE central, independent
    cft:            pd.DataFrame   # consequence frequency table
    alarp_options:  list           # list[ALARPOption]
    plot_paths:     dict[str, str] = field(default_factory=dict)
    output_dir:     Path = Path("outputs")


def run_pipeline(
    phase5_dir: Path = _cfg.PHASE5_OUTPUT_DIR,
    output_dir: Path = _cfg.DEFAULT_OUTPUT_DIR,
    lambda_ie:  float = _cfg.LAMBDA_IE_CENTRAL,
) -> Phase6Result:
    """
    Run the complete Phase 6 pipeline.

    Parameters
    ----------
    phase5_dir : Path
        Directory containing Phase 5 output CSVs.
    output_dir : Path
        Directory to write Phase 6 outputs.
    lambda_ie : float
        Central initiating event frequency override (default: 0.5/yr).

    Returns
    -------
    Phase6Result
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load Phase 5 inputs ────────────────────────────────────────────
    gate_Q_path = phase5_dir / "gate_Q_table.csv"
    ccf_path    = phase5_dir / "ccf_sensitivity.csv"

    gate_Q        = load_gate_Q_values(gate_Q_path, t_filter="365d")
    basic_event_Q = load_basic_event_Q_values(gate_Q_path, t_filter="365d")
    ccf_uplift    = load_ccf_uplift(ccf_path, beta=_cfg.CCF_BETA_CENTRAL)

    print(f"[Phase 6] Loaded gate Q values from {gate_Q_path}")
    print(f"[Phase 6] Q_G1={gate_Q[_cfg.GATE_NAMES['G1']]:.5f}  "
          f"Q_G2={gate_Q[_cfg.GATE_NAMES['G2']]:.5f}  "
          f"Q_G3={gate_Q[_cfg.GATE_NAMES['G3']]:.5f}  "
          f"Q_top={gate_Q[_cfg.GATE_NAMES['TE']]:.4e}")
    print(f"[Phase 6] CCF uplift at β={_cfg.CCF_BETA_CENTRAL}: {ccf_uplift:.6f}×")

    # ── 2. Build event tree at central λ_IE ───────────────────────────────
    Q_G1 = gate_Q[_cfg.GATE_NAMES["G1"]]
    Q_G2 = gate_Q[_cfg.GATE_NAMES["G2"]]
    Q_G3 = gate_Q[_cfg.GATE_NAMES["G3"]]

    branches = build_event_tree(
        lambda_IE=lambda_ie,
        Q_G1=Q_G1, Q_G2=Q_G2, Q_G3=Q_G3,
        ccf_adjusted=False,
        ccf_uplift=ccf_uplift,
    )
    print(f"[Phase 6] Event tree built at λ_IE={lambda_ie}/yr")
    for b in branches:
        print(f"         {b.label:35s}  "
              f"P={b.P_branch_given_IE:.4e}  "
              f"λ={b.lambda_outcome_per_turbine:.3e}/yr  "
              f"[{b.risk_acceptability()}]")

    # ── 3. Consequence frequency table (full λ_IE sweep) ─────────────────
    cft = consequence_frequency_table(gate_Q, include_ccf=True, ccf_uplift=ccf_uplift)
    cft_path = output_dir / "consequence_frequency_table.csv"
    cft.to_csv(cft_path, index=False)
    print(f"[Phase 6] Consequence frequency table → {cft_path}")

    plot_paths: dict[str, str] = {}

    # ── 4. Risk matrix (central estimate, independent) ────────────────────
    risk_pts_central = build_risk_points(cft, lambda_IE=lambda_ie, ccf_adjusted=False)
    rm_path = output_dir / "risk_matrix.png"
    plot_risk_matrix(
        risk_pts_central, rm_path,
        title_suffix=f"λ_IE = {lambda_ie}/turbine/yr | Independent (β=0)",
    )
    plot_paths["risk_matrix"] = str(rm_path)
    print(f"[Phase 6] Risk matrix → {rm_path}")

    # ── 5. Risk matrix (CCF-adjusted) ────────────────────────────────────
    risk_pts_ccf = build_risk_points(cft, lambda_IE=lambda_ie, ccf_adjusted=True)
    rm_ccf_path = output_dir / "risk_matrix_ccf.png"
    plot_risk_matrix(
        risk_pts_ccf, rm_ccf_path,
        title_suffix=f"λ_IE = {lambda_ie}/turbine/yr | CCF-adjusted (β={_cfg.CCF_BETA_CENTRAL})",
    )
    plot_paths["risk_matrix_ccf"] = str(rm_ccf_path)
    print(f"[Phase 6] Risk matrix (CCF) → {rm_ccf_path}")

    # ── 6. ALARP quantification ───────────────────────────────────────────
    alarp_options = compute_all_options(gate_Q, lambda_IE=lambda_ie)
    alarp_df = alarp_table(alarp_options)
    alarp_path = output_dir / "alarp_table.csv"
    alarp_df.to_csv(alarp_path, index=False)
    plot_paths["alarp_table"] = str(alarp_path)
    print(f"[Phase 6] ALARP table → {alarp_path}")
    for opt in alarp_options:
        print(f"         {opt.name:50s}  "
              f"Q_top: {opt.Q_top_before:.3e} → {opt.Q_top_after:.3e}  "
              f"({opt.Q_top_reduction_pct:.1f}% reduction)  "
              f"[{opt.risk_acceptability_before} → {opt.risk_acceptability_after}]")

    # ── 7. Risk matrix with ALARP Option 1 applied ───────────────────────
    branches_opt1 = build_event_tree(
        lambda_IE=lambda_ie,
        Q_G1=Q_G1, Q_G2=Q_G2,
        Q_G3=alarp_options[0].Q_gate_after,
        ccf_adjusted=False,
    )
    # Inject updated branches into a temporary mini-DataFrame for build_risk_points
    _mini_rows = []
    for b in branches_opt1:
        _mini_rows.append({
            "lambda_IE": lambda_ie, "branch_label": b.label,
            "severity_category": b.severity_category,
            "P_branch": b.P_branch_given_IE,
            "lambda_per_turbine": b.lambda_outcome_per_turbine,
            "lambda_per_farm": b.lambda_outcome_per_farm,
            "return_period_yr": b.return_period_turbine_years,
            "P_one_plus_25yr_farm": b.P_one_or_more_in_design_life,
            "freq_category": b.frequency_category(),
            "risk_acceptability": b.risk_acceptability(),
            "ccf_adjusted": False,
        })
    _cft_opt1 = pd.DataFrame(_mini_rows)
    risk_pts_opt1 = build_risk_points(_cft_opt1, lambda_IE=lambda_ie, ccf_adjusted=False)
    rm_opt1_path = output_dir / "risk_matrix_alarp_option1.png"
    plot_risk_matrix(
        risk_pts_opt1, rm_opt1_path,
        title_suffix="After ALARP Option 1: Duplicate Safety Relay (G3b → SIL-2)",
    )
    plot_paths["risk_matrix_alarp_option1"] = str(rm_opt1_path)
    print(f"[Phase 6] Risk matrix (ALARP Option 1 applied) → {rm_opt1_path}")

    # ── 8. Bow-tie diagram ────────────────────────────────────────────────
    bowtie_path = output_dir / "bowtie.png"
    plot_bowtie(gate_Q, branches, bowtie_path, ccf_uplift=ccf_uplift)
    plot_paths["bowtie"] = str(bowtie_path)
    print(f"[Phase 6] Bow-tie diagram → {bowtie_path}")

    # ── 9. Phase report ───────────────────────────────────────────────────
    report_path = output_dir / "phase6_report.md"
    generate_report(
        gate_Q=gate_Q,
        branches=branches,
        alarp_options=alarp_options,
        ccf_uplift=ccf_uplift,
        cft=cft,
        output_path=report_path,
    )
    plot_paths["phase6_report"] = str(report_path)
    print(f"[Phase 6] Report → {report_path}")

    result = Phase6Result(
        gate_Q=gate_Q,
        basic_event_Q=basic_event_Q,
        ccf_uplift=ccf_uplift,
        branches=branches,
        cft=cft,
        alarp_options=alarp_options,
        plot_paths=plot_paths,
        output_dir=output_dir,
    )
    print("\nPHASE 6 COMPLETE")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="AEOLUS-RAMS Phase 6 — Event Tree Analysis (ETA)"
    )
    parser.add_argument(
        "--phase5-dir",
        type=Path,
        default=_cfg.PHASE5_OUTPUT_DIR,
        help="Path to Phase 5 outputs directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_cfg.DEFAULT_OUTPUT_DIR,
        help="Path to write Phase 6 outputs (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda-ie",
        type=float,
        default=_cfg.LAMBDA_IE_CENTRAL,
        help="Initiating event frequency /turbine/yr (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    try:
        run_pipeline(
            phase5_dir=args.phase5_dir,
            output_dir=args.output_dir,
            lambda_ie=args.lambda_ie,
        )
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    print(f"[Phase 6] Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()