"""
aeolus_rams_phase7.pipeline
================================
End-to-end Phase 7 orchestrator.

Flow
----
 1. Load Phase 5 Q_G3b from gate_Q_table.csv (or fallback)
 2. Derive λ_G3b from Phase 5 Q_G3b + τ_implied
 3. Compute τ_SIL2 for SIL-2 (Phase 6 ALARP Option 1 follow-through)
 4. Run Barlow-Proschan optimisation for Bearing, Gearbox, Generator
 5. Run Cp/Cf sensitivity sweep for all AR components
 6. Build failure-finding table (PFDavg vs. τ for G3b)
 7. Build master RCM schedule (all 13 components)
 8. Generate Figure 1: cost-rate curves
 9. Generate Figure 2: PFDavg vs. τ
10. Generate Figure 3: RCM summary chart
11. Generate Figure 4: T* sensitivity
12. Generate Phase 7 Markdown report
13. Write all outputs to --output-dir
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as _cfg
from .age_replacement import (
    optimal_replacement_age, cost_ratio_sensitivity, AgeReplacementResult,
)
from .failure_finding import (
    proof_test_interval_table, required_proof_test_interval,
    PFDavg_exact, derive_lambda_from_Q,
)
from .rcm_schedule import (
    build_rcm_schedule, build_age_replacement_table,
    build_cost_ratio_sensitivity_table, build_failure_finding_table,
)
from .plots import (
    plot_cost_rate_curves, plot_pfd_curve,
    plot_rcm_summary_chart, plot_cost_ratio_sensitivity,
)
from .reporting import render_phase7_report

logger = logging.getLogger("aeolus_rams_phase7.pipeline")


@dataclass
class Phase7Result:
    rcm_df: pd.DataFrame
    ar_table: pd.DataFrame
    ff_table: pd.DataFrame
    sensitivity_df: pd.DataFrame
    plot_paths: dict[str, str] = field(default_factory=dict)
    report_markdown: str = ""
    tau_sil2_days: float = 0.0
    lambda_G3b: float = 0.0
    pfd_current: float = 0.0


def _load_Q_G3b(gate_q_path: Path) -> float:
    """Load Q_G3b from Phase 5 gate_Q_table.csv, or return fallback."""
    try:
        df = pd.read_csv(gate_q_path)
        # Filter: basic event G3b at t=365d
        row = df[
            df["node"].str.contains("G3b", na=False) & (df["t"] == "365d")
        ]
        if not row.empty:
            val = float(row["Q"].iloc[0])
            logger.info("Loaded Q_G3b = %.6f from %s", val, gate_q_path)
            return val
    except Exception as exc:
        logger.warning("Could not load gate_Q_table.csv: %s — using fallback", exc)
    logger.info("Using fallback Q_G3b = %.6f", _cfg.PHASE5_Q_G3b)
    return _cfg.PHASE5_Q_G3b


def run_phase7(
    gate_q_path: Path | None = None,
    output_dir: Path | None = None,
    write_outputs: bool = True,
) -> Phase7Result:
    """Run the complete Phase 7 pipeline.

    Parameters
    ----------
    gate_q_path : Path, optional
        Path to Phase 5 gate_Q_table.csv. Defaults to config.PHASE5_GATE_Q.
    output_dir : Path, optional
        Output directory. Defaults to config.DEFAULT_OUTPUT_DIR.
    write_outputs : bool
        False for in-memory only (tests).
    """
    if gate_q_path is None:
        gate_q_path = _cfg.PHASE5_GATE_Q
    if output_dir is None:
        output_dir = _cfg.DEFAULT_OUTPUT_DIR
    output_dir = Path(output_dir)
    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1–2: Load Q_G3b and derive λ_G3b ────────────────────────────
    Q_G3b = _load_Q_G3b(gate_q_path)
    # Verify against config-derived LAMBDA_G3b
    lam_G3b = _cfg.LAMBDA_G3b
    pfd_verify = PFDavg_exact(lam_G3b, _cfg.TAU_IMPLIED_G3b_DAYS)
    logger.info(
        "λ_G3b = %.4e/d | PFDavg_verify(λ,730d) = %.6f | Q_G3b = %.6f | match: %s",
        lam_G3b, pfd_verify, Q_G3b, abs(pfd_verify - Q_G3b) < 1e-5,
    )

    # ── Step 3: Compute τ_SIL2 ───────────────────────────────────────────
    tau_sil2 = required_proof_test_interval(lam_G3b, _cfg.SIL2_PFDavg_TARGET)
    pfd_current = PFDavg_exact(lam_G3b, _cfg.TAU_IMPLIED_G3b_DAYS)
    logger.info(
        "τ_SIL2 = %.1f days (%.1f months) | current PFDavg = %.4f",
        tau_sil2, tau_sil2 / 30.44, pfd_current,
    )

    # ── Step 4: Barlow-Proschan optimisation ─────────────────────────────
    logger.info("Running age-replacement optimisation for Bearing, Gearbox, Generator...")
    ar_table = build_age_replacement_table()
    for _, row in ar_table.iterrows():
        if row.get("T_star_years") and not pd.isna(row["T_star_years"]):
            logger.info(
                "  %s: T* = %.1f yr, savings = %.1f%%",
                row["component"], row["T_star_years"], row["savings_fraction"] * 100,
            )
        else:
            logger.info("  %s: T* > design life or β≤1", row["component"])

    # ── Step 5: Cp/Cf sensitivity sweep ──────────────────────────────────
    logger.info("Running cost-ratio sensitivity sweep...")
    sensitivity_df = build_cost_ratio_sensitivity_table()

    # ── Step 6: Failure-finding table ────────────────────────────────────
    logger.info("Building failure-finding table (G3b PFDavg vs. τ)...")
    ff_table = build_failure_finding_table()

    # ── Step 7: RCM schedule ─────────────────────────────────────────────
    logger.info("Building master RCM schedule...")
    rcm_df = build_rcm_schedule(Q_G3b=Q_G3b)
    logger.info("RCM schedule: %d components", len(rcm_df))

    # ── Steps 8–11: Plots ─────────────────────────────────────────────────
    plot_paths: dict[str, str] = {}
    if write_outputs:
        logger.info("Generating figures...")

        p1 = plot_cost_rate_curves(output_dir / "cost_rate_curves.png")
        plot_paths["cost_rate_curves"] = str(p1)

        p2 = plot_pfd_curve(output_dir / "pfd_curve.png")
        plot_paths["pfd_curve"] = str(p2)

        p3 = plot_rcm_summary_chart(rcm_df, output_dir / "rcm_summary_chart.png")
        plot_paths["rcm_summary_chart"] = str(p3)

        if not sensitivity_df.empty:
            p4 = plot_cost_ratio_sensitivity(sensitivity_df, output_dir / "cost_ratio_sensitivity.png")
            plot_paths["cost_ratio_sensitivity"] = str(p4)

    # ── Step 12: Report ───────────────────────────────────────────────────
    logger.info("Generating Phase 7 Markdown report...")
    report_md = render_phase7_report(
        rcm_df=rcm_df,
        ar_table=ar_table,
        ff_table=ff_table,
        sensitivity_df=sensitivity_df,
        plot_paths=plot_paths,
    )

    result = Phase7Result(
        rcm_df=rcm_df,
        ar_table=ar_table,
        ff_table=ff_table,
        sensitivity_df=sensitivity_df,
        plot_paths=plot_paths,
        report_markdown=report_md,
        tau_sil2_days=tau_sil2,
        lambda_G3b=lam_G3b,
        pfd_current=pfd_current,
    )

    if write_outputs:
        _write_outputs(result, output_dir)

    return result


def _write_outputs(result: Phase7Result, output_dir: Path) -> None:
    result.rcm_df.to_csv(output_dir / "rcm_schedule.csv", index=False)
    result.ar_table.to_csv(output_dir / "age_replacement_table.csv", index=False)
    result.ff_table.to_csv(output_dir / "failure_finding_table.csv", index=False)
    if not result.sensitivity_df.empty:
        result.sensitivity_df.to_csv(output_dir / "cost_ratio_sensitivity.csv", index=False)
    (output_dir / "phase7_report.md").write_text(result.report_markdown, encoding="utf-8")
    logger.info("All Phase 7 outputs written to %s", output_dir.resolve())


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aeolus-rams-phase7",
        description=(
            "AEOLUS-RAMS Phase 7 — RCM: Barlow-Proschan age-replacement, "
            "IEC 61511 failure-finding intervals, CBM trigger for Pitch System."
        ),
    )
    p.add_argument(
        "--gate-q-table", type=Path, default=_cfg.PHASE5_GATE_Q,
        help=f"Path to Phase 5 gate_Q_table.csv (default: {_cfg.PHASE5_GATE_Q})",
    )
    p.add_argument(
        "--output-dir", type=Path, default=_cfg.DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {_cfg.DEFAULT_OUTPUT_DIR})",
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
        result = run_phase7(
            gate_q_path=args.gate_q_table,
            output_dir=args.output_dir,
        )
    except Exception:
        logger.exception("Phase 7 run failed")
        return 1

    print()
    print("=" * 72)
    print("PHASE 7 COMPLETE")
    print("=" * 72)
    ar_bearing = result.ar_table[result.ar_table["component"] == "Main/Rotor Bearing"]
    ar_gearbox = result.ar_table[result.ar_table["component"] == "Gearbox"]
    if not ar_bearing.empty and not pd.isna(ar_bearing.iloc[0].get("T_star_years")):
        b_row = ar_bearing.iloc[0]
        g_row = ar_gearbox.iloc[0]
        print(f"  Bearing T*        : {b_row['T_star_years']:.1f} yr  (savings {b_row['savings_fraction']*100:.1f}%)")
        print(f"  Gearbox T*        : {g_row['T_star_years']:.1f} yr  (savings {g_row['savings_fraction']*100:.1f}%)")
    print(f"  Generator T*      : >25yr design life → CBM recommended")
    print(f"  Pitch System      : β={_cfg.PITCH_BETA:.4f}<1 → CBM (ML anomaly score trigger)")
    print(f"  G3b λ_derived     : {result.lambda_G3b:.4e} /day  (from Q_G3b={_cfg.PHASE5_Q_G3b:.5f})")
    print(f"  G3b current PFD   : {result.pfd_current:.5f}  (<SIL-1 without proof-testing)")
    print(f"  τ_SIL2 required   : {result.tau_sil2_days:.1f} days ({result.tau_sil2_days/30.44:.1f} months)")
    print(f"  Outputs           : {args.output_dir.resolve()}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())