"""
aeolus_rams_phase5.pipeline
================================
End-to-end Phase 5 orchestrator.

Flow
----
 1. Load Q values from Phase 3 component_rt_table.csv
 2. Build fault tree at t=1yr and t=5yr
 3. Sanity-check Q_top against pre-computed expected value
 4. Extract all 12 MCS and compute MCS table
 5. Compute importance measures (IB, CIM, FV) at 1yr and 5yr
 6. Compute gate-level importance table
 7. Run CCF β-factor sensitivity sweep
 8. Run sub-cause fraction sensitivity check
 9. Run probabilistic FTA (MTBF uncertainty propagation)
10. Generate fault tree diagram
11. Generate importance figure
12. Generate CCF sensitivity figure
13. Generate Q_top uncertainty figure
14. Generate Phase 5 Markdown report
15. Write all outputs

Run as script:
    python -m aeolus_rams_phase5.pipeline --output-dir outputs/

or import and call run_phase5(...) from a notebook.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as _cfg
from .fault_tree import build_overspeed_fault_tree, gate_Q_table
from .mcs import mcs_table, extract_minimal_cut_sets
from .importance import compute_importance_measures, gate_importance_table
from .ccf import ccf_sensitivity_sweep, apply_beta_factor_ccf, subcause_fraction_sensitivity
from .probabilistic import probabilistic_fta
from .diagrams import render_fault_tree
from .reporting import render_phase5_report

logger = logging.getLogger("aeolus_rams_phase5.pipeline")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class Phase5Result:
    Q_1yr: dict[str, float]
    Q_5yr: dict[str, float]
    gate_Q_1yr: pd.DataFrame
    gate_Q_5yr: pd.DataFrame
    mcs_df_1yr: pd.DataFrame
    mcs_df_5yr: pd.DataFrame
    importance_df_1yr: pd.DataFrame
    importance_df_5yr: pd.DataFrame
    gate_importance_df: pd.DataFrame
    ccf_sweep_df: pd.DataFrame
    subcause_sensitivity_df: pd.DataFrame
    Q_top_samples: np.ndarray
    prob_fta_summary: pd.DataFrame
    plot_paths: dict[str, str] = field(default_factory=dict)
    report_markdown: str = ""
    bridge_check_passed: bool = False
    Q_top_1yr: float = 0.0
    Q_top_5yr: float = 0.0


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def _sanity_check_Q_top(Q_top: float) -> tuple[bool, str]:
    """Verify computed Q_top against pre-calculated expected value.

    Expected: ~7.278×10⁻⁴ (verified by hand before any code was written).
    Tolerance: 5% (allows for float arithmetic differences).
    """
    expected = _cfg.EXPECTED_Q_TOP_1YR
    tol = _cfg.EXPECTED_Q_TOP_TOLERANCE
    error = abs(Q_top - expected) / expected if expected > 0 else float("inf")
    passed = error < tol
    detail = (
        f"computed={Q_top:.6e}, expected≈{expected:.6e}, "
        f"error={error*100:.2f}% (tolerance {tol*100:.0f}%)"
    )
    return passed, detail


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_importance(
    imp_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Figure: horizontal bar chart of FV importance per basic event."""
    fig, ax = plt.subplots(figsize=(9, max(4, len(imp_df) * 0.5 + 1.5)))
    colors = ["#2196F3" if "G1" in be else "#4CAF50" if "G2" in be else "#FF9800"
              for be in imp_df["basic_event"]]
    ax.barh(range(len(imp_df)), imp_df["FV"].values[::-1],
            color=colors[::-1], alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(imp_df)))
    ax.set_yticklabels([be.split(":")[0].strip() for be in imp_df["basic_event"].values[::-1]],
                       fontsize=9)
    ax.set_xlabel("Fussell-Vesely Importance (FV)")
    ax.set_title(
        "AEOLUS-RAMS Phase 5 — Basic Event Importance (Fussell-Vesely)\n"
        "Blue: G1 (Pitch)  Green: G2 (Brake)  Orange: G3 (SCADA)"
    )
    # Value labels
    for i, (fv, be) in enumerate(zip(imp_df["FV"].values[::-1], imp_df["basic_event"].values[::-1])):
        ax.text(fv + 0.005, i, f"{fv:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_ccf_sensitivity(ccf_df: pd.DataFrame, output_path: Path) -> Path:
    """Figure: Q_top vs β_G1G2 showing CCF uplift."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Q_top vs beta
    ax1.plot(ccf_df["beta_G1G2"], ccf_df["Q_top_independent"],
             "b--", linewidth=1.5, label="Independent (β=0)", alpha=0.7)
    ax1.plot(ccf_df["beta_G1G2"], ccf_df["Q_top_ccf_adjusted"],
             "r-", linewidth=2.0, label="CCF-adjusted")
    ax1.axvline(_cfg.CCF_BETA_CENTRAL, color="gray", linestyle=":", alpha=0.6,
                label=f"Central β={_cfg.CCF_BETA_CENTRAL}")
    ax1.set_xlabel("β_G1G2 (common-cause fraction)")
    ax1.set_ylabel("Q_top (per turbine per year)")
    ax1.set_title("Q_top vs CCF β-factor")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Right: return period vs beta
    ax2.plot(ccf_df["beta_G1G2"], ccf_df["return_period_ccf"], "r-", linewidth=2.0)
    ax2.axvline(_cfg.CCF_BETA_CENTRAL, color="gray", linestyle=":", alpha=0.6)
    ax2.set_xlabel("β_G1G2")
    ax2.set_ylabel("Return period (turbine-years)")
    ax2.set_title("Return Period vs CCF β-factor")
    ax2.grid(alpha=0.3)

    fig.suptitle("AEOLUS-RAMS Phase 5 — CCF Sensitivity Analysis", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_Q_top_uncertainty(Q_top_samples: np.ndarray, output_path: Path) -> Path:
    """Figure: histogram of Q_top uncertainty from probabilistic FTA."""
    fig, ax = plt.subplots(figsize=(8, 5))
    p5 = np.percentile(Q_top_samples, 5)
    p95 = np.percentile(Q_top_samples, 95)
    mean_Q = Q_top_samples.mean()

    n_bins = min(80, int(np.sqrt(len(Q_top_samples))) * 2)
    counts, edges, patches = ax.hist(
        Q_top_samples, bins=n_bins, color="#5C6BC0", alpha=0.7,
        edgecolor="white", linewidth=0.3,
    )
    # Shade 5th–95th band
    for patch, left, right in zip(patches, edges[:-1], edges[1:]):
        if p5 <= (left + right) / 2 <= p95:
            patch.set_facecolor("#9FA8DA")
            patch.set_alpha(0.85)

    ax.axvline(mean_Q, color="#1A237E", linewidth=2, linestyle="-",
               label=f"Mean Q_top = {mean_Q:.3e}")
    ax.axvline(p5, color="#F44336", linewidth=1.5, linestyle="--",
               label=f"P5 = {p5:.3e}")
    ax.axvline(p95, color="#F44336", linewidth=1.5, linestyle="--",
               label=f"P95 = {p95:.3e}")

    ax.set_xlabel("Q_top (probability of overspeed per turbine per year)")
    ax.set_ylabel("Frequency (samples)")
    ax.set_title(
        "AEOLUS-RAMS Phase 5 — Probabilistic FTA: Q_top Uncertainty\n"
        f"(n={len(Q_top_samples):,} samples, shading = 90% CI)"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.text(
        0.97, 0.95,
        f"Mean return period: {1/mean_Q:.0f} yr\n"
        f"P5 return period:   {1/p95:.0f} yr\n"
        f"P95 return period:  {1/p5:.0f} yr",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Core pipeline function
# ---------------------------------------------------------------------------

def run_phase5(
    rt_table_path: str | Path = _cfg.PHASE3_RT_TABLE,
    output_dir: str | Path | None = None,
    write_outputs: bool = True,
    run_probabilistic: bool = True,
) -> Phase5Result:
    """Run the complete Phase 5 FTA pipeline.

    Parameters
    ----------
    rt_table_path : Path
        Path to Phase 3's component_rt_table.csv.
    output_dir : Path, optional
        Where to write outputs.
    write_outputs : bool
        False for in-memory only (tests).
    run_probabilistic : bool
        False to skip the 50,000-sample probabilistic FTA (saves ~2 min).
    """
    rt_table_path = Path(rt_table_path)
    output_dir = Path(output_dir) if output_dir else _cfg.DEFAULT_OUTPUT_DIR
    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load Q values ─────────────────────────────────────────────
    try:
        Q_1yr, Q_5yr = _cfg.load_Q_values(rt_table_path)
        logger.info("Loaded Q values from %s", rt_table_path)
    except FileNotFoundError:
        logger.warning(
            "Phase 3 RT table not found at %s — using fallback values.", rt_table_path
        )
        Q_1yr = _cfg.FALLBACK_Q_1YR.copy()
        Q_5yr = _cfg.FALLBACK_Q_5YR.copy()

    # ── Step 2: Build fault trees ─────────────────────────────────────────
    logger.info("Building fault trees...")
    top_1yr = build_overspeed_fault_tree(Q_1yr)
    top_5yr = build_overspeed_fault_tree(Q_5yr)
    Q_top_1yr = top_1yr.Q
    Q_top_5yr = top_5yr.Q
    logger.info("Q_top(1yr) = %.4e, Q_top(5yr) = %.4e", Q_top_1yr, Q_top_5yr)

    # ── Step 3: Sanity check ──────────────────────────────────────────────
    bridge_passed, bridge_detail = _sanity_check_Q_top(Q_top_1yr)
    status = "PASSED ✓" if bridge_passed else "FAILED ✗"
    logger.info("Phase 3 bridge check: %s — %s", status, bridge_detail)

    # ── Step 4: Gate Q tables ─────────────────────────────────────────────
    gq_1yr = gate_Q_table(top_1yr, "365d")
    gq_5yr = gate_Q_table(top_5yr, "1825d")

    # ── Step 5: MCS extraction ────────────────────────────────────────────
    logger.info("Extracting minimal cut sets...")
    mcs_1yr = mcs_table(top_1yr, "365d")
    mcs_5yr = mcs_table(top_5yr, "1825d")
    logger.info("Found %d MCS (all order 3)", len(extract_minimal_cut_sets(top_1yr)))

    # ── Step 6: Importance measures ───────────────────────────────────────
    logger.info("Computing importance measures...")
    imp_1yr = compute_importance_measures(top_1yr, "365d")
    imp_5yr = compute_importance_measures(top_5yr, "1825d")
    gi_df   = gate_importance_table(top_1yr, "365d")

    # ── Step 7: CCF sensitivity sweep ────────────────────────────────────
    logger.info("Running CCF sensitivity sweep...")
    ccf_df = ccf_sensitivity_sweep(top_1yr, Q_1yr)

    # ── Step 8: Sub-cause fraction sensitivity ────────────────────────────
    logger.info("Running sub-cause fraction sensitivity check...")
    # Variants: central ±50% on fractions
    g1_variants = [(0.45, 0.35, 0.20), (0.225, 0.175, 0.10), (0.675, 0.525, 0.30)]
    # Normalise each set
    g1_variants = [tuple(f/sum(v) for f in v) for v in g1_variants]
    g2_variants = [(0.60, 0.40), (0.30, 0.20), (0.90, 0.60)]
    g2_variants = [tuple(f/sum(v) for f in v) for v in g2_variants]
    g3_variants = [(0.60, 0.40), (0.30, 0.20), (0.90, 0.60)]
    g3_variants = [tuple(f/sum(v) for f in v) for v in g3_variants]
    subcase_sens_df = subcause_fraction_sensitivity(Q_1yr, g1_variants, g2_variants, g3_variants)

    # ── Step 9: Probabilistic FTA ─────────────────────────────────────────
    Q_top_samples = np.array([Q_top_1yr])  # placeholder
    prob_summary = pd.DataFrame()
    if run_probabilistic:
        logger.info("Running probabilistic FTA (%d samples)...", _cfg.N_PROBABILISTIC_SAMPLES)
        Q_top_samples, prob_summary = probabilistic_fta(
            t=365.25,
            n_samples=_cfg.N_PROBABILISTIC_SAMPLES,
            seed=_cfg.PROBABILISTIC_SEED,
        )
        logger.info(
            "Q_top 95%% CI: [%.3e, %.3e]",
            np.percentile(Q_top_samples, 5), np.percentile(Q_top_samples, 95)
        )

    # ── Steps 10–13: Plots ────────────────────────────────────────────────
    plot_paths: dict[str, str] = {}
    if write_outputs:
        logger.info("Generating figures...")

        ft_path = render_fault_tree(top_1yr, output_dir / "fault_tree.png")
        plot_paths["fault_tree"] = str(ft_path)

        imp_path = _plot_importance(imp_1yr, output_dir / "importance_chart.png")
        plot_paths["importance"] = str(imp_path)

        ccf_path = _plot_ccf_sensitivity(ccf_df, output_dir / "ccf_sensitivity.png")
        plot_paths["ccf_sensitivity"] = str(ccf_path)

        if run_probabilistic and len(Q_top_samples) > 1:
            unc_path = _plot_Q_top_uncertainty(
                Q_top_samples, output_dir / "Q_top_uncertainty.png"
            )
            plot_paths["Q_top_uncertainty"] = str(unc_path)

    # ── Step 14: Report ───────────────────────────────────────────────────
    logger.info("Generating Phase 5 report...")
    report_md = render_phase5_report(
        top_gate_1yr=top_1yr,
        top_gate_5yr=top_5yr,
        Q_1yr=Q_1yr,
        Q_5yr=Q_5yr,
        ccf_sweep_df=ccf_df,
        prob_fta_summary=prob_summary,
        plot_paths=plot_paths,
        bridge_check_passed=bridge_passed,
    )

    result = Phase5Result(
        Q_1yr=Q_1yr, Q_5yr=Q_5yr,
        gate_Q_1yr=gq_1yr, gate_Q_5yr=gq_5yr,
        mcs_df_1yr=mcs_1yr, mcs_df_5yr=mcs_5yr,
        importance_df_1yr=imp_1yr, importance_df_5yr=imp_5yr,
        gate_importance_df=gi_df,
        ccf_sweep_df=ccf_df,
        subcause_sensitivity_df=subcase_sens_df,
        Q_top_samples=Q_top_samples,
        prob_fta_summary=prob_summary,
        plot_paths=plot_paths,
        report_markdown=report_md,
        bridge_check_passed=bridge_passed,
        Q_top_1yr=Q_top_1yr,
        Q_top_5yr=Q_top_5yr,
    )

    if write_outputs:
        _write_outputs(result, output_dir)

    return result


def _write_outputs(result: Phase5Result, output_dir: Path) -> None:
    """Write all CSV and report outputs to output_dir."""
    pd.concat([result.gate_Q_1yr, result.gate_Q_5yr]).to_csv(
        output_dir / "gate_Q_table.csv", index=False
    )
    pd.concat([result.mcs_df_1yr, result.mcs_df_5yr]).to_csv(
        output_dir / "mcs_table.csv", index=False
    )
    pd.concat([result.importance_df_1yr, result.importance_df_5yr]).to_csv(
        output_dir / "importance_table.csv", index=False
    )
    result.gate_importance_df.to_csv(
        output_dir / "gate_importance_table.csv", index=False
    )
    result.ccf_sweep_df.to_csv(output_dir / "ccf_sensitivity.csv", index=False)
    result.subcause_sensitivity_df.to_csv(
        output_dir / "subcause_sensitivity.csv", index=False
    )
    if not result.prob_fta_summary.empty:
        result.prob_fta_summary.to_csv(
            output_dir / "probabilistic_fta_summary.csv", index=False
        )
    (output_dir / "phase5_report.md").write_text(
        result.report_markdown, encoding="utf-8"
    )
    logger.info("Wrote all Phase 5 outputs to %s", output_dir.resolve())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aeolus-rams-phase5",
        description=(
            "AEOLUS-RAMS Phase 5 — FTA: turbine overspeed catastrophic failure, "
            "MCS, importance measures, CCF analysis."
        ),
    )
    p.add_argument(
        "--rt-table", type=Path, default=_cfg.PHASE3_RT_TABLE,
        help=f"Path to Phase 3 component_rt_table.csv (default: {_cfg.PHASE3_RT_TABLE})",
    )
    p.add_argument(
        "--output-dir", type=Path, default=_cfg.DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {_cfg.DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--skip-probabilistic", action="store_true",
        help="Skip 50,000-sample probabilistic FTA (saves ~2 min).",
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
        result = run_phase5(
            rt_table_path=args.rt_table,
            output_dir=args.output_dir,
            run_probabilistic=not args.skip_probabilistic,
        )
    except Exception:
        logger.exception("Phase 5 run failed")
        return 1

    print()
    print("=" * 70)
    print("PHASE 5 COMPLETE")
    print("=" * 70)
    print(f"  Q_top (1yr, independent)  : {result.Q_top_1yr:.4e}")
    print(f"  Q_top (5yr, independent)  : {result.Q_top_5yr:.4e}")
    print(f"  Return period (1yr)       : {1/result.Q_top_1yr:.0f} turbine-years")
    print(f"  Farm 25yr P(≥1 event)     : {1-(1-result.Q_top_1yr)**(22*25):.1%}")
    bridge_status = "PASSED ✓" if result.bridge_check_passed else "FAILED ✗"
    print(f"  Phase 3 bridge check      : {bridge_status}")
    ccf_central = result.ccf_sweep_df[
        np.abs(result.ccf_sweep_df["beta_G1G2"] - _cfg.CCF_BETA_CENTRAL) < 0.005
    ]
    if not ccf_central.empty:
        uplift = float(ccf_central["ccf_risk_uplift"].iloc[0])
        print(f"  CCF uplift (β={_cfg.CCF_BETA_CENTRAL})       : {uplift:.2f}×")
    print(f"  MCS count                 : {len(result.mcs_df_1yr)}")
    print(f"  Outputs written to        : {args.output_dir.resolve()}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())