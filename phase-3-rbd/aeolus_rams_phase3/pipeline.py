"""
aeolus_rams_phase3.pipeline
===============================
End-to-end Phase 3 orchestrator.

Flow
----
1. Load Phase 2 mtbf_table.csv
2. Inject Option A placeholders for 6 NaN-MTBF components
3. Build ComponentRT objects (all exponential, no NaN)
4. Compute R_turbine(t) series product
5. Compute farm-level k-of-N + BoP
6. Compute Birnbaum + Criticality importance
7. Run sensitivity analysis for placeholder MTBFs
8. Render turbine and farm RBD diagrams
9. Render sensitivity band plot
10. Generate Markdown report with live DoD checklist
11. Write all outputs to --output-dir

Run as a script:
    python -m aeolus_rams_phase3.pipeline \\
        --mtbf-table ../phase-2-weibull-mtbf-hazard/outputs/mtbf_table.csv \\
        --output-dir outputs/

or import and call run_phase3(...) directly from a notebook.
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
from .component_rt import (
    load_all_components, ComponentRT,
    reliability_table as comp_rt_table_fn,
    lambda_system, mtbf_system,
)
from .topology import ORDERED_COMPONENTS
from .turbine_rbd import (
    system_reliability_table,
    lambda_contributions,
)
from .farm_rbd import farm_system_table, R_kofN
from .importance import importance_table, rank_summary
from .sensitivity import aggregate_sensitivity, render_sensitivity_band
from .diagrams import render_turbine_rbd, render_farm_rbd
from .reporting import render_phase3_report

logger = logging.getLogger("aeolus_rams_phase3.pipeline")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class Phase3Result:
    components: dict[str, ComponentRT]
    component_rt_table: pd.DataFrame
    lambda_decomp: pd.DataFrame
    system_rt_table: pd.DataFrame
    farm_table: pd.DataFrame
    imp_table: pd.DataFrame
    imp_rank: pd.DataFrame
    agg_sensitivity: pd.DataFrame
    plot_paths: dict[str, str] = field(default_factory=dict)
    report_markdown: str = ""

    @property
    def R_turbine_1yr(self) -> float:
        row = self.system_rt_table[
            np.abs(self.system_rt_table["t_days"] - config.T_1YR) < 1.0
        ]
        return float(row["R_turbine"].iloc[0]) if not row.empty else float("nan")

    @property
    def R_turbine_5yr(self) -> float:
        row = self.system_rt_table[
            np.abs(self.system_rt_table["t_days"] - config.T_5YR) < 1.0
        ]
        return float(row["R_turbine"].iloc[0]) if not row.empty else float("nan")

    @property
    def lambda_sys(self) -> float:
        return lambda_system(self.components)

    @property
    def top_two_components(self) -> list[str]:
        ic_col = [c for c in self.imp_rank.columns if c.startswith("IC_")]
        if not ic_col:
            return []
        sort_col = ic_col[-1]
        return self.imp_rank.sort_values(sort_col, ascending=False)["component"].tolist()[:2]


# ---------------------------------------------------------------------------
# Core pipeline function
# ---------------------------------------------------------------------------

def run_phase3(
    mtbf_table_path: str | Path = config.DEFAULT_MTBF_TABLE_PATH,
    output_dir: str | Path | None = None,
    write_outputs: bool = True,
) -> Phase3Result:
    """
    Run the complete Phase 3 pipeline.

    Parameters
    ----------
    mtbf_table_path : path to Phase 2's mtbf_table.csv
    output_dir      : where to write all outputs; defaults to config.DEFAULT_OUTPUT_DIR
    write_outputs   : set False for in-memory only (e.g. inside tests)
    """
    mtbf_table_path = Path(mtbf_table_path)
    output_dir = Path(output_dir) if output_dir else config.DEFAULT_OUTPUT_DIR

    # Step 1–3: Load, inject, build
    logger.info("Loading Phase 2 mtbf_table.csv from %s", mtbf_table_path)
    components = load_all_components(mtbf_table_path)
    n_ph = sum(1 for c in components.values() if c.is_placeholder)
    logger.info(
        "Loaded %d components (%d with Option A placeholders)", len(components), n_ph
    )

    # Step 4: R_turbine(t) table
    logger.info("Computing per-component and system reliability tables")
    comp_rt_df = comp_rt_table_fn(components)
    lambda_decomp = lambda_contributions(components)
    system_rt = system_reliability_table(components)

    # Step 5: Farm-level
    logger.info("Computing farm k-of-N + BoP table")
    farm_tbl = farm_system_table(components)

    # Step 6: Importance
    logger.info("Computing Birnbaum + Criticality importance measures")
    imp_tbl = importance_table(components, t_values=(config.T_1YR, config.T_5YR))
    imp_rk = rank_summary(imp_tbl, t=config.T_5YR)

    # Step 7: Sensitivity
    logger.info("Running sensitivity analysis for %d placeholder components", n_ph)
    agg_sens = aggregate_sensitivity(components)

    plot_paths: dict[str, str] = {}

    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 8: Diagrams
        logger.info("Rendering RBD diagrams")
        R_5yr = float(np.exp(-lambda_system(components) * config.T_5YR))
        turb_path = render_turbine_rbd(components, output_dir / "turbine_rbd.png")
        farm_path = render_farm_rbd(
            output_dir / "farm_rbd.png",
            R_turbine_5yr=R_5yr,
        )
        plot_paths["turbine_rbd"] = str(turb_path)
        plot_paths["farm_rbd"] = str(farm_path)

        # Step 9: Sensitivity band plot
        logger.info("Rendering sensitivity band plot")
        sens_path = render_sensitivity_band(components, output_dir / "sensitivity_band.png")
        plot_paths["sensitivity_band"] = str(sens_path)

    # Step 10: Report
    logger.info("Generating Phase 3 Markdown report")
    report_md = render_phase3_report(
        components=components,
        component_rt_table=comp_rt_df,
        system_rt_table=system_rt,
        farm_table=farm_tbl,
        imp_table=imp_tbl,
        imp_rank=imp_rk,
        agg_sensitivity=agg_sens,
        plot_paths=plot_paths,
    )

    result = Phase3Result(
        components=components,
        component_rt_table=comp_rt_df,
        lambda_decomp=lambda_decomp,
        system_rt_table=system_rt,
        farm_table=farm_tbl,
        imp_table=imp_tbl,
        imp_rank=imp_rk,
        agg_sensitivity=agg_sens,
        plot_paths=plot_paths,
        report_markdown=report_md,
    )

    if write_outputs:
        _write_outputs(result, output_dir)

    return result


def _write_outputs(result: Phase3Result, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.component_rt_table.to_csv(output_dir / "component_rt_table.csv", index=False)
    result.lambda_decomp.to_csv(output_dir / "lambda_decomposition.csv", index=False)
    result.system_rt_table.to_csv(output_dir / "system_reliability_table.csv", index=False)
    result.farm_table.to_csv(output_dir / "farm_reliability_table.csv", index=False)
    result.imp_table.to_csv(output_dir / "importance_table.csv", index=False)
    result.imp_rank.to_csv(output_dir / "importance_rank.csv", index=False)
    result.agg_sensitivity.to_csv(output_dir / "sensitivity_aggregate.csv", index=False)
    (output_dir / "phase3_report.md").write_text(result.report_markdown, encoding="utf-8")
    for name, path in result.plot_paths.items():
        logger.info("Plot: %s → %s", name, path)
    logger.info("Wrote all Phase 3 outputs to %s", output_dir.resolve())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aeolus-rams-phase3",
        description=(
            "AEOLUS-RAMS Phase 3 — RBD: turbine series system, farm k-of-N, "
            "importance measures, and sensitivity analysis."
        ),
    )
    p.add_argument(
        "--mtbf-table", type=Path, default=config.DEFAULT_MTBF_TABLE_PATH,
        help=f"Path to Phase 2 mtbf_table.csv (default: {config.DEFAULT_MTBF_TABLE_PATH})",
    )
    p.add_argument(
        "--output-dir", type=Path, default=config.DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {config.DEFAULT_OUTPUT_DIR})",
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
        result = run_phase3(
            mtbf_table_path=args.mtbf_table,
            output_dir=args.output_dir,
        )
    except Exception:
        logger.exception("Phase 3 run failed")
        return 1

    lam = result.lambda_sys
    print()
    print("=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)
    print(f"  Components loaded        : {len(result.components)}"
          f" ({sum(c.is_placeholder for c in result.components.values())} placeholders)")
    print(f"  λ_system                 : {lam:.6f} /day  ({lam*365.25:.3f} /yr)")
    print(f"  MTBF_system              : {1/lam:.0f} days ({1/lam/365.25:.2f} yr)")
    print(f"  R_turbine(1yr)           : {result.R_turbine_1yr:.4f}")
    print(f"  R_turbine(5yr)           : {result.R_turbine_5yr:.4f}")
    print(f"  Top 2 IC components      : {result.top_two_components}")
    print(f"  Outputs written to       : {args.output_dir.resolve()}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
