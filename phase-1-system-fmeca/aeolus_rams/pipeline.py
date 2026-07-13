"""
aeolus_rams.pipeline
=======================
End-to-end Phase 1 orchestrator. Wires together:

    data_loader  ->  validation (Step 0)  ->  component_tagger  ->  fmeca
                                                                       |
                                                                       v
                                                                  reporting

Run as a script:

    python -m aeolus_rams.pipeline --data-root data/raw/care \
        --output-dir phase-1-system-fmeca

Or import and call `run_phase1(...)` directly from a notebook.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config
from .data_loader import discover_farms, load_all_event_tables, inventory_summary
from .validation import run_step0_checks, Step0Report
from .component_tagger import tag_events
from .fmeca import (
    build_fmeca_table,
    unclassified_events,
    SEVERITY_DETECTION_DEFAULTS,
    ComponentScoreDefaults,
)
from .reporting import render_phase1_report

logger = logging.getLogger("aeolus_rams.pipeline")


@dataclass
class Phase1Result:
    inventory: pd.DataFrame
    step0: Step0Report
    tagged_events: pd.DataFrame
    fmeca_table: pd.DataFrame
    unclassified: pd.DataFrame
    report_markdown: str


def run_phase1(
    data_root: str | Path,
    output_dir: str | Path | None = None,
    sample_files_per_farm: int | None = 3,
    scores: dict[str, ComponentScoreDefaults] | None = None,
    write_outputs: bool = True,
) -> Phase1Result:
    """Run the complete Phase 1 pipeline and (by default) write the
    Section 1.9 deliverables to `output_dir`.

    Parameters
    ----------
    data_root : folder containing your unzipped CARE download.
    output_dir : where to write tagged_events.csv, fmeca_table.csv,
        phase1_report.md. Defaults to config.DEFAULT_OUTPUT_DIR.
    sample_files_per_farm : dataset files scanned per farm for the
        status-legend cross-check (Section 1.1.1). Pass None for an
        exhaustive scan across every turbine file (slower, authoritative).
    scores : optional Severity/Detection override map — see
        `fmeca.with_custom_scores`. Defaults to the Section 1.6 worked
        table.
    write_outputs : set False to run in-memory only (e.g. inside a test).
    """
    scores = scores or SEVERITY_DETECTION_DEFAULTS
    output_dir = Path(output_dir) if output_dir is not None else config.DEFAULT_OUTPUT_DIR

    logger.info("Discovering CARE farms under %s", data_root)
    farm_paths = discover_farms(data_root)

    logger.info("Loading event_info.csv for %d farm(s)", len(farm_paths))
    event_tables = load_all_event_tables(farm_paths)

    logger.info("Running Step 0 validation checks")
    step0 = run_step0_checks(farm_paths, event_tables, sample_files_per_farm)

    logger.info("Concatenating events across farms")
    events_master = pd.concat(event_tables.values(), ignore_index=True)

    logger.info("Tagging %d event(s) with failure-mode components", len(events_master))
    tagged_events = tag_events(events_master)

    logger.info("Assembling FMECA table")
    fmeca_table = build_fmeca_table(tagged_events, scores=scores)

    review_queue = unclassified_events(tagged_events)
    if not review_queue.empty:
        logger.warning(
            "%d event(s) need manual review before trusting Occurrence "
            "counts (see 'needs_manual_review' rows in tagged_events.csv)",
            len(review_queue),
        )

    report_md = render_phase1_report(
        inventory=step0.inventory,
        step0=step0,
        tagged_events=tagged_events,
        fmeca_table=fmeca_table,
        unclassified=review_queue,
    )

    result = Phase1Result(
        inventory=step0.inventory,
        step0=step0,
        tagged_events=tagged_events,
        fmeca_table=fmeca_table,
        unclassified=review_queue,
        report_markdown=report_md,
    )

    if write_outputs:
        _write_outputs(result, output_dir)

    return result


def _write_outputs(result: Phase1Result, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    tagged_path = output_dir / "tagged_events.csv"
    fmeca_path = output_dir / "fmeca_table.csv"
    report_path = output_dir / "phase1_report.md"
    inventory_path = output_dir / "inventory_summary.csv"

    result.tagged_events.to_csv(tagged_path, index=False)
    result.fmeca_table.to_csv(fmeca_path, index=False)
    result.inventory.to_csv(inventory_path, index=False)
    report_path.write_text(result.report_markdown, encoding="utf-8")

    logger.info("Wrote %s", tagged_path)
    logger.info("Wrote %s", fmeca_path)
    logger.info("Wrote %s", inventory_path)
    logger.info("Wrote %s", report_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aeolus-rams-phase1",
        description="AEOLUS-RAMS Phase 1 — System Definition + FMECA over "
                     "a CARE-to-Compare download.",
    )
    parser.add_argument(
        "--data-root", type=Path, default=config.DEFAULT_DATA_ROOT,
        help=f"Folder containing your unzipped CARE download "
             f"(default: {config.DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=config.DEFAULT_OUTPUT_DIR,
        help=f"Where to write Phase 1 deliverables "
             f"(default: {config.DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--full-status-scan", action="store_true",
        help="Scan every turbine dataset file (not just a 3-file sample "
             "per farm) for the status_type_id legend cross-check. "
             "Slower, but exhaustive.",
    )
    parser.add_argument(
        "--sample-files", type=int, default=3,
        help="Dataset files sampled per farm for the status-legend "
             "cross-check when --full-status-scan is not set (default: 3)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    sample_files = None if args.full_status_scan else args.sample_files

    try:
        result = run_phase1(
            data_root=args.data_root,
            output_dir=args.output_dir,
            sample_files_per_farm=sample_files,
        )
    except Exception:
        logger.exception("Phase 1 run failed")
        return 1

    print()
    print("=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print(f"  Farms processed        : {len(result.inventory)}")
    print(f"  Events tagged          : {len(result.tagged_events)}")
    print(f"  Flagged for review     : {len(result.unclassified)}")
    print(f"  Top-ranked component   : {result.fmeca_table.iloc[0]['component']} "
          f"(RPN={result.fmeca_table.iloc[0]['rpn']})")
    print(f"  Outputs written to     : {args.output_dir.resolve()}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
