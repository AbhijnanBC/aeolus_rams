"""
aeolus_rams_phase2.pipeline
===============================
End-to-end Phase 2 orchestrator. Wires together:

    linkage (2.1)  ->  tiering (2.2), read live from Phase 1's fmeca_table.csv
                              |
                              v
                    tbf_extraction (2.3)
                              |
              re-validate tier against REAL usable TBF counts
                              |
          +-------------------+-------------------+
          |                   |                   |
      Tier A              Tier B              Tier C
  distributions.py    distributions.py    literature_priors.py
  + bootstrap.py       (+ bootstrap.py,
                         not DoD-required)
          |                   |                   |
          +-------------------+-------------------+
                              |
                        hazard.py (plots)
                              |
                        reporting.py

Run as a script:

    python -m aeolus_rams_phase2.pipeline \
        --data-root data/raw/care \
        --phase1-dir phase-1-system-fmeca \
        --output-dir phase-2-weibull-mtbf-hazard

Or import and call `run_phase2(...)` directly from a notebook.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from aeolus_rams.data_loader import discover_farms
from aeolus_rams.taxonomy import COMPONENT_NAMES

from . import config
from .linkage import (
    link_all_farms, linkage_summary, build_observation_end_lookup,
    build_observation_start_lookup, check_offset_consistency_per_asset,
)
from .tiering import load_fmeca_table, tier_table, components_by_tier
from .tbf_extraction import (
    extract_tbf_all_components, tbf_summary, uncensored_tbf_days,
    extract_tbf_by_failure_mode, tbf_summary_by_mode,
)
from .distributions import (
    fit_weibull, fit_exponential_only, InsufficientDataError,
    fit_weibull_with_censoring, fit_exponential_with_censoring,
)
from .bootstrap import bootstrap_weibull_ci, bootstrap_exponential_ci
from .literature_priors import LITERATURE_PRIORS, LiteraturePrior, get_prior, NOT_YET_SOURCED
from .bayesian_inference import bayesian_posterior_poisson, bayesian_weibull_with_literature_shape
from .hazard import render_component_hazard, render_illustrative_system_bathtub
from .reporting import render_phase2_report

logger = logging.getLogger("aeolus_rams_phase2.pipeline")


@dataclass
class Phase2Result:
    linked_events: pd.DataFrame
    linkage_counts: pd.Series
    offset_diagnostic: pd.DataFrame
    tiered_fmeca: pd.DataFrame
    tbf_table: pd.DataFrame
    tbf_summary_df: pd.DataFrame
    tier_downgrades: list[dict]
    tier_a_fits: dict = field(default_factory=dict)
    tier_a_bootstrap: dict = field(default_factory=dict)
    tier_b_fits: dict = field(default_factory=dict)
    tier_b_bootstrap: dict = field(default_factory=dict)
    tier_c_priors: dict = field(default_factory=dict)
    tier_c_bayesian: dict = field(default_factory=dict)  # Bayesian updates for Tier C (new)
    failure_modes_by_component: dict = field(default_factory=dict)  # TBF by failure mode (new)
    failure_mode_summaries: dict = field(default_factory=dict)  # Mode-specific statistics (new)
    mtbf_table: pd.DataFrame = None
    hazard_plot_paths: dict = field(default_factory=dict)
    report_markdown: str = ""


def _load_phase1_tagged_events(phase1_dir: Path) -> pd.DataFrame:
    path = phase1_dir / "tagged_events.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Phase 1 tagged_events.csv not found at {path.resolve()}. "
            "Pass --phase1-dir pointing at your Phase 1 output folder."
        )
    df = pd.read_csv(path)
    for col in ("event_start", "event_end"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def _get_prior_or_fallback(component: str) -> LiteraturePrior:
    try:
        return get_prior(component)
    except KeyError:
        return LiteraturePrior(
            component=component, mtbf_days=None, confidence=NOT_YET_SOURCED,
            source="No literature_priors.py entry exists for this "
                   "component name — check for a taxonomy naming "
                   "mismatch between Phase 1 and Phase 2.",
            derivation_note="Auto-generated fallback; do not treat as "
                             "having been researched.",
        )


def run_phase2(
    data_root: str | Path,
    phase1_dir: str | Path,
    output_dir: str | Path | None = None,
    n_boot: int = 2000,
    seed: int = 42,
    write_outputs: bool = True,
) -> Phase2Result:
    """Run the complete Phase 2 pipeline and (by default) write the
    Section 2.9 deliverables to `output_dir`.
    """
    data_root = Path(data_root)
    phase1_dir = Path(phase1_dir)
    output_dir = Path(output_dir) if output_dir is not None else config.DEFAULT_OUTPUT_DIR

    logger.info("Discovering CARE farms under %s", data_root)
    farm_paths = discover_farms(data_root)

    logger.info("Loading Phase 1 tagged_events.csv from %s", phase1_dir)
    tagged_events = _load_phase1_tagged_events(phase1_dir)
    events_by_farm = {
        farm_id: tagged_events[tagged_events["farm"] == farm_id].copy()
        for farm_id in farm_paths
    }

    logger.info("Linking events to assets (Section 2.1.1)")
    # NOTE: Keep sensor_tiebreaker=False to avoid wide CSV reads (even with csv.reader, MinGW sends SIGINT)
    # Use fast hard-fault filtering + timespan tiebreaker instead. tbf_extraction.py now accepts
    # both _sensor_resolved and _timespan_resolved matches via generic "resolved" suffix check.
    linked_events = link_all_farms(farm_paths, events_by_farm, use_status_filtering=True, use_sensor_tiebreaker=True)
    linkage_counts = linkage_summary(linked_events)
    for label, count in linkage_counts.items():
        level = logging.INFO if label == "unique_match" else logging.WARNING
        logger.log(level, "link_confidence=%s: %d event(s)", label, count)

    logger.info("Running timestamp-offset diagnostic (Section 2.1.2)")
    offset_diagnostic = check_offset_consistency_per_asset(linked_events)

    logger.info("Loading Phase 1 fmeca_table.csv and assigning tiers (Section 2.2)")
    fmeca_df = load_fmeca_table(phase1_dir / "fmeca_table.csv")
    tiered_fmeca = tier_table(fmeca_df)
    initial_tiers = dict(zip(tiered_fmeca["component"], tiered_fmeca["tier"]))

    logger.info("Building asset observation-end lookup for censoring")
    asset_observation_end = build_observation_end_lookup(farm_paths)
    
    logger.info("Building asset observation-start lookup for 'time to first failure' intervals")
    asset_observation_start = build_observation_start_lookup(farm_paths)

    logger.info("Extracting time-between-failures for all %d components (Section 2.3)", len(COMPONENT_NAMES))
    tbf_table = extract_tbf_all_components(
        linked_events, list(COMPONENT_NAMES), asset_observation_end,
        asset_observation_start=asset_observation_start,
    )
    tbf_summary_df = tbf_summary(tbf_table)
    n_uncensored_by_component = dict(
        zip(tbf_summary_df["component"], tbf_summary_df["n_uncensored"])
    )

    # --- Verify right-censoring handling (Section 2.3, strict validation) ---
    logger.info("Verifying strict right-censoring implementation")
    for component in COMPONENT_NAMES:
        component_data = tbf_table[tbf_table["component"] == component]
        if component_data.empty:
            continue
        
        n_censored = (component_data["censored"]).sum()
        n_uncensored = (~component_data["censored"]).sum()
        
        if n_censored > 0:
            mean_censored_days = component_data[component_data["censored"]]["tbf_days"].mean()
            logger.debug(
                "%s: right-censoring verified — %d complete intervals + %d right-censored "
                "(mean censored tail %.0f days). MLE fitting will use complete intervals only.",
                component, n_uncensored, n_censored, mean_censored_days,
            )

    # --- Re-validate tier assignment against REAL usable TBF counts ----
    final_tiers: dict[str, str] = {}
    tier_downgrades: list[dict] = []
    for component in COMPONENT_NAMES:
        initial_tier = initial_tiers.get(component, config.TIER_C)
        n_usable = n_uncensored_by_component.get(component, 0)

        if initial_tier == config.TIER_A and n_usable < config.MIN_TBF_FOR_WEIBULL:
            downgraded_to = (
                config.TIER_B if n_usable >= config.MIN_TBF_FOR_EXPONENTIAL else config.TIER_C
            )
            logger.warning(
                "%s: Phase 1 incident count suggested Tier A, but only "
                "%d usable (uncensored) TBF interval(s) survive linkage — "
                "downgrading to Tier %s.", component, n_usable, downgraded_to,
            )
            tier_downgrades.append({
                "component": component, "from_tier": "A", "to_tier": downgraded_to,
                "n_usable_tbf": n_usable,
                "reason": "raw incident count overstated usable TBF intervals "
                          "once split across assets (Section 2.3)",
            })
            final_tiers[component] = downgraded_to
        elif initial_tier == config.TIER_B and n_usable < config.MIN_TBF_FOR_EXPONENTIAL:
            logger.warning(
                "%s: Phase 1 incident count suggested Tier B, but only "
                "%d usable TBF interval(s) survive linkage — downgrading "
                "to Tier C.", component, n_usable,
            )
            tier_downgrades.append({
                "component": component, "from_tier": "B", "to_tier": "C",
                "n_usable_tbf": n_usable,
                "reason": "raw incident count overstated usable TBF intervals "
                          "once split across assets (Section 2.3)",
            })
            final_tiers[component] = config.TIER_C
        else:
            final_tiers[component] = initial_tier

    tiers = {config.TIER_A: [], config.TIER_B: [], config.TIER_C: []}
    for component, tier in final_tiers.items():
        tiers[tier].append(component)

    # --- Tier A: 2-parameter Weibull + bootstrap -----------------------
    tier_a_fits, tier_a_bootstrap = {}, {}
    for component in tiers[config.TIER_A]:
        tbf_days = uncensored_tbf_days(tbf_table, component)
        try:
            fit = fit_weibull(tbf_days)
            tier_a_fits[component] = fit
            tier_a_bootstrap[component] = bootstrap_weibull_ci(tbf_days, n_boot=n_boot, seed=seed)
        except InsufficientDataError as exc:
            logger.error("%s: Tier A fit failed despite re-validation: %s", component, exc)

    # --- Tier B: exponential-only + (informational) bootstrap ----------
    tier_b_fits, tier_b_bootstrap = {}, {}
    for component in tiers[config.TIER_B]:
        tbf_days = uncensored_tbf_days(tbf_table, component)
        try:
            fit = fit_exponential_only(tbf_days)
            tier_b_fits[component] = fit
            tier_b_bootstrap[component] = bootstrap_exponential_ci(tbf_days, n_boot=n_boot, seed=seed)
        except InsufficientDataError as exc:
            logger.error("%s: Tier B fit failed despite re-validation: %s", component, exc)

    # --- Tier C: literature priors + Bayesian updating (new enhancement) ---
    tier_c_priors = {c: _get_prior_or_fallback(c) for c in tiers[config.TIER_C]}
    tier_c_bayesian = {}
    
    # Apply Bayesian updating to Tier C components with 1-4 observed failures
    logger.info("Applying Bayesian inference to Tier C components with limited data")
    for component in tiers[config.TIER_C]:
        prior = tier_c_priors.get(component)
        if prior is None or not prior.is_usable:
            continue
        
        tbf_complete = uncensored_tbf_days(tbf_table, component)
        n_failures = len(tbf_complete)
        
        if 0 <= n_failures <= 4:  # BUG FIX: Include 0-failure components for Bayesian update
            # Compute total observation time across all assets
            component_obs = tbf_table[tbf_table["component"] == component]
            t_total = component_obs["tbf_days"].sum()
            
            # Apply Bayesian Poisson updating
            try:
                bayesian_result = bayesian_posterior_poisson(
                    component=component,
                    n_failures=n_failures,
                    t_observation_days=t_total,
                    prior_mtbf_days=prior.mtbf_days,
                )
                tier_c_bayesian[component] = bayesian_result
                
                # Log the improvement
                ratio = bayesian_result.prior_lambda_to_posterior_ratio
                if ratio > 1.2:
                    logger.warning(
                        "%s: Bayesian MTBF %.0f → %.0f days (posterior λ is %.2f× prior λ). "
                        "Farm data shows HIGHER failure rate than literature.",
                        component, prior.mtbf_days, bayesian_result.posterior_mtbf_days, ratio,
                    )
                elif ratio < 0.8:
                    logger.info(
                        "%s: Bayesian MTBF %.0f → %.0f days (posterior λ is %.2f× prior λ). "
                        "Farm data shows LOWER failure rate than literature.",
                        component, prior.mtbf_days, bayesian_result.posterior_mtbf_days, ratio,
                    )
            except Exception as e:
                logger.warning("%s: Bayesian update failed: %s", component, e)

    # --- Failure Mode Analysis (new enhancement) -------------------------
    logger.info("Analyzing failure modes within components")
    failure_modes_by_component = {}
    failure_mode_summaries = {}
    
    for component in COMPONENT_NAMES:
        try:
            modes_tbf = extract_tbf_by_failure_mode(
                linked_events, component, asset_observation_end,
                asset_observation_start=asset_observation_start,
            )
            if modes_tbf:
                failure_modes_by_component[component] = modes_tbf
                mode_summary = tbf_summary_by_mode(modes_tbf, component)
                failure_mode_summaries[component] = mode_summary
                
                if len(modes_tbf) > 1:
                    logger.info(
                        "%s: identified %d distinct failure modes — may warrant separate fits",
                        component, len(modes_tbf),
                    )
        except Exception as e:
            logger.debug("%s: failure mode analysis skipped: %s", component, e)

    # --- Unified MTBF table ---------------------------------------------
    mtbf_table = _build_mtbf_table(
        tier_a_fits, tier_a_bootstrap, tier_b_fits, tier_c_priors, tiered_fmeca,
        tier_c_bayesian=tier_c_bayesian,  # BUG FIX: Pass Bayesian posteriors
    )

    # --- Hazard plots -----------------------------------------------------
    hazard_plot_paths: dict[str, str] = {}
    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        component_plot = render_component_hazard(
            tier_a_fits, tier_b_fits, tiers[config.TIER_C],
            output_dir / "hazard_tier_a_b.png",
        )
        hazard_plot_paths["hazard_tier_a_b.png"] = str(component_plot)

        composite_plot = render_illustrative_system_bathtub(
            tier_a_fits, tier_b_fits, tier_c_priors,
            output_dir / "hazard_system_illustrative.png",
        )
        hazard_plot_paths["hazard_system_illustrative.png"] = str(composite_plot)

    report_md = render_phase2_report(
        linkage_counts, offset_diagnostic, tiered_fmeca, tbf_summary_df,
        tier_a_fits, tier_a_bootstrap, tier_b_fits, tier_c_priors,
        mtbf_table, hazard_plot_paths, tier_downgrades,
    )

    result = Phase2Result(
        linked_events=linked_events,
        linkage_counts=linkage_counts,
        offset_diagnostic=offset_diagnostic,
        tiered_fmeca=tiered_fmeca,
        tbf_table=tbf_table,
        tbf_summary_df=tbf_summary_df,
        tier_downgrades=tier_downgrades,
        tier_a_fits=tier_a_fits,
        tier_a_bootstrap=tier_a_bootstrap,
        tier_b_fits=tier_b_fits,
        tier_b_bootstrap=tier_b_bootstrap,
        tier_c_priors=tier_c_priors,
        tier_c_bayesian=tier_c_bayesian,
        failure_modes_by_component=failure_modes_by_component,
        failure_mode_summaries=failure_mode_summaries,
        mtbf_table=mtbf_table,
        hazard_plot_paths=hazard_plot_paths,
        report_markdown=report_md,
    )

    if write_outputs:
        _write_outputs(result, output_dir)

    return result


def _build_mtbf_table(
    tier_a_fits, tier_a_bootstrap, tier_b_fits, tier_c_priors, tiered_fmeca,
    tier_c_bayesian: dict | None = None,  # BUG FIX: Accept Bayesian posteriors
) -> pd.DataFrame:
    rows = []
    incidents_lookup = dict(zip(tiered_fmeca["component"], tiered_fmeca["distinct_incidents_observed"]))

    for component, fit in tier_a_fits.items():
        boot = tier_a_bootstrap.get(component)
        has_ci = boot is not None and boot.is_meaningful
        ci = boot.mtbf_ci_days if has_ci else (None, None)
        rows.append({
            "component": component, "tier": "A",
            "n_incidents_phase1": incidents_lookup.get(component),
            "n_tbf_used": fit.n_used,
            "mtbf_days": fit.mtbf_days,
            "beta": fit.beta, "eta": fit.eta, "lambda": None,
            "ci_low_days": ci[0], "ci_high_days": ci[1],
            "confidence": "fitted_tier_a",
            "source": f"Weibull MLE fit ({fit.n_used} TBF intervals); "
                      f"preferred over exponential: {fit.preferred == 'weibull'}",
        })

    for component, fit in tier_b_fits.items():
        rows.append({
            "component": component, "tier": "B",
            "n_incidents_phase1": incidents_lookup.get(component),
            "n_tbf_used": fit.n_used,
            "mtbf_days": fit.mtbf_days,
            "beta": None, "eta": None, "lambda": fit.lambda_rate,
            "ci_low_days": None, "ci_high_days": None,
            "confidence": "fitted_tier_b",
            "source": f"Exponential MLE fit ({fit.n_used} TBF intervals)",
        })

    for component, prior in tier_c_priors.items():
        # BUG FIX: Use Bayesian posterior MTBF if available; otherwise fall back to literature prior
        if tier_c_bayesian and component in tier_c_bayesian:
            bayesian_result = tier_c_bayesian[component]
            final_mtbf = bayesian_result.posterior_mtbf_days
            final_confidence = bayesian_result.confidence
            final_source = f"Bayesian posterior (observational + {prior.source}): {bayesian_result.n_observed_failures} failures in {bayesian_result.t_observation_days:.0f} days"
        else:
            final_mtbf = prior.mtbf_days
            final_confidence = prior.confidence
            final_source = prior.source
        
        rows.append({
            "component": component, "tier": "C",
            "n_incidents_phase1": incidents_lookup.get(component),
            "n_tbf_used": 0,
            "mtbf_days": final_mtbf,
            "beta": None, "eta": None, "lambda": None,
            "ci_low_days": None, "ci_high_days": None,
            "confidence": final_confidence,
            "source": final_source,
        })

    df = pd.DataFrame(rows)
    tier_order = {"A": 0, "B": 1, "C": 2}
    df["_sort"] = df["tier"].map(tier_order)
    df = df.sort_values(["_sort", "mtbf_days"], na_position="last").drop(columns="_sort")
    return df.reset_index(drop=True)


def _write_outputs(result: Phase2Result, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    result.linked_events.to_csv(output_dir / "linked_events.csv", index=False)
    result.tbf_table.to_csv(output_dir / "tbf_table.csv", index=False)
    result.mtbf_table.to_csv(output_dir / "mtbf_table.csv", index=False)
    if not result.offset_diagnostic.empty:
        result.offset_diagnostic.to_csv(output_dir / "offset_diagnostic.csv", index=False)
    (output_dir / "phase2_report.md").write_text(result.report_markdown, encoding="utf-8")

    logger.info("Wrote %s", output_dir / "linked_events.csv")
    logger.info("Wrote %s", output_dir / "tbf_table.csv")
    logger.info("Wrote %s", output_dir / "mtbf_table.csv")
    logger.info("Wrote %s", output_dir / "phase2_report.md")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aeolus-rams-phase2",
        description="AEOLUS-RAMS Phase 2 — Weibull/exponential fits, "
                     "MTBF, and hazard rate, tiered by data sufficiency.",
    )
    parser.add_argument("--data-root", type=Path, default=config.DEFAULT_DATA_ROOT)
    parser.add_argument("--phase1-dir", type=Path, default=config.DEFAULT_PHASE1_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        result = run_phase2(
            data_root=args.data_root,
            phase1_dir=args.phase1_dir,
            output_dir=args.output_dir,
            n_boot=args.n_boot,
            seed=args.seed,
        )
    except Exception:
        logger.exception("Phase 2 run failed")
        return 1

    print()
    print("=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)
    print(f"  Tier A (Weibull)        : {len(result.tier_a_fits)} component(s)")
    print(f"  Tier B (exponential)    : {len(result.tier_b_fits)} component(s)")
    print(f"  Tier C (literature)     : {len(result.tier_c_priors)} component(s)")
    print(f"  Tier downgrades applied : {len(result.tier_downgrades)}")
    print(f"  Outputs written to      : {args.output_dir.resolve()}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
