"""
AEOLUS-RAMS — Phase 2: Weibull / Exponential / MTBF / MTTF / Hazard Rate
===========================================================================

Built directly on top of the installed Phase 1 package (`aeolus_rams`) —
this package does NOT re-implement CARE file discovery or CSV parsing; it
imports `aeolus_rams.data_loader` for that and focuses entirely on what's
new in Phase 2: asset linkage, time-between-failures, data-sufficiency
tiering, distribution fitting, bootstrap uncertainty, and hazard curves.

Module map
----------
config              paths, tier thresholds, fitting minimums
linkage             Section 2.1.1 — event -> asset_id resolution
                    + Section 2.1.2 — timestamp-offset diagnostic
tbf_extraction      Section 2.3 — per-asset, per-component TBF with
                    right-censoring of the final interval
tiering             Section 2.2 — data-sufficiency tier assignment,
                    read live from Phase 1's fmeca_table.csv
distributions       Section 2.4 — Weibull (Tier A) / exponential (Tier B)
                    fitting with a hard guard against small-sample fits
bootstrap           Section 2.7 — bootstrap confidence intervals (Tier A,
                    optionally Tier B), same small-sample guard
hazard              Section 2.6 — hazard-rate curves + illustrative
                    system-level composite, Tier C explicitly unplotted
literature_priors   Section 2.4 Tier C — sourced, cited MTBF priors
reporting           Markdown report generation
pipeline            CLI orchestrator (python -m aeolus_rams_phase2.pipeline)

Quick start
-----------
    python -m aeolus_rams_phase2.pipeline \
        --data-root data/raw/care \
        --phase1-dir phase-1-system-fmeca \
        --output-dir phase-2-weibull-mtbf-hazard

or, programmatically:

    from aeolus_rams_phase2.pipeline import run_phase2
    result = run_phase2("data/raw/care", "phase-1-system-fmeca")
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("aeolus-rams-phase2")
except PackageNotFoundError:  # pragma: no cover - local/dev install
    __version__ = "1.0.0-phase2"

__all__ = ["__version__"]
