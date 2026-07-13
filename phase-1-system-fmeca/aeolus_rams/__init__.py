"""
AEOLUS-RAMS — Phase 1: System Definition + FMECA
==================================================

A production-grade implementation of Phase 1 of the AEOLUS-RAMS methodology
for the CARE-to-Compare wind turbine SCADA benchmark dataset
(Gueck, Roelofs & Faulstich, 2024 — https://doi.org/10.5281/zenodo.14006163).

Phase 1 turns raw CARE event logs into a ranked, evidence-backed FMECA table
that every downstream RAMS phase (Weibull fitting, RBD, Monte Carlo, FTA,
ETA, preventive maintenance optimisation) consumes as its component priority
list.

Module map
----------
config              canonical schema, status legend, scope boundary, paths
taxonomy            13-component ReliaWind-informed taxonomy + keyword rules
data_loader         robust farm discovery + CARE file I/O
validation          Step 0 pre-flight checks (status legend, inventory,
                    value_counts discipline)
component_tagger    hybrid curated-lookup + keyword-fallback failure-mode
                    tagger, with compound-entry and manual-review handling
fmeca               Severity / Occurrence / Detection scoring engine + RPN
reporting           Markdown report generation (Definition-of-Done checklist)
pipeline            CLI orchestrator tying every step together end-to-end

Quick start
-----------
    python -m aeolus_rams.pipeline --data-root data/raw/care --output-dir phase-1-system-fmeca

or, programmatically:

    from aeolus_rams.pipeline import run_phase1
    result = run_phase1(data_root="data/raw/care", output_dir="phase-1-system-fmeca")
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("aeolus-rams-phase1")
except PackageNotFoundError:  # pragma: no cover - local/dev install
    __version__ = "1.0.0-phase1"

__all__ = ["__version__"]
