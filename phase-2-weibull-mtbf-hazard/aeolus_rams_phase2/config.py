"""
aeolus_rams_phase2.config
============================
Central configuration: default paths, Section 2.2's tier thresholds, and
the minimum usable-sample sizes Section 2.4/2.7 insist on before any fit
or bootstrap is allowed to run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem defaults
# ---------------------------------------------------------------------------

DEFAULT_DATA_ROOT = Path("data/raw/care")
DEFAULT_PHASE1_DIR = Path("phase-1-system-fmeca")
DEFAULT_OUTPUT_DIR = Path("phase-2-weibull-mtbf-hazard")

#: Column in Phase 1's fmeca_table.csv that Section 2.2's tiering reads.
INCIDENT_COUNT_COLUMN = "distinct_incidents_observed"

#: Column in Phase 1's tagged_events.csv identifying each event's tagged
#: primary component — the join key between Phase 1 and Phase 2.
COMPONENT_COLUMN = "component_primary"


# ---------------------------------------------------------------------------
# Section 2.2 — data-sufficiency tiers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TierThresholds:
    """>= min_a incidents -> Tier A (2-parameter Weibull).
    min_b <= incidents < min_a -> Tier B (exponential only).
    < min_b -> Tier C (literature-informed placeholder)."""
    min_a: int = 8
    min_b: int = 5


TIER_THRESHOLDS = TierThresholds()

TIER_A = "A"
TIER_B = "B"
TIER_C = "C"

TIER_DESCRIPTIONS = {
    TIER_A: "Fit directly (2-parameter Weibull)",
    TIER_B: "Fit a 1-parameter exponential only, no shape parameter",
    TIER_C: "Literature-informed placeholder, no standalone fit",
}


# ---------------------------------------------------------------------------
# Section 2.4 / 2.7 — minimum sample sizes for fitting and bootstrapping
# ---------------------------------------------------------------------------
# These are USABLE, UNCENSORED TIME-BETWEEN-FAILURE INTERVAL counts, which
# Section 2.3's own "hard limit" warning notes can be SMALLER than a
# component's raw incident count once incidents are split across multiple
# assets (n incidents on k assets yields at most n-k uncensored intervals).
# `pipeline.py` re-validates each component's tier against this real,
# post-linkage count and downgrades with a loud warning if the incident-
# count-based tier from Section 2.2 turns out to be too optimistic.

MIN_TBF_FOR_WEIBULL = TIER_THRESHOLDS.min_a       # Tier A fitting floor
MIN_TBF_FOR_EXPONENTIAL = TIER_THRESHOLDS.min_b   # Tier B fitting floor
