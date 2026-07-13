"""
aeolus_rams.config
===================
Central configuration for Phase 1: canonical schema mapping, the CARE
turbine-status legend (Section 1.1.1 / CARE paper Table 2), the system
boundary (Section 1.2), and filesystem defaults.

Everything here is a plain, importable constant so downstream modules and
notebooks can inspect or override it without touching pipeline logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Filesystem defaults
# ---------------------------------------------------------------------------

#: Default project-relative CARE data root (Phase 0 folder convention).
#: CARE's own zip distribution nests each farm inside a doubled folder name
#: (e.g. "Wind Farm A/Wind Farm A/") — this is inherent to how the archive
#: unzips and is handled transparently by `data_loader.discover_farms`,
#: which walks the tree with `rglob` rather than assuming a fixed depth.
DEFAULT_DATA_ROOT = Path("data/raw/care")

#: Default Phase 1 output directory (per Section 1.9 deliverables).
DEFAULT_OUTPUT_DIR = Path("phase-1-system-fmeca")

#: The field separator CARE ships its CSVs with. Confirmed against the
#: published dataset and against every recon pull performed so far —
#: NOT a comma.
CARE_CSV_SEPARATOR = ";"


# ---------------------------------------------------------------------------
# Canonical schema
# ---------------------------------------------------------------------------
# Maps AEOLUS-RAMS's internal field names to the actual CARE column names.
# NOTE: an earlier recon pass mapped "status" -> "status_type" — that column
# does not exist in the published files. The real column is
# `status_type_id`. This mapping is the corrected, verified version and is
# the single source of truth used by every module in this package.

CANONICAL_SCHEMA: dict[str, str] = {
    "timestamp": "time_stamp",
    "asset_id": "asset_id",
    "row_id": "id",
    "split": "train_test",
    "status": "status_type_id",
    "component_free_text": "event_description",
    "event_label": "event_label",
}

#: event_info.csv columns, in the order CARE publishes them.
EVENT_INFO_COLUMNS = (
    "event_id",
    "event_label",
    "event_start",
    "event_start_id",
    "event_end",
    "event_end_id",
    "event_description",
)

#: feature_description.csv columns, in the order CARE publishes them.
FEATURE_DESCRIPTION_COLUMNS = (
    "sensor_name",
    "statistics_type",
    "description",
    "unit",
    "is_angle",
    "is_counter",
)

#: Non-sensor ("metadata") columns present in every per-turbine dataset CSV.
DATASET_METADATA_COLUMNS = (
    "time_stamp",
    "asset_id",
    "id",
    "train_test",
    "status_type_id",
)


# ---------------------------------------------------------------------------
# Turbine status legend  (CARE paper, Table 2 — Gueck et al. 2024)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StatusCode:
    status_id: int
    description: str
    considered_normal: bool


#: The authoritative status_type_id legend, as published in the CARE paper.
#: Section 1.1.1 of the Phase 1 spec explicitly warns against assuming
#: `failure = (status_type_id == 4)` without checking this table first —
#: it is reproduced here verbatim as the single source of truth, and
#: `validation.confirm_status_legend` cross-checks it empirically against
#: whatever status codes actually appear in the user's own download.
STATUS_LEGEND: dict[int, StatusCode] = {
    0: StatusCode(0, "Normal operation without limitations", True),
    1: StatusCode(1, "Derated power generation with a power restriction", False),
    2: StatusCode(2, "Asset is idling and waits to operate again", True),
    3: StatusCode(3, "Asset is in service mode / service team is at the site", False),
    4: StatusCode(4, "Asset is down due to a fault or other reasons", False),
    5: StatusCode(5, "Other operational states (system test, setup, ice build-up, "
                     "emergency power, etc.)", False),
}

#: The status_type_id value that Section 1.1.1 identifies as the fault /
#: downtime code. Used as the *default* failure flag ONLY after
#: `validation.confirm_status_legend` has been run — never assume this
#: blindly on a new CARE-style dataset without checking documentation.
FAULT_STATUS_ID = 4


def normal_status_ids() -> set[int]:
    """status_type_id values considered normal operating behaviour."""
    return {code.status_id for code in STATUS_LEGEND.values() if code.considered_normal}


def abnormal_status_ids() -> set[int]:
    """status_type_id values considered NOT normal operating behaviour."""
    return {code.status_id for code in STATUS_LEGEND.values() if not code.considered_normal}


# ---------------------------------------------------------------------------
# System boundary (Section 1.2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScopeBoundary:
    in_scope: tuple[str, ...] = field(default_factory=tuple)
    out_of_scope: tuple[str, ...] = field(default_factory=tuple)
    out_of_scope_rationale: str = ""


PHASE1_SCOPE = ScopeBoundary(
    in_scope=(
        "rotor & blades", "pitch system", "main/rotor bearing", "gearbox",
        "generator", "mechanical brake", "hydraulic system", "converter",
        "transformer", "yaw system", "SCADA/control", "electrical safety "
        "systems", "cooling", "grounding/lightning protection",
    ),
    out_of_scope=(
        "array cables", "offshore substation", "export cable",
        "onshore grid connection",
    ),
    out_of_scope_rationale=(
        "These are balance-of-plant items with no fault data in CARE's "
        "event_info.csv or SCADA files. They are parameterised with "
        "literature-typical failure-rate figures in Phase 3's farm-level "
        "RBD instead of being forced into this phase's FMECA."
    ),
)


# ---------------------------------------------------------------------------
# Farm-name discovery
# ---------------------------------------------------------------------------

#: Regex-friendly pattern fragment used by data_loader to robustly recover
#: a farm letter/id from an arbitrary path, before falling back to the
#: original "last character of the parent folder name" heuristic from the
#: Phase 1 spec's reference snippet (Section 1.8).
FARM_NAME_PATTERN = r"Wind\s*Farm\s*([A-Za-z0-9]+)"
