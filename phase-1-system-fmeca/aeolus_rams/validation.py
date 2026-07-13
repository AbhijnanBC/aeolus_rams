"""
aeolus_rams.validation
=========================
Section 1.1 — "Confirm your data before trusting anything downstream."

Three checks, each under an hour combined per the spec, implemented here
as reusable, testable functions rather than one-off notebook cells:

1.1.1  confirm_status_legend()        — status_type_id legend cross-check
1.1.2  event_description_value_counts() — true frequency, not `.unique()`
1.1.3  (data_loader.inventory_summary — basic inventory sanity check)

Every function returns a structured result object; nothing here just
prints and forgets, so `pipeline.py` can fail fast on a real problem
instead of silently continuing past it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from . import config
from .data_loader import FarmPaths, load_dataset_file

logger = logging.getLogger("aeolus_rams.validation")


# ---------------------------------------------------------------------------
# 1.1.1 — status_type_id legend confirmation
# ---------------------------------------------------------------------------

@dataclass
class StatusLegendReport:
    farm_id: str
    observed_ids: set[int]
    documented_ids: set[int]
    unknown_ids: set[int]              # observed but not in STATUS_LEGEND
    undocumented_but_expected: set[int]  # in STATUS_LEGEND, never observed
    counts: dict[int, int] = field(default_factory=dict)

    @property
    def is_clean(self) -> bool:
        """True if every status code observed in the data is accounted for
        in the documented legend. This does NOT mean it's safe to assume
        `failure = (status_type_id == 4)` — only that no *unknown* codes
        were found. Read Section 1.1.1 in full before locking that
        definition in."""
        return len(self.unknown_ids) == 0


def confirm_status_legend(
    farm_paths: dict[str, FarmPaths],
    sample_files_per_farm: int | None = 3,
) -> dict[str, StatusLegendReport]:
    """Cross-check the documented STATUS_LEGEND against what actually
    appears in each farm's SCADA files.

    Parameters
    ----------
    sample_files_per_farm : int | None
        Number of dataset files to scan per farm (for speed — a farm-C
        file can carry 60k+ rows across 957 columns). Pass None to scan
        every file for an exhaustive count (slow, but authoritative).

    This is the *documentation cross-check* half of Section 1.1.1. The
    *empirical* half — checking whether an ambiguous code clusters around
    known anomaly windows — is `empirical_status_check_around_events`
    below; run both before locking down a failure definition.
    """
    reports: dict[str, StatusLegendReport] = {}
    documented = set(config.STATUS_LEGEND.keys())

    for farm_id, fp in farm_paths.items():
        files = fp.dataset_files()
        if sample_files_per_farm is not None:
            files = files[:sample_files_per_farm]

        counts: dict[int, int] = {}
        for f in files:
            df = load_dataset_file(f, columns=["status_type_id"])
            vc = df["status_type_id"].value_counts()
            for status_id, n in vc.items():
                counts[int(status_id)] = counts.get(int(status_id), 0) + int(n)

        observed = set(counts.keys())
        report = StatusLegendReport(
            farm_id=farm_id,
            observed_ids=observed,
            documented_ids=documented,
            unknown_ids=observed - documented,
            undocumented_but_expected=documented - observed,
            counts=dict(sorted(counts.items())),
        )
        if report.unknown_ids:
            logger.warning(
                "Farm %s: status_type_id value(s) %s appear in the data but "
                "are NOT in the documented STATUS_LEGEND. Do not fold them "
                "into any failure/normal flag until you've identified them "
                "(check the CARE Zenodo record / feature_description.csv).",
                farm_id, sorted(report.unknown_ids),
            )
        reports[farm_id] = report

    return reports


def empirical_status_check_around_events(
    fp: FarmPaths,
    events: pd.DataFrame,
    status_col: str = "status_type_id",
    window_hours: float = 2.0,
) -> pd.DataFrame:
    """Section 1.1.1, Step 2 — for each known anomaly event, inspect what
    status codes appear in the surrounding SCADA data.

    If an ambiguous code clusters tightly around real anomaly windows,
    treat it as fault-adjacent. If it appears mostly during normal-labeled
    stretches, it's something else (curtailment, grid-imposed derating)
    and should NOT be folded into a failure flag.

    Returns a tidy DataFrame — one row per (event, status_id) — instead of
    print statements, so results can be filtered/joined programmatically.
    """
    files = fp.dataset_files()
    if not files:
        raise ValueError(f"No dataset files found for farm {fp.farm_id}")
    df = load_dataset_file(files[0])  # matches spec's "first file" sampling

    anomaly_events = events[events["event_label"] == "anomaly"]
    rows = []
    window = pd.Timedelta(hours=window_hours)

    for _, event in anomaly_events.iterrows():
        mask = (
            (df["time_stamp"] >= event["event_start"] - window)
            & (df["time_stamp"] <= event["event_end"] + window)
        )
        subset = df.loc[mask, status_col]
        if subset.empty:
            continue
        for status_id, n in subset.value_counts().items():
            rows.append({
                "farm": fp.farm_id,
                "event_id": event["event_id"],
                "event_description": event.get("event_description"),
                "status_id": int(status_id),
                "count_in_window": int(n),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1.1.2 — true occurrence frequency (.value_counts(), never .unique())
# ---------------------------------------------------------------------------

def event_description_value_counts(
    event_tables: dict[str, pd.DataFrame],
    event_label: str = "anomaly",
) -> dict[str, pd.Series]:
    """Per-farm `.value_counts()` on `event_description`, restricted to
    anomaly-labeled events (normal-labeled events have no meaningful
    description). This is the ONLY correct way to get true incident
    frequency per Section 1.1.2 — `.unique()` silently discards repeat
    counts that Occurrence scoring (Section 1.5) needs.
    """
    result: dict[str, pd.Series] = {}
    for farm_id, df in event_tables.items():
        anomalies = df[df["event_label"] == event_label]
        result[farm_id] = anomalies["event_description"].value_counts()
    return result


# ---------------------------------------------------------------------------
# Aggregate Step-0 report
# ---------------------------------------------------------------------------

@dataclass
class Step0Report:
    inventory: pd.DataFrame
    status_legend: dict[str, StatusLegendReport]
    description_counts: dict[str, pd.Series]

    @property
    def all_inventory_reconciles(self) -> bool:
        return bool(self.inventory["counts_reconcile"].all())

    @property
    def any_unknown_status_codes(self) -> bool:
        return any(r.unknown_ids for r in self.status_legend.values())

    def summary_lines(self) -> list[str]:
        lines = [
            f"Inventory reconciles for all farms: {self.all_inventory_reconciles}",
            f"Unknown status_type_id codes found: {self.any_unknown_status_codes}",
        ]
        for farm_id, report in self.status_legend.items():
            lines.append(
                f"  Farm {farm_id}: observed={sorted(report.observed_ids)} "
                f"unknown={sorted(report.unknown_ids)} "
                f"never_observed_but_documented={sorted(report.undocumented_but_expected)}"
            )
        return lines


def run_step0_checks(
    farm_paths: dict[str, FarmPaths],
    event_tables: dict[str, pd.DataFrame],
    sample_files_per_farm: int | None = 3,
) -> Step0Report:
    """Run every Section 1.1 check and bundle the results. Does not raise
    on findings (unknown codes, non-reconciling counts) — it *reports*
    them, so a human makes the final call, per the spec's intent. Use
    `Step0Report.any_unknown_status_codes` / `.all_inventory_reconciles`
    in calling code if you want a hard fail-fast gate instead.
    """
    from .data_loader import inventory_summary

    inventory = inventory_summary(farm_paths)
    status_legend = confirm_status_legend(farm_paths, sample_files_per_farm)
    description_counts = event_description_value_counts(event_tables)

    report = Step0Report(
        inventory=inventory,
        status_legend=status_legend,
        description_counts=description_counts,
    )
    for line in report.summary_lines():
        logger.info(line)
    return report
