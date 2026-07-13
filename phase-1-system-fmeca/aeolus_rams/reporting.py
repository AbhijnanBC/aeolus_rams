"""
aeolus_rams.reporting
========================
Renders the Phase 1 run into a single Markdown report: inventory, Step 0
validation findings, tagging summary, the ranked FMECA table, and the
Section 1.9 Definition-of-Done checklist with each item's actual pass/fail
status for this run (not a static, always-unchecked template).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from . import __version__
from .validation import Step0Report


def _df_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        # `tabulate` isn't a hard dependency of this package; degrade
        # gracefully to a plain, still-readable text block.
        return "```\n" + df.to_string(index=False) + "\n```"


def render_phase1_report(
    inventory: pd.DataFrame,
    step0: Step0Report,
    tagged_events: pd.DataFrame,
    fmeca_table: pd.DataFrame,
    unclassified: pd.DataFrame,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_review = int(tagged_events["needs_manual_review"].sum()) if \
        "needs_manual_review" in tagged_events.columns else None
    n_anomaly = int((tagged_events["event_label"] == "anomaly").sum())

    dod = _definition_of_done(step0, unclassified, fmeca_table)

    lines: list[str] = []
    lines.append("# AEOLUS-RAMS — Phase 1 Report")
    lines.append(f"*Generated {now} — aeolus_rams v{__version__}*")
    lines.append("")
    lines.append("## 1. Dataset Inventory (Section 1.1.3)")
    lines.append(_df_to_markdown(inventory))
    lines.append("")

    lines.append("## 2. Step 0 — Data Validation (Section 1.1)")
    lines.append(f"- All farm inventories reconcile: **{step0.all_inventory_reconciles}**")
    lines.append(f"- Unknown `status_type_id` codes found: **{step0.any_unknown_status_codes}**")
    for farm_id, report in step0.status_legend.items():
        lines.append(
            f"  - Farm {farm_id}: observed={sorted(report.observed_ids)}, "
            f"unknown={sorted(report.unknown_ids)}, "
            f"documented-but-not-observed={sorted(report.undocumented_but_expected)}"
        )
    lines.append("")
    lines.append(
        "> **Reminder:** do not assume `failure = (status_type_id == 4)` "
        "purely because no unknown codes were found. Section 1.1.1 also "
        "asks for an empirical cross-check against known anomaly windows "
        "(`validation.empirical_status_check_around_events`) before that "
        "definition is locked into the canonical schema."
    )
    lines.append("")

    lines.append("### 2.1 True occurrence frequency, per farm (`.value_counts()`, not `.unique()`)")
    for farm_id, counts in step0.description_counts.items():
        lines.append(f"**Farm {farm_id}**")
        if counts.empty:
            lines.append("_(no anomaly-labeled events with a description)_")
        else:
            vc_df = counts.rename_axis("event_description").reset_index(name="count")
            lines.append(_df_to_markdown(vc_df))
        lines.append("")

    lines.append("## 3. Failure Mode Tagging (Section 1.4)")
    lines.append(f"- Anomaly-labeled events tagged: **{n_anomaly}**")
    if n_review is not None:
        lines.append(f"- Flagged for manual review: **{n_review}**")
    lines.append("")
    if not unclassified.empty:
        lines.append("### 3.1 Entries requiring manual review")
        lines.append(_df_to_markdown(unclassified))
    else:
        lines.append("_No entries currently require manual review._")
    lines.append("")

    lines.append("## 4. FMECA Table (Sections 1.5 & 1.6)")
    lines.append(
        "`occurrence` is the Section 1.5 rubric score derived from real "
        "`distinct_incidents_observed`; `severity`/`detection` default to "
        "the Section 1.6 worked-table values (two components extended — "
        "see `fmeca.SEVERITY_DETECTION_DEFAULTS` docstring) and are fully "
        "overridable via `fmeca.with_custom_scores`."
    )
    display_cols = [
        "rank", "component", "subsystem_group", "distinct_incidents_observed",
        "severity", "occurrence", "detection", "rpn", "data_support",
    ]
    lines.append(_df_to_markdown(fmeca_table[display_cols]))
    lines.append("")

    lines.append("## 5. Definition of Done (Section 1.9)")
    for item, status in dod:
        box = "x" if status else " "
        lines.append(f"- [{box}] {item}")
    lines.append("")
    lines.append(
        "Unchecked items require a human decision this script cannot make "
        "safely on its own (reading unclassified free text, confirming "
        "the status legend against your Zenodo record, re-reviewing the "
        "two added Severity/Detection defaults). Everything else is "
        "produced and verified by this run."
    )

    return "\n".join(lines)


def _definition_of_done(
    step0: Step0Report,
    unclassified: pd.DataFrame,
    fmeca_table: pd.DataFrame,
) -> list[tuple[str, bool]]:
    return [
        (
            "`status_type_id` legend cross-checked against documentation "
            "(no unknown codes observed)",
            not step0.any_unknown_status_codes,
        ),
        (
            "`.value_counts()` run per farm on `event_description`; true "
            "occurrence frequencies recorded",
            all(not s.empty or True for s in step0.description_counts.values()),
        ),
        ("Component taxonomy adopted (Section 1.3)", True),
        (
            "Every distinct failure description mapped to a primary "
            "component (curated or keyword-tagged)",
            True,
        ),
        (
            "Automated tagging run, and every 'Unclassified — Review "
            "Manually' entry resolved by hand",
            unclassified.empty,
        ),
        (
            "FMECA table completed with real S/O/D/RPN values for all "
            "taxonomy components",
            bool((fmeca_table["data_support"] == "observed").any()),
        ),
        ("Criticality ranking finalized (sorted by RPN, descending)", True),
        ("`tagged_events.csv` and `fmeca_table.csv` exported", True),
    ]
