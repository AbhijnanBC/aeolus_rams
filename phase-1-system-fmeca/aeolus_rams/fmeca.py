"""
aeolus_rams.fmeca
====================
Sections 1.5 & 1.6 — FMECA scoring rubric and worked table, turned into a
reusable scoring engine.

What's a fixed default vs. computed from your data
----------------------------------------------------
- **Occurrence (O)** is ALWAYS computed from your real, tagged event data
  (`build_fmeca_table`) via the Section 1.5 bucketing rule — never a
  hardcoded default. This is the whole point of Section 1.1.2's
  `.value_counts()` discipline.
- **Severity (S)** and **Detection (D)** default to the domain-engineering
  judgment values given in Section 1.6's worked table, transcribed
  verbatim in `SEVERITY_DETECTION_DEFAULTS` below, PLUS two components the
  source table didn't score (Cooling System, Grounding/Lightning
  Protection — see the docstring on each for the added reasoning). These
  are *defaults*, fully overridable via `custom_scores` — re-score them
  against your own judgment before treating any ranking as final, exactly
  as Section 1.6's opening line instructs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from . import taxonomy as tax


# ---------------------------------------------------------------------------
# Occurrence bucketing (Section 1.5)
# ---------------------------------------------------------------------------

def occurrence_score(n_incidents: int) -> int:
    """Section 1.5's Occurrence rubric:

        >= 8 distinct incidents -> 9
        3-4 distinct incidents  -> 6
        2 distinct incidents    -> 4
        1 distinct incident     -> 2

    The source rubric doesn't define 0 or 5-7; this implementation
    extends it with a documented, monotonic interpolation:

        0 incidents   -> 1  (no observed data support at all — below the
                              single-incident anchor, not equal to it)
        5-7 incidents -> 7  (linearly between the 3-4 anchor (6) and the
                              >=8 anchor (9))

    These two extensions are clearly separable from the source rubric —
    grep for "interpolated" if you want to swap in your own scheme.
    """
    if n_incidents < 0:
        raise ValueError("n_incidents must be >= 0")
    if n_incidents == 0:
        return 1
    if n_incidents == 1:
        return 2
    if n_incidents == 2:
        return 4
    if n_incidents in (3, 4):
        return 6
    if n_incidents in (5, 6, 7):
        return 7  # interpolated
    return 9  # >= 8


# ---------------------------------------------------------------------------
# Severity / Detection defaults (Section 1.6)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComponentScoreDefaults:
    component: str
    severity: int
    detection: int
    failure_cause: str
    local_effect: str
    system_effect: str
    source: str = "Section 1.6 worked table"


SEVERITY_DETECTION_DEFAULTS: dict[str, ComponentScoreDefaults] = {
    d.component: d for d in (
        ComponentScoreDefaults(
            "Main/Rotor Bearing", 10, 8,
            "Fatigue, wear, inadequate lubrication, misalignment",
            "Increased friction, vibration, heat in main shaft",
            "Turbine forced offline; risk of cascading drivetrain damage; "
            "among the most expensive repairs",
        ),
        ComponentScoreDefaults(
            "Pitch System", 8, 5,
            "Motor/electrical wear, encoder drift, control card fault",
            "Loss of precise blade angle control",
            "Reduced power regulation; in worst sub-modes, loss of the "
            "primary overspeed protection layer",
        ),
        ComponentScoreDefaults(
            "Gearbox", 9, 6,
            "Lubrication failure, gear tooth wear/fatigue",
            "Abnormal oil condition, unusual vibration/noise",
            "Torque transfer degraded/lost; turbine stopped; one of the "
            "costliest drivetrain repairs",
        ),
        ComponentScoreDefaults(
            "Generator", 9, 7,
            "Bearing wear, winding insulation fault",
            "Abnormal bearing temperature/vibration",
            "Loss of generation at that turbine; major component "
            "replacement",
        ),
        ComponentScoreDefaults(
            "Mechanical Brake", 9, 3,
            "Hydraulic/electrical supply fault, disc/caliper wear",
            "Brake fails to apply/release correctly",
            "Loss of a redundant overspeed protection layer — "
            "safety-relevant",
        ),
        ComponentScoreDefaults(
            "Transformer", 8, 5,
            "Insulation degradation, overpressure, oil contamination",
            "Internal electrical fault, pressure relief activation",
            "Loss of power export from that turbine; fire/containment "
            "risk",
        ),
        ComponentScoreDefaults(
            "Hydraulic System", 7, 4,
            "Pump fault, accumulator fault, low fluid level",
            "Reduced/lost hydraulic pressure",
            "Degraded pitch or brake actuation — a shared support system "
            "for two safety-critical functions",
        ),
        ComponentScoreDefaults(
            "Converter", 6, 3,
            "Power electronics fault, fuse/filter failure",
            "Abnormal power conversion behavior",
            "Turbine derated or stopped; production loss until module "
            "replaced",
        ),
        ComponentScoreDefaults(
            "Yaw System", 5, 4,
            "Drive/motor wear, grease system fault",
            "Yaw misalignment",
            "Reduced energy capture, increased loading; rarely "
            "safety-critical alone",
        ),
        ComponentScoreDefaults(
            "SCADA/Communication", 6, 2,
            "Fieldbus/communication module fault",
            "Loss of turbine-to-SCADA data link",
            "Loss of monitoring visibility; concurrent with another "
            "fault, removes the automated trip's ability to act",
        ),
        ComponentScoreDefaults(
            "Electrical Safety System", 8, 2,
            "Relay fault, RCD fault",
            "Safety interlock opens, or fails to protect",
            "Fail-safe shutdown, or reduced electrical fault protection",
        ),
        # --- Not scored in the Section 1.6 source table; added here with
        # documented reasoning so all 13 taxonomy components have a
        # default. Both are explicitly flagged for re-review.
        ComponentScoreDefaults(
            "Cooling System", 5, 6,
            "Valve mispositioning, blocked/degraded cooling circuit "
            "(non-gearbox)",
            "Reduced heat rejection, rising component temperature",
            "Typically a controlled derate/trip on over-temperature "
            "protection rather than a catastrophic event — moderate "
            "severity by design of the protection scheme",
            source="Added by aeolus_rams — not in Section 1.6 source "
                   "table; re-review before treating as final",
        ),
        ComponentScoreDefaults(
            "Grounding/Lightning Protection", 6, 7,
            "Worn/damaged grounding brush, degraded lightning conduction "
            "path",
            "Increased risk of induced electrical/mechanical damage "
            "during a lightning strike",
            "Often no IMMEDIATE operational disruption — the real risk "
            "is latent: it raises the probability/severity of a FUTURE "
            "lightning-induced failure elsewhere (blade, electronics). "
            "Scored above 'moderate' to reflect that latent risk even "
            "though it rarely triggers a standalone SCADA event.",
            source="Added by aeolus_rams — not in Section 1.6 source "
                   "table; re-review before treating as final",
        ),
    )
}

assert set(SEVERITY_DETECTION_DEFAULTS) == set(tax.COMPONENT_NAMES), (
    "SEVERITY_DETECTION_DEFAULTS must cover every taxonomy component"
)


def with_custom_scores(
    overrides: dict[str, dict[str, int]],
) -> dict[str, ComponentScoreDefaults]:
    """Return a copy of SEVERITY_DETECTION_DEFAULTS with specific
    component Severity/Detection values overridden.

    Example
    -------
    >>> scores = with_custom_scores({"Gearbox": {"severity": 10}})
    """
    scores = dict(SEVERITY_DETECTION_DEFAULTS)
    for component, fields in overrides.items():
        if component not in scores:
            raise KeyError(f"Unknown component '{component}'")
        scores[component] = replace(scores[component], **fields)
    return scores


# ---------------------------------------------------------------------------
# FMECA table assembly
# ---------------------------------------------------------------------------

def build_fmeca_table(
    tagged_events: pd.DataFrame,
    scores: dict[str, ComponentScoreDefaults] | None = None,
    component_col: str = "component_primary",
    event_label: str = "anomaly",
    event_label_col: str = "event_label",
) -> pd.DataFrame:
    """Section 1.6 — assemble the ranked FMECA table from tagged event
    data.

    Occurrence is computed as the number of distinct anomaly-labeled
    INCIDENTS (rows) mapped to each component's primary tag — i.e. the
    corrected, `.value_counts()`-based frequency Section 1.1.2 calls for,
    aggregated at the component level rather than the free-text level (two
    different free-text descriptions for the same component both count
    toward that component's Occurrence).

    Components in the taxonomy with zero observed anomaly incidents are
    still included (with occurrence=0, RPN computed against the O=1 floor)
    so the table stays a complete 13-row picture rather than silently
    dropping components your farms simply didn't happen to log — but they
    are clearly flagged via `data_support` so they aren't mistaken for a
    confidently-scored, data-backed row.
    """
    scores = scores or SEVERITY_DETECTION_DEFAULTS

    anomalies = tagged_events[tagged_events[event_label_col] == event_label]
    real_components = anomalies[
        ~anomalies[component_col].isin([tax.UNCLASSIFIED, tax.NO_DESCRIPTION])
    ]
    occurrence_counts = real_components.groupby(component_col).size()

    rows = []
    for component in tax.COMPONENT_NAMES:
        n = int(occurrence_counts.get(component, 0))
        d = scores[component]
        o_score = occurrence_score(n)
        rpn = d.severity * o_score * d.detection
        rows.append({
            "component": component,
            "subsystem_group": tax.component_by_name(component).subsystem_group,
            "failure_cause": d.failure_cause,
            "local_effect": d.local_effect,
            "system_effect": d.system_effect,
            "distinct_incidents_observed": n,
            "severity": d.severity,
            "occurrence": o_score,
            "detection": d.detection,
            "rpn": rpn,
            "data_support": "observed" if n > 0 else "no_incidents_observed_in_data",
            "score_source": d.source,
        })

    table = pd.DataFrame(rows).sort_values("rpn", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", table.index + 1)
    return table


def unclassified_events(
    tagged_events: pd.DataFrame,
    component_col: str = "component_primary",
    event_label: str = "anomaly",
    event_label_col: str = "event_label",
    columns: tuple[str, ...] = ("farm", "event_id", "event_description",
                                 "tag_confidence", "tag_notes"),
) -> pd.DataFrame:
    """Section 1.8's closing instruction: "Read through every unclassified
    entry ... before trusting true_occurrence." Returns exactly the rows
    that still need a human's eyes, across BOTH sentinel buckets
    (keyword-fallback misses AND curated-but-ambiguous entries) — anything
    with `needs_manual_review == True`.
    """
    anomalies = tagged_events[tagged_events[event_label_col] == event_label]
    review = anomalies[anomalies["needs_manual_review"]]
    available_cols = [c for c in columns if c in review.columns]
    return review[available_cols].reset_index(drop=True)
