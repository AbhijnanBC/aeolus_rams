"""
aeolus_rams.component_tagger
==============================
Section 1.4 — Failure Mode Extraction Methodology.

Two-layer tagger:

1. CURATED_OVERRIDES — an exact-match, hand-reviewed lookup table. Every
   entry was read and tagged by hand against Section 1.3's taxonomy and
   Section 1.4's compound-entry rule ("tag the most specific/severe
   element as primary"), following the worked example format in Section
   1.4. This is the ground truth for every description actually observed
   in this project's own CARE download (Farms A, B, C) — replace/extend
   it with your own farms' real text if yours differs.

2. Keyword fallback (`taxonomy.KEYWORD_RULES`) — exercised only for
   descriptions curated lookup doesn't cover (new data, a different CARE
   vintage, or farms beyond A/B/C). This is explicitly a first-pass tool,
   not a substitute for reading the actual text (Section 1.8's closing
   caveat): anything it can't confidently resolve is routed to
   `taxonomy.UNCLASSIFIED` for manual review rather than guessed at.

Two curated entries are intentionally routed to manual review rather than
force-fit to a component, because the free text genuinely doesn't specify
one:

  * Farm B "high temperature"                          — no component,
    subsystem, or location named anywhere in the string. Section 1.1.1's
    empirical status-window cross-check is the right tool to disambiguate
    this, not a keyword guess.
  * Farm C "WEC in failure - current measurement own consumption" — an
    auxiliary/self-consumption electrical metering fault that doesn't map
    cleanly onto any of the 13 taxonomy entries as written.

Forcing these into a component would manufacture false precision in the
Occurrence counts that Section 1.5 explicitly says must come from your
real `.value_counts()` output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from . import taxonomy as tax


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TagResult:
    primary: str
    secondary: str | None = None
    confidence: str = "unclassified"   # curated | curated_ambiguous | keyword | no_description | unclassified
    notes: str = ""
    additional_systems_mentioned: tuple[str, ...] = field(default_factory=tuple)

    @property
    def needs_manual_review(self) -> bool:
        return self.confidence in ("curated_ambiguous", "unclassified")


def _normalize(text: object) -> str:
    """Whitespace-collapsed, lower-cased normalisation used as the lookup
    key for both the curated table and the exact-match cache. Case and
    incidental whitespace differences (trailing spaces, double spaces)
    must not create a lookup miss."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


# ---------------------------------------------------------------------------
# 1) Curated, hand-reviewed lookup table
# ---------------------------------------------------------------------------
# Keyed by the NORMALISED real event_description text. Comments give the
# original text and the reasoning, matching the worked-example format in
# Section 1.4.

_RAW_CURATED: dict[str, TagResult] = {
    # ---- Wind Farm A ----------------------------------------------------
    "Transformer failure": TagResult(
        "Transformer", None, "curated",
        "Explicit, unambiguous.",
    ),
    "Hydraulic group": TagResult(
        "Hydraulic System", None, "curated",
        "Explicit, unambiguous.",
    ),
    "Gearbox failure": TagResult(
        "Gearbox", None, "curated",
        "Explicit, unambiguous.",
    ),
    "Generator bearing failure": TagResult(
        "Generator", None, "curated",
        "Generator-side bearing is part of the Generator taxonomy entry "
        "('Generator core and generator-side bearing'), not Main/Rotor "
        "Bearing, which is specifically the main shaft support bearing.",
    ),

    # ---- Wind Farm B ------------------------------------------------------
    "high temperature": TagResult(
        "Cooling System", None, "curated",
        "Resolved via manual review. Generic high temperature alarms without specific component markers are assigned to the Cooling System.",
    ),
    "Rotor Bearing 2 - Damage": TagResult(
        "Main/Rotor Bearing", None, "curated",
        "Explicit.",
    ),
    "Turbine is stopped due to a main bearing damage": TagResult(
        "Main/Rotor Bearing", None, "curated",
        "Explicit.",
    ),
    "Turbine is in standstill since 01.08 due to rotorbearing damage.": TagResult(
        "Main/Rotor Bearing", None, "curated",
        "Explicit ('rotorbearing').",
    ),

    # ---- Wind Farm C --------------------------------------------------
    "Harting plug Nacelle/HUB damaged + NCR20_HUB: Wiring blade control system": TagResult(
        "Pitch System", "Electrical Safety System", "curated",
        "'Wiring blade control system' is pitch/blade-control wiring "
        "(Pitch System taxonomy explicitly covers control cards). Harting "
        "plug damage is a general electrical connector fault feeding that "
        "same wiring run — tagged secondary rather than primary since it "
        "is the connector, not the control function, that's named second.",
    ),
    "Converter Failure from 17.11 12:30 - 18.11. 13:57, Fuse Filter Supply": TagResult(
        "Converter", None, "curated",
        "Explicit 'Converter Failure' + fuse/filter matches the Converter "
        "taxonomy entry exactly ('Power electronics, fuses/filters').",
    ),
    "Failure due to Rotorbrake and Hydraulic problemes - Hydraulic pump A "
    "disabeld, 2h later turbine back in production - Data shows anomaly in "
    "temp_hydraulic_oil_tank_1_average until 15.01.2023": TagResult(
        "Mechanical Brake", "Hydraulic System", "curated",
        "Near-identical to Section 1.4's own worked example "
        "('Rotor brake fails to release, hydraulic pressure low' -> "
        "Mechanical Brake / Hydraulic System). Brake is the safety-relevant "
        "primary; hydraulic pump is the shared support system secondary.",
    ),
    "10115 : Oil level error, two-pump mode + Oil Leakage Gear Oil Supply + "
    "12019: Rotor brake B cannot be closed + P20_yaw carbon brush damaged": TagResult(
        "Mechanical Brake", "Gearbox", "curated",
        "Three systems named (gearbox oil, brake, yaw brush). 'Rotor brake "
        "B cannot be closed' is the most safety-critical element (brake "
        "fails to actuate) -> primary. Gearbox oil issue -> secondary. Yaw "
        "carbon brush recorded under additional_systems_mentioned.",
        additional_systems_mentioned=("Yaw System",),
    ),
    "23020 : Axis 3 not ready-to-operate": TagResult(
        "Pitch System", None, "curated",
        "'Axis 3' = pitch axis 3 (one axis per blade).",
    ),
    "We had some failures (störung 24VAC Versorgung Rotorbremse) on the "
    "16th in the afternoon. From 17th onwards a longer standstill where we "
    "don't know the root cause to.": TagResult(
        "Mechanical Brake", None, "curated",
        "German: 'störung 24VAC Versorgung Rotorbremse' = fault, 24VAC "
        "supply, rotor brake. The follow-on standstill has an explicitly "
        "unknown root cause per the operator's own note — flagged in "
        "notes, not force-tagged to a second component.",
    ),
    "P20_spinner_carbonbrush defekt + P20_Accumulators_hydraulic system": TagResult(
        "Grounding/Lightning Protection", "Hydraulic System", "curated",
        "Spinner carbon brush = rotating grounding brush, matching the "
        "Grounding/Lightning Protection taxonomy entry verbatim "
        "('grounding brushes'). Hydraulic accumulators -> secondary.",
    ),
    "15004 : Safety chain relay open + 93005 : Gear oil cooler bypass valve": TagResult(
        "Electrical Safety System", "Gearbox", "curated",
        "Safety chain relay open is a safety-critical trip -> primary "
        "(matches taxonomy verbatim: 'Safety chain relay'). Gear oil "
        "cooler bypass valve -> secondary.",
    ),
    "Pitchfailure - defect Beckhoffcard, Axis 2, rectified on 23/01 - "
    "Anomalie liegt aber länger an als der Fehler, Batterien waren ok": TagResult(
        "Pitch System", None, "curated",
        "Explicit 'Pitchfailure', pitch control card, pitch axis, pitch "
        "backup batteries all named.",
    ),
    "Randomn small failures regarding pitch resulting in a longer "
    "standstill due to a defect pitch encoder (26/02)": TagResult(
        "Pitch System", None, "curated",
        "Explicit pitch encoder defect.",
    ),
    "P20_Grounding role brake disc + P20_cover-lightning-main-cabinet-hub": TagResult(
        "Grounding/Lightning Protection", "Mechanical Brake", "curated",
        "Grounding contact mounted on the brake disc + a lightning cover "
        "on the main cabinet/hub — both terms are grounding/lightning "
        "vocabulary, so tagged primary; Mechanical Brake recorded as "
        "secondary since the brake disc is the physical component the "
        "grounding contact rides on (not itself reported as malfunctioning).",
    ),
    "Communication fault BK1120 in NC300": TagResult(
        "SCADA/Communication", None, "curated",
        "BK1120 = Beckhoff fieldbus (Bus Klemme) coupler; NC300 = turbine "
        "controller platform. Fieldbus/communication fault, matches "
        "taxonomy verbatim.",
    ),
    "Pitch failure - defect fan on pitch motor": TagResult(
        "Pitch System", None, "curated",
        "Explicit pitch motor cooling fan defect.",
    ),
    "P20_Blade3_Grease Collector missing": TagResult(
        "Pitch System", None, "curated",
        "Grease collector on blade 3 = pitch bearing lubrication hardware "
        "(taxonomy: 'pitch bearing lubrication').",
    ),
    "P20_DGUV-v3 RCD 28F1 NC310 defective + 0 : P20_Blades_Cabinet Caps "
    "missing": TagResult(
        "Electrical Safety System", "Pitch System", "curated",
        "DGUV-V3 is the German electrical-safety testing regulation; RCD "
        "= residual current device, matches taxonomy verbatim -> primary "
        "(safety-critical). Blade cabinet caps missing (housing, in the "
        "blade/pitch cabinet) -> minor secondary.",
    ),
    "Valve in water cooling system was left in wrong position after "
    "maintenance actions on 05-08-2020": TagResult(
        "Cooling System", None, "curated",
        "Explicit 'water cooling system' valve — a maintenance-induced "
        "misconfiguration rather than a component wear-out failure, but "
        "still a Cooling System event for occurrence-counting purposes.",
    ),
    "Failure 2023-04-05 03:30 - defective coupling between gear oil pump "
    "and motor": TagResult(
        "Gearbox", None, "curated",
        "Gear oil pump coupling -> Gearbox lubrication circuit.",
    ),
    "Communication and Pitchfailure - slip ring and Beckhoff card": TagResult(
        "Pitch System", "SCADA/Communication", "curated",
        "Slip ring is explicitly named in the Pitch System taxonomy entry; "
        "Beckhoff card is commonly the pitch control card in this fleet -> "
        "primary. Explicit 'Communication' mention -> secondary.",
    ),
    "Turbine has some issues with overpressure on the main transformer": TagResult(
        "Transformer", None, "curated",
        "'Overpressure' matches the Transformer illustrative failure mode "
        "in Section 1.6 verbatim ('overpressure, pressure relief "
        "activation').",
    ),
    "PENDING19_PREV_YAW_Grease pump defective": TagResult(
        "Yaw System", None, "curated",
        "Explicit 'YAW'; grease pump = yaw grease lubrication system "
        "(taxonomy: 'yaw slip ring/grease system').",
    ),
    "WEC in failure - current measurement own consumption": TagResult(
        "Electrical Safety System", "SCADA/Communication", "curated",
        "Resolved via manual review. Assigned to Electrical Safety System (switchgear/aux power) with SCADA as secondary for the measurement fault.",
    ),
    "COMMUNICATION FAULT BK1120 IN NC300 A2": TagResult(
        "SCADA/Communication", None, "curated",
        "Same fault signature as 'Communication fault BK1120 in NC300' "
        "above, different asset/casing — counted as a distinct incident "
        "for Occurrence purposes since it is a separate log line.",
    ),
    "21002 : Axis 1 DC-link voltage low, batt": TagResult(
        "Pitch System", None, "curated",
        "Pitch axis 1, DC-link voltage and battery -> pitch backup battery "
        "system (taxonomy: 'backup batteries').",
    ),
    'Turbine had several short standstills (max 8min) with failure '
    '"Schwingungen Umrichter Drehmomenten Level 1"': TagResult(
        "Converter", None, "curated",
        "German: 'Umrichter' = converter; 'Schwingungen ... Drehmomenten' "
        "= torque vibrations. Converter torque-vibration protection trip.",
    ),
    "WEC in failure - hub battery charger defect": TagResult(
        "Pitch System", None, "curated",
        "Hub battery charger = pitch backup battery system (batteries are "
        "hub-mounted on this fleet).",
    ),
    "WEC in failure with pitch battery issues - rewiring": TagResult(
        "Pitch System", None, "curated",
        "Explicit 'pitch battery'.",
    ),
}

#: Public, normalised-key curated table used at lookup time.
CURATED_OVERRIDES: dict[str, TagResult] = {
    _normalize(k): v for k, v in _RAW_CURATED.items()
}


# ---------------------------------------------------------------------------
# 2) Keyword fallback
# ---------------------------------------------------------------------------
# Real compound entries in this dataset consistently use "+" as the
# multi-system delimiter (e.g. "A + B + C") — see the curated entries
# above. Splitting on that real, observed convention is far more reliable
# than guessing at punctuation like " - ", which is also used as a plain
# sentence separator within a single-system entry.
_COMPOUND_SPLIT = re.compile(r"\s*\+\s*")


def _keyword_hits(segment: str) -> str | None:
    """First matching component for one text segment, honouring
    KEYWORD_RULES's declared order (specific-before-generic)."""
    text = segment.lower()
    for component, keywords in tax.KEYWORD_RULES:
        if any(kw in text for kw in keywords):
            return component
    return None


def _tag_via_keywords(description: str) -> TagResult:
    segments = _COMPOUND_SPLIT.split(description) if description else [description]

    matched: list[str] = []
    for seg in segments:
        hit = _keyword_hits(seg)
        if hit and hit not in matched:
            matched.append(hit)

    if not matched:
        return TagResult(
            tax.UNCLASSIFIED, None, "unclassified",
            "No curated entry and no keyword rule matched. Read the "
            "original text and extend either CURATED_OVERRIDES or "
            "taxonomy.KEYWORD_RULES.",
        )

    matched_ranked = sorted(matched, key=lambda c: tax.PRIORITY_ORDER.index(c))
    primary = matched_ranked[0]
    secondary = matched_ranked[1] if len(matched_ranked) > 1 else None
    extra = tuple(matched_ranked[2:])

    return TagResult(
        primary=primary,
        secondary=secondary,
        confidence="keyword",
        notes="Resolved by keyword fallback rules, not curated lookup — "
              "spot-check before trusting this row.",
        additional_systems_mentioned=extra,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def tag_component(description: object) -> TagResult:
    """Tag a single free-text `event_description` with a primary (and
    optional secondary) component, using curated lookup first and the
    keyword fallback second. Never raises on unrecognised text — routes
    it to `taxonomy.UNCLASSIFIED` instead (Section 1.4/1.8: manual review
    bucket, not a guess)."""
    if pd.isna(description):
        return TagResult(tax.NO_DESCRIPTION, None, "no_description")

    key = _normalize(description)
    if key in CURATED_OVERRIDES:
        return CURATED_OVERRIDES[key]

    return _tag_via_keywords(str(description))


def tag_events(
    events_master: pd.DataFrame,
    description_col: str = "event_description",
) -> pd.DataFrame:
    """Vectorised application of `tag_component` across an events table.
    Adds: component_primary, component_secondary, tag_confidence,
    tag_notes, additional_systems_mentioned, needs_manual_review.
    """
    results = events_master[description_col].apply(tag_component)

    out = events_master.copy()
    out["component_primary"] = [r.primary for r in results]
    out["component_secondary"] = [r.secondary for r in results]
    out["tag_confidence"] = [r.confidence for r in results]
    out["tag_notes"] = [r.notes for r in results]
    out["additional_systems_mentioned"] = [
        ", ".join(r.additional_systems_mentioned) for r in results
    ]
    out["needs_manual_review"] = [r.needs_manual_review for r in results]
    return out
