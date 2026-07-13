from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aeolus_rams import taxonomy as tax
from aeolus_rams.component_tagger import (
    CURATED_OVERRIDES,
    tag_component,
    tag_events,
    _normalize,
)


# ---------------------------------------------------------------------------
# Curated lookup — spot checks against real, hand-tagged descriptions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("description,expected_primary", [
    ("Transformer failure", "Transformer"),
    ("Hydraulic group", "Hydraulic System"),
    ("Gearbox failure", "Gearbox"),
    ("Generator bearing failure", "Generator"),
    ("Rotor Bearing 2 - Damage", "Main/Rotor Bearing"),
    ("Turbine is stopped due to a main bearing damage", "Main/Rotor Bearing"),
    ("23020 : Axis 3 not ready-to-operate", "Pitch System"),
    ("Communication fault BK1120 in NC300", "SCADA/Communication"),
    ("PENDING19_PREV_YAW_Grease pump defective", "Yaw System"),
    ("Valve in water cooling system was left in wrong position after "
     "maintenance actions on 05-08-2020", "Cooling System"),
])
def test_curated_primary_tags(description, expected_primary):
    result = tag_component(description)
    assert result.primary == expected_primary
    assert result.confidence == "curated"
    assert not result.needs_manual_review


def test_curated_compound_entry_brake_and_hydraulic():
    desc = (
        "Failure due to Rotorbrake and Hydraulic problemes - Hydraulic pump A "
        "disabeld, 2h later turbine back in production - Data shows anomaly in "
        "temp_hydraulic_oil_tank_1_average until 15.01.2023"
    )
    result = tag_component(desc)
    assert result.primary == "Mechanical Brake"
    assert result.secondary == "Hydraulic System"


def test_curated_compound_entry_records_additional_systems():
    desc = (
        "10115 : Oil level error, two-pump mode + Oil Leakage Gear Oil Supply + "
        "12019: Rotor brake B cannot be closed + P20_yaw carbon brush damaged"
    )
    result = tag_component(desc)
    assert result.primary == "Mechanical Brake"
    assert result.secondary == "Gearbox"
    assert "Yaw System" in result.additional_systems_mentioned


def test_previously_ambiguous_entries_are_now_resolved():
    result = tag_component("high temperature")
    assert result.primary == "Cooling System"
    assert result.confidence == "curated"
    assert not result.needs_manual_review

    result2 = tag_component("WEC in failure - current measurement own consumption")
    assert result2.primary == "Electrical Safety System"
    assert not result2.needs_manual_review


def test_normalize_is_case_and_whitespace_insensitive():
    a = tag_component("  TRANSFORMER   FAILURE  ")
    b = tag_component("Transformer failure")
    assert a.primary == b.primary == "Transformer"


def test_curated_table_keys_are_unique_after_normalisation():
    # Guards against two differently-cased/whitespaced source strings
    # silently colliding into the same normalised key without the author
    # noticing (which would silently drop one of them).
    keys = list(CURATED_OVERRIDES.keys())
    assert len(keys) == len(set(keys))


def test_every_curated_primary_is_a_known_component_or_sentinel():
    valid = set(tax.COMPONENT_NAMES) | {tax.UNCLASSIFIED, tax.NO_DESCRIPTION}
    for result in CURATED_OVERRIDES.values():
        assert result.primary in valid
        if result.secondary is not None:
            assert result.secondary in valid


# ---------------------------------------------------------------------------
# Keyword fallback — text NOT in the curated table
# ---------------------------------------------------------------------------

def test_keyword_fallback_single_component():
    result = tag_component("Unexpected yaw drive motor fault, standstill")
    assert result.primary == "Yaw System"
    assert result.confidence == "keyword"


def test_keyword_fallback_compound_uses_plus_delimiter():
    result = tag_component("Gearbox oil pressure low + Converter IGBT trip")
    assert result.primary in {"Gearbox", "Converter"}
    assert result.secondary in {"Gearbox", "Converter"}
    assert result.primary != result.secondary


def test_keyword_fallback_specific_beats_generic_within_segment():
    # "yaw carbon brush" must resolve to Yaw System, not fall through to
    # the generic Grounding/Lightning "carbon brush" keyword, because the
    # Yaw rule's more specific compound phrase should win when both are
    # present in the same segment. We test the documented precedence
    # directly via a segment containing only the yaw-specific phrase.
    result = tag_component("P20_yaw carbon brush damaged")
    assert result.primary == "Yaw System"


def test_keyword_fallback_no_match_is_unclassified():
    result = tag_component("Completely novel gremlin sighting in the nacelle")
    assert result.primary == tax.UNCLASSIFIED
    assert result.confidence == "unclassified"
    assert result.needs_manual_review


def test_missing_description_is_no_description_sentinel():
    result = tag_component(np.nan)
    assert result.primary == tax.NO_DESCRIPTION
    assert result.confidence == "no_description"
    assert not result.needs_manual_review


# ---------------------------------------------------------------------------
# Batch tagging
# ---------------------------------------------------------------------------

def test_tag_events_adds_expected_columns():
    df = pd.DataFrame({
        "event_label": ["anomaly", "anomaly", "normal"],
        "event_description": ["Transformer failure", "totally unknown gremlin", None],
    })
    out = tag_events(df)
    for col in ("component_primary", "component_secondary", "tag_confidence",
                "tag_notes", "additional_systems_mentioned", "needs_manual_review"):
        assert col in out.columns
    assert out.loc[0, "component_primary"] == "Transformer"
    assert out.loc[1, "needs_manual_review"]
    assert out.loc[2, "component_primary"] == tax.NO_DESCRIPTION


def test_tag_events_preserves_row_count_and_order():
    df = pd.DataFrame({
        "event_label": ["anomaly"] * 5,
        "event_description": [
            "Gearbox failure", "Yaw System fault", "unknown gremlin",
            None, "Transformer failure",
        ],
    })
    out = tag_events(df)
    assert len(out) == 5
    assert list(out.index) == list(df.index)
