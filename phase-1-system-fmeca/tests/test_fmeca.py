from __future__ import annotations

import pandas as pd
import pytest

from aeolus_rams import taxonomy as tax
from aeolus_rams.fmeca import (
    occurrence_score,
    SEVERITY_DETECTION_DEFAULTS,
    with_custom_scores,
    build_fmeca_table,
    unclassified_events,
)
from aeolus_rams.component_tagger import tag_events


# ---------------------------------------------------------------------------
# Occurrence bucketing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,expected", [
    (0, 1),
    (1, 2),
    (2, 4),
    (3, 6),
    (4, 6),
    (5, 7),
    (6, 7),
    (7, 7),
    (8, 9),
    (20, 9),
])
def test_occurrence_score_buckets(n, expected):
    assert occurrence_score(n) == expected


def test_occurrence_score_is_monotonic_non_decreasing():
    scores = [occurrence_score(n) for n in range(0, 15)]
    assert all(a <= b for a, b in zip(scores, scores[1:]))


def test_occurrence_score_rejects_negative():
    with pytest.raises(ValueError):
        occurrence_score(-1)


# ---------------------------------------------------------------------------
# Severity/Detection defaults
# ---------------------------------------------------------------------------

def test_all_taxonomy_components_have_score_defaults():
    assert set(SEVERITY_DETECTION_DEFAULTS.keys()) == set(tax.COMPONENT_NAMES)


def test_score_defaults_within_1_to_10_range():
    for d in SEVERITY_DETECTION_DEFAULTS.values():
        assert 1 <= d.severity <= 10
        assert 1 <= d.detection <= 10


def test_with_custom_scores_overrides_only_target_component():
    updated = with_custom_scores({"Gearbox": {"severity": 10}})
    assert updated["Gearbox"].severity == 10
    assert updated["Gearbox"].detection == SEVERITY_DETECTION_DEFAULTS["Gearbox"].detection
    # everything else untouched
    assert updated["Converter"] == SEVERITY_DETECTION_DEFAULTS["Converter"]


def test_with_custom_scores_unknown_component_raises():
    with pytest.raises(KeyError):
        with_custom_scores({"Not A Real Component": {"severity": 5}})


# ---------------------------------------------------------------------------
# FMECA table assembly
# ---------------------------------------------------------------------------

def _tagged_fixture() -> pd.DataFrame:
    df = pd.DataFrame({
        "event_label": ["anomaly"] * 6 + ["normal"],
        "event_description": [
            "Gearbox failure", "Gearbox failure", "Gearbox failure",
            "Transformer failure",
            "another unknown gremlin",  # keep a second unclassified string for testing
            "totally unknown gremlin event",  # keyword-miss -> excluded
            None,
        ],
    })
    return tag_events(df)


def test_build_fmeca_table_has_one_row_per_taxonomy_component():
    tagged = _tagged_fixture()
    table = build_fmeca_table(tagged)
    assert len(table) == len(tax.COMPONENT_NAMES)
    assert set(table["component"]) == set(tax.COMPONENT_NAMES)


def test_build_fmeca_table_counts_real_occurrences_only():
    tagged = _tagged_fixture()
    table = build_fmeca_table(tagged)
    gearbox_row = table.loc[table["component"] == "Gearbox"].iloc[0]
    transformer_row = table.loc[table["component"] == "Transformer"].iloc[0]

    assert gearbox_row["distinct_incidents_observed"] == 3
    assert transformer_row["distinct_incidents_observed"] == 1
    # Unclassified/ambiguous entries must NOT be counted against any
    # component's occurrence.
    assert table["distinct_incidents_observed"].sum() == 4


def test_build_fmeca_table_rpn_is_product_of_s_o_d():
    tagged = _tagged_fixture()
    table = build_fmeca_table(tagged)
    for _, row in table.iterrows():
        assert row["rpn"] == row["severity"] * row["occurrence"] * row["detection"]


def test_build_fmeca_table_sorted_descending_by_rpn():
    tagged = _tagged_fixture()
    table = build_fmeca_table(tagged)
    rpns = table["rpn"].tolist()
    assert rpns == sorted(rpns, reverse=True)
    assert table["rank"].tolist() == list(range(1, len(table) + 1))


def test_build_fmeca_table_flags_unobserved_components():
    tagged = _tagged_fixture()
    table = build_fmeca_table(tagged)
    unobserved = table[table["distinct_incidents_observed"] == 0]
    assert (unobserved["data_support"] == "no_incidents_observed_in_data").all()
    observed = table[table["distinct_incidents_observed"] > 0]
    assert (observed["data_support"] == "observed").all()


def test_build_fmeca_table_respects_custom_scores():
    tagged = _tagged_fixture()
    custom = with_custom_scores({"Gearbox": {"severity": 1, "detection": 1}})
    table = build_fmeca_table(tagged, scores=custom)
    gearbox_row = table.loc[table["component"] == "Gearbox"].iloc[0]
    assert gearbox_row["severity"] == 1
    assert gearbox_row["detection"] == 1
    assert gearbox_row["rpn"] == 1 * gearbox_row["occurrence"] * 1


def test_unclassified_events_returns_only_flagged_rows():
    tagged = _tagged_fixture()
    review = unclassified_events(tagged)
    assert len(review) == 2  # "high temperature" + the gremlin event
    assert "event_description" in review.columns
