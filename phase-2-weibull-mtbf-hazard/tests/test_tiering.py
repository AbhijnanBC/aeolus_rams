from __future__ import annotations

import pandas as pd
import pytest

from aeolus_rams_phase2 import config
from aeolus_rams_phase2.tiering import (
    assign_tier, tier_table, components_by_tier, tier_summary_counts,
    load_fmeca_table,
)


@pytest.mark.parametrize("n,expected", [
    (0, "C"), (1, "C"), (4, "C"),
    (5, "B"), (6, "B"), (7, "B"),
    (8, "A"), (11, "A"), (100, "A"),
])
def test_assign_tier_boundaries(n, expected):
    assert assign_tier(n) == expected


def test_assign_tier_rejects_negative():
    with pytest.raises(ValueError):
        assign_tier(-1)


def test_assign_tier_respects_custom_thresholds():
    custom = config.TierThresholds(min_a=10, min_b=3)
    assert assign_tier(9, custom) == "B"
    assert assign_tier(10, custom) == "A"
    assert assign_tier(2, custom) == "C"


def _fmeca_df():
    return pd.DataFrame({
        "component": ["Pitch System", "Hydraulic System", "Cooling System"],
        "distinct_incidents_observed": [11, 6, 0],
        "rpn": [360, 196, 30],
    })


def test_tier_table_adds_tier_column():
    out = tier_table(_fmeca_df())
    assert list(out["tier"]) == ["A", "B", "C"]
    assert "tier_description" in out.columns


def test_components_by_tier_groups_correctly():
    tiered = tier_table(_fmeca_df())
    groups = components_by_tier(tiered)
    assert groups["A"] == ["Pitch System"]
    assert groups["B"] == ["Hydraulic System"]
    assert groups["C"] == ["Cooling System"]


def test_tier_summary_counts():
    tiered = tier_table(_fmeca_df())
    counts = tier_summary_counts(tiered)
    assert counts["A"] == 1
    assert counts["B"] == 1
    assert counts["C"] == 1


def test_load_fmeca_table_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_fmeca_table(tmp_path / "does_not_exist.csv")


def test_load_fmeca_table_wrong_schema_raises(tmp_path):
    bad = tmp_path / "fmeca_table.csv"
    pd.DataFrame({"component": ["X"], "rpn": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_fmeca_table(bad)


def test_load_fmeca_table_reads_real_file(tmp_path):
    path = tmp_path / "fmeca_table.csv"
    _fmeca_df().to_csv(path, index=False)
    df = load_fmeca_table(path)
    assert len(df) == 3
