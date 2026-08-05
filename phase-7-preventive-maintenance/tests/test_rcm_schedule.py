"""Tests for rcm_schedule.py."""
from __future__ import annotations

import pytest

from aeolus_rams_phase7.rcm_schedule import (
    build_rcm_schedule, build_age_replacement_table,
    build_failure_finding_table,
)


EXPECTED_COMPONENTS = {
    "Main/Rotor Bearing", "Gearbox", "Generator",
    "Electrical Safety System (G3b Safety Relay)", "Pitch System",
    "Hydraulic System", "Mechanical Brake", "Transformer", "Yaw System",
    "SCADA/Communication", "Converter", "Cooling System",
    "Grounding/Lightning Protection",
}


def test_rcm_schedule_has_13_rows():
    """RCM schedule must cover all 13 AEOLUS components."""
    df = build_rcm_schedule()
    assert len(df) == 13, f"Expected 13 rows, got {len(df)}"


def test_rcm_schedule_covers_all_components():
    """Every expected component must appear in the schedule."""
    df = build_rcm_schedule()
    in_schedule = set(df["component"].tolist())
    for expected in EXPECTED_COMPONENTS:
        # Allow partial match (names may be slightly different)
        found = any(expected in s or s in expected for s in in_schedule)
        assert found, f"Component '{expected}' not found in schedule. Got: {in_schedule}"


def test_rcm_categories_are_valid():
    """All rcm_category values must be from the known set."""
    valid = {"AGE_REPLACEMENT", "TIME_DIRECTED", "CONDITION_BASED",
             "FAILURE_FINDING", "RUN_TO_FAILURE"}
    df = build_rcm_schedule()
    for cat in df["rcm_category"]:
        assert cat in valid, f"Unknown category: {cat}"


def test_G3b_is_failure_finding():
    """The safety relay must be classified as FAILURE_FINDING."""
    df = build_rcm_schedule()
    g3b_row = df[df["component"].str.contains("Safety Relay|G3b", na=False)]
    assert not g3b_row.empty
    assert g3b_row.iloc[0]["rcm_category"] == "FAILURE_FINDING"


def test_pitch_is_condition_based():
    """Pitch System β < 1 → must be CONDITION_BASED."""
    df = build_rcm_schedule()
    pitch_row = df[df["component"].str.contains("Pitch", na=False)]
    assert not pitch_row.empty
    assert pitch_row.iloc[0]["rcm_category"] == "CONDITION_BASED"


def test_G3b_tau_required_is_approximately_27d():
    """G3b τ_required must be approximately 26–27 days (SIL-2 target)."""
    df = build_rcm_schedule()
    g3b_row = df[df["component"].str.contains("Safety Relay|G3b", na=False)].iloc[0]
    tau = g3b_row["tau_required_days"]
    assert tau is not None and not __import__("math").isnan(float(tau))
    assert 20 <= float(tau) <= 35, (
        f"τ_required = {float(tau):.1f}d — expected 20–35d for SIL-2 at λ_G3b"
    )


def test_no_null_trigger_rules():
    """All rows must have a non-empty trigger rule."""
    df = build_rcm_schedule()
    for _, row in df.iterrows():
        assert row["trigger_rule"] and len(str(row["trigger_rule"])) > 10, (
            f"Empty or missing trigger_rule for {row['component']}"
        )


def test_age_replacement_table_content():
    """Age-replacement table must have 3 rows with expected components."""
    ar = build_age_replacement_table()
    assert len(ar) == 3
    assert "Main/Rotor Bearing" in ar["component"].values
    assert "Gearbox" in ar["component"].values
    assert "Generator" in ar["component"].values


def test_bearing_exceeds_design_life_flag():
    """Generator should be flagged as exceeding design life."""
    ar = build_age_replacement_table()
    gen = ar[ar["component"] == "Generator"].iloc[0]
    assert gen["exceeds_design_life"] is True or gen["exceeds_design_life"] == True