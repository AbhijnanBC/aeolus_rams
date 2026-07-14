from __future__ import annotations

import pandas as pd
import pytest

from aeolus_rams.data_loader import FarmPaths
from aeolus_rams_phase2.linkage import (
    build_asset_time_ranges,
    link_events_to_assets,
    link_all_farms,
    linkage_summary,
    build_observation_end_lookup,
    extract_implied_dates,
    check_offset_consistency_per_asset,
    UNIQUE_MATCH,
    NO_MATCH,
)


def _write_file(datasets_dir, name, asset_id, t_min, t_max):
    ts = pd.date_range(t_min, t_max, freq="10min")
    df = pd.DataFrame({
        "time_stamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "asset_id": asset_id,
        "id": range(len(ts)),
        "train_test": "test",
        "status_type_id": 0,
    })
    df.to_csv(datasets_dir / name, sep=";", index=False)


@pytest.fixture
def simple_farm(tmp_path):
    root = tmp_path / "Wind Farm X" / "Wind Farm X"
    datasets_dir = root / "datasets"
    datasets_dir.mkdir(parents=True)
    farm = FarmPaths(farm_id="X", root=root)

    # Three well-separated, non-overlapping episodes on three assets.
    _write_file(datasets_dir, "0.csv", asset_id=0,
                t_min="2022-01-01 00:00", t_max="2022-01-02 00:00")
    _write_file(datasets_dir, "1.csv", asset_id=1,
                t_min="2022-02-01 00:00", t_max="2022-02-02 00:00")
    _write_file(datasets_dir, "2.csv", asset_id=2,
                t_min="2022-03-01 00:00", t_max="2022-03-02 00:00")
    return farm, datasets_dir


def _event(event_id, start, end, description=None, farm="X"):
    return {
        "farm": farm, "event_id": event_id, "event_label": "anomaly",
        "event_start": pd.Timestamp(start), "event_end": pd.Timestamp(end),
        "event_description": description,
    }


# ---------------------------------------------------------------------------
# build_asset_time_ranges
# ---------------------------------------------------------------------------

def test_build_asset_time_ranges_one_row_per_file(simple_farm):
    farm, _ = simple_farm
    ranges = build_asset_time_ranges(farm)
    assert len(ranges) == 3
    assert set(ranges["asset_id"]) == {0, 1, 2}
    assert (ranges["asset_id_is_constant"]).all()


# ---------------------------------------------------------------------------
# link_events_to_assets — unique / ambiguous / no_match
# ---------------------------------------------------------------------------

def test_unique_match(simple_farm):
    farm, _ = simple_farm
    events = pd.DataFrame([
        _event(1, "2022-01-01 06:00", "2022-01-01 12:00"),
        _event(2, "2022-02-01 06:00", "2022-02-01 12:00"),
    ])
    linked = link_events_to_assets(farm, events)
    assert list(linked["link_confidence"]) == [UNIQUE_MATCH, UNIQUE_MATCH]
    assert linked.loc[linked["event_id"] == 1, "asset_id"].iloc[0] == 0
    assert linked.loc[linked["event_id"] == 2, "asset_id"].iloc[0] == 1


def test_no_match_for_event_outside_any_window(simple_farm):
    farm, _ = simple_farm
    events = pd.DataFrame([_event(1, "2022-06-01 00:00", "2022-06-01 06:00")])
    linked = link_events_to_assets(farm, events)
    assert linked.iloc[0]["link_confidence"] == NO_MATCH
    assert pd.isna(linked.iloc[0]["asset_id"])


def test_ambiguous_match_resolved_by_tightest_span(tmp_path):
    root = tmp_path / "Wind Farm Y" / "Wind Farm Y"
    datasets_dir = root / "datasets"
    datasets_dir.mkdir(parents=True)
    farm = FarmPaths(farm_id="Y", root=root)

    # A wide file and a tight file, both fully containing the same event.
    _write_file(datasets_dir, "wide.csv", asset_id=99,
                t_min="2022-01-01 00:00", t_max="2022-01-10 00:00")
    _write_file(datasets_dir, "tight.csv", asset_id=7,
                t_min="2022-01-04 00:00", t_max="2022-01-06 00:00")

    events = pd.DataFrame([_event(1, "2022-01-05 00:00", "2022-01-05 06:00", farm="Y")])
    linked = link_events_to_assets(farm, events)

    assert linked.iloc[0]["link_confidence"] == "ambiguous_2_matches"
    assert linked.iloc[0]["asset_id"] == 7  # the tighter-span file wins


def test_link_events_to_assets_no_files_returns_no_match(tmp_path):
    root = tmp_path / "Wind Farm Z" / "Wind Farm Z"
    (root / "datasets").mkdir(parents=True)
    farm = FarmPaths(farm_id="Z", root=root)
    events = pd.DataFrame([_event(1, "2022-01-01", "2022-01-02", farm="Z")])
    linked = link_events_to_assets(farm, events)
    assert linked.iloc[0]["link_confidence"] == NO_MATCH


def test_link_all_farms_preserves_farm_column(simple_farm):
    farm, _ = simple_farm
    events_by_farm = {"X": pd.DataFrame([_event(1, "2022-01-01 06:00", "2022-01-01 12:00")])}
    linked = link_all_farms({"X": farm}, events_by_farm)
    assert (linked["farm"] == "X").all()


def test_linkage_summary_value_counts(simple_farm):
    farm, _ = simple_farm
    events = pd.DataFrame([
        _event(1, "2022-01-01 06:00", "2022-01-01 12:00"),
        _event(2, "2022-06-01 00:00", "2022-06-01 06:00"),  # no match
    ])
    linked = link_events_to_assets(farm, events)
    counts = linkage_summary(linked)
    assert counts[UNIQUE_MATCH] == 1
    assert counts[NO_MATCH] == 1


def test_build_observation_end_lookup(simple_farm):
    farm, _ = simple_farm
    lookup = build_observation_end_lookup({"X": farm})
    assert lookup[("X", 0)] == pd.Timestamp("2022-01-02 00:00")
    assert lookup[("X", 2)] == pd.Timestamp("2022-03-02 00:00")


# ---------------------------------------------------------------------------
# Offset diagnostic (Section 2.1.2)
# ---------------------------------------------------------------------------

def test_extract_implied_dates_dmy():
    dates = extract_implied_dates("anomaly until 15.01.2023 confirmed")
    assert len(dates) == 1
    assert dates[0].implied_date == pd.Timestamp("2023-01-15")


def test_extract_implied_dates_ymd():
    dates = extract_implied_dates("Failure 2023-04-05 03:30 - coupling")
    assert len(dates) == 1
    assert dates[0].implied_date == pd.Timestamp("2023-04-05")


def test_extract_implied_dates_ignores_partial_dates():
    # Day/month only, no year -> cannot support the diagnostic, must be excluded.
    assert extract_implied_dates("rectified on 23/01") == []
    assert extract_implied_dates("on the 16th in the afternoon") == []


def test_extract_implied_dates_empty_for_missing_text():
    assert extract_implied_dates(None) == []
    assert extract_implied_dates(float("nan")) == []


def test_check_offset_consistency_detects_constant_offset():
    linked = pd.DataFrame([
        {"farm": "C", "asset_id": 1, "event_id": 1,
         "event_start": pd.Timestamp("2022-01-10"),
         "event_description": "issue until 15.01.2022"},
        {"farm": "C", "asset_id": 1, "event_id": 2,
         "event_start": pd.Timestamp("2022-02-10"),
         "event_description": "issue until 15.02.2022"},
    ])
    summary = check_offset_consistency_per_asset(linked)
    row = summary[(summary["farm"] == "C") & (summary["asset_id"] == 1)].iloc[0]
    assert row["offset_days_constant"]
    assert "systematic" in row["interpretation"]


def test_check_offset_consistency_detects_varying_offset():
    linked = pd.DataFrame([
        {"farm": "C", "asset_id": 2, "event_id": 1,
         "event_start": pd.Timestamp("2022-01-10"),
         "event_description": "issue until 15.01.2022"},
        {"farm": "C", "asset_id": 2, "event_id": 2,
         "event_start": pd.Timestamp("2022-02-10"),
         "event_description": "issue until 20.02.2022"},
    ])
    summary = check_offset_consistency_per_asset(linked)
    row = summary[(summary["farm"] == "C") & (summary["asset_id"] == 2)].iloc[0]
    assert not row["offset_days_constant"]
    assert "varies" in row["interpretation"]


def test_check_offset_consistency_empty_when_no_dates():
    linked = pd.DataFrame([
        {"farm": "C", "asset_id": 1, "event_id": 1,
         "event_start": pd.Timestamp("2022-01-10"),
         "event_description": "high temperature"},
    ])
    summary = check_offset_consistency_per_asset(linked)
    assert summary.empty
