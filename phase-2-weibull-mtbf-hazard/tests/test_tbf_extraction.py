from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aeolus_rams_phase2.tbf_extraction import (
    extract_tbf,
    extract_tbf_all_components,
    uncensored_tbf_days,
    tbf_summary,
)


def _linked_row(farm, asset_id, event_start, component="Gearbox", label="anomaly"):
    return {
        "farm": farm, "asset_id": asset_id, "event_start": pd.Timestamp(event_start),
        "component_primary": component, "event_label": label,
    }


def test_extract_tbf_single_asset_two_intervals():
    linked = pd.DataFrame([
        _linked_row("A", 0, "2022-01-01"),
        _linked_row("A", 0, "2022-01-11"),   # +10 days
        _linked_row("A", 0, "2022-01-26"),   # +15 days
    ])
    obs_end = {("A", 0): pd.Timestamp("2022-02-01")}  # +6 more days, censored
    tbf = extract_tbf(linked, "Gearbox", obs_end)

    assert len(tbf) == 3  # 2 uncensored + 1 censored
    uncensored = tbf[~tbf["censored"]].sort_values("interval_start")
    assert list(uncensored["tbf_days"]) == pytest.approx([10.0, 15.0])
    censored = tbf[tbf["censored"]]
    assert len(censored) == 1
    assert censored.iloc[0]["tbf_days"] == pytest.approx(6.0)


def test_extract_tbf_no_censored_row_if_no_time_remains():
    linked = pd.DataFrame([_linked_row("A", 0, "2022-01-01")])
    obs_end = {("A", 0): pd.Timestamp("2022-01-01")}  # observation ends exactly at the failure
    tbf = extract_tbf(linked, "Gearbox", obs_end)
    assert tbf.empty  # no intervals at all: 1 incident, no follow-on time


def test_extract_tbf_groups_by_farm_and_asset_separately():
    # Same asset_id (0) on two DIFFERENT farms must not be merged.
    linked = pd.DataFrame([
        _linked_row("A", 0, "2022-01-01"),
        _linked_row("A", 0, "2022-01-11"),
        _linked_row("B", 0, "2022-06-01"),
        _linked_row("B", 0, "2022-06-21"),
    ])
    obs_end = {("A", 0): pd.Timestamp("2022-01-11"), ("B", 0): pd.Timestamp("2022-06-21")}
    tbf = extract_tbf(linked, "Gearbox", obs_end)

    farm_a_tbf = tbf[tbf["farm"] == "A"]["tbf_days"].tolist()
    farm_b_tbf = tbf[tbf["farm"] == "B"]["tbf_days"].tolist()
    assert farm_a_tbf == pytest.approx([10.0])
    assert farm_b_tbf == pytest.approx([20.0])


def test_extract_tbf_filters_by_component_and_label():
    linked = pd.DataFrame([
        _linked_row("A", 0, "2022-01-01", component="Gearbox", label="anomaly"),
        _linked_row("A", 0, "2022-01-11", component="Yaw System", label="anomaly"),
        _linked_row("A", 0, "2022-01-21", component="Gearbox", label="normal"),
    ])
    tbf = extract_tbf(linked, "Gearbox", {})
    # Only 1 anomaly+Gearbox row survives filtering -> 0 intervals possible.
    assert tbf.empty


def test_extract_tbf_drops_unlinked_rows():
    linked = pd.DataFrame([
        _linked_row("A", 0, "2022-01-01"),
        _linked_row("A", None, "2022-01-11"),
    ])
    tbf = extract_tbf(linked, "Gearbox", {("A", 0): pd.Timestamp("2022-01-01")})
    # Only 1 valid row remains after dropping the unlinked one -> no intervals.
    assert tbf.empty


def test_extract_tbf_missing_observation_end_omits_final_interval():
    linked = pd.DataFrame([
        _linked_row("A", 0, "2022-01-01"),
        _linked_row("A", 0, "2022-01-11"),
    ])
    tbf = extract_tbf(linked, "Gearbox", {})  # no obs_end entry at all
    assert len(tbf) == 1  # the one uncensored interval, no censored tail
    assert not tbf.iloc[0]["censored"]


def test_extract_tbf_all_components_concatenates():
    linked = pd.DataFrame([
        _linked_row("A", 0, "2022-01-01", component="Gearbox"),
        _linked_row("A", 0, "2022-01-11", component="Gearbox"),
        _linked_row("A", 1, "2022-02-01", component="Yaw System"),
        _linked_row("A", 1, "2022-02-11", component="Yaw System"),
    ])
    obs_end = {("A", 0): pd.Timestamp("2022-01-11"), ("A", 1): pd.Timestamp("2022-02-11")}
    tbf = extract_tbf_all_components(linked, ["Gearbox", "Yaw System"], obs_end)
    assert set(tbf["component"]) == {"Gearbox", "Yaw System"}


def test_uncensored_tbf_days_excludes_censored():
    tbf = pd.DataFrame([
        {"component": "Gearbox", "tbf_days": 10.0, "censored": False},
        {"component": "Gearbox", "tbf_days": 999.0, "censored": True},
        {"component": "Yaw System", "tbf_days": 5.0, "censored": False},
    ])
    result = uncensored_tbf_days(tbf, component="Gearbox")
    assert list(result) == [10.0]


def test_tbf_summary_counts_censored_and_uncensored():
    tbf = pd.DataFrame([
        {"component": "Gearbox", "asset_id": 0, "tbf_days": 10.0, "censored": False},
        {"component": "Gearbox", "asset_id": 0, "tbf_days": 5.0, "censored": True},
        {"component": "Gearbox", "asset_id": 1, "tbf_days": 20.0, "censored": False},
    ])
    summary = tbf_summary(tbf)
    row = summary[summary["component"] == "Gearbox"].iloc[0]
    assert row["n_uncensored"] == 2
    assert row["n_censored"] == 1
    assert row["n_assets"] == 2


def test_tbf_summary_empty_input():
    summary = tbf_summary(pd.DataFrame())
    assert summary.empty
    assert list(summary.columns) == ["component", "n_uncensored", "n_censored", "n_assets"]
