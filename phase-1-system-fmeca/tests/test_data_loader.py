from __future__ import annotations

import pandas as pd
import pytest

from aeolus_rams.data_loader import (
    CAREDataError,
    discover_farms,
    load_event_table,
    load_feature_table,
    load_dataset_file,
    load_all_event_tables,
    inventory_summary,
)


def test_discover_farms_finds_all_three(care_root):
    farms = discover_farms(care_root)
    assert set(farms.keys()) == {"A", "B", "C"}


def test_discover_farms_resolves_doubled_nesting(care_root):
    farms = discover_farms(care_root)
    # The real CARE archive nests "Wind Farm A/Wind Farm A/" — confirm the
    # resolved root points at the INNER folder, not the outer one.
    assert farms["A"].root.name == "Wind Farm A"
    assert farms["A"].root.parent.name == "Wind Farm A"


def test_discover_farms_missing_root_raises(tmp_path):
    with pytest.raises(CAREDataError):
        discover_farms(tmp_path / "does_not_exist")


def test_discover_farms_empty_root_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(CAREDataError):
        discover_farms(empty)


def test_dataset_files_sorted_numerically(care_root):
    farms = discover_farms(care_root)
    files = farms["A"].dataset_files()
    stems = [f.stem for f in files]
    assert stems == sorted(stems, key=int)


def test_load_event_table_has_expected_columns(care_root):
    farms = discover_farms(care_root)
    df = load_event_table(farms["A"])
    for col in ("event_id", "event_label", "event_start", "event_end",
                "event_description", "farm"):
        assert col in df.columns
    assert (df["farm"] == "A").all()


def test_load_event_table_parses_datetimes(care_root):
    farms = discover_farms(care_root)
    df = load_event_table(farms["A"])
    assert pd.api.types.is_datetime64_any_dtype(df["event_start"])
    assert pd.api.types.is_datetime64_any_dtype(df["event_end"])


def test_load_feature_table(care_root):
    farms = discover_farms(care_root)
    df = load_feature_table(farms["B"])
    assert "sensor_name" in df.columns
    assert (df["farm"] == "B").all()


def test_load_dataset_file_parses_timestamp(care_root):
    farms = discover_farms(care_root)
    f = farms["A"].dataset_files()[0]
    df = load_dataset_file(f)
    assert pd.api.types.is_datetime64_any_dtype(df["time_stamp"])
    assert "status_type_id" in df.columns


def test_load_dataset_file_column_subset_is_fast_and_correct(care_root):
    farms = discover_farms(care_root)
    f = farms["A"].dataset_files()[0]
    df = load_dataset_file(f, columns=["status_type_id"])
    assert list(df.columns) == ["status_type_id"]


def test_load_all_event_tables(care_root):
    farms = discover_farms(care_root)
    tables = load_all_event_tables(farms)
    assert set(tables.keys()) == {"A", "B", "C"}
    assert all(isinstance(v, pd.DataFrame) for v in tables.values())


def test_inventory_summary_reconciles(care_root):
    farms = discover_farms(care_root)
    summary = inventory_summary(farms)
    assert len(summary) == 3
    assert summary["counts_reconcile"].all()
    assert (summary["n_turbine_datasets"] == 2).all()
