from __future__ import annotations

import pandas as pd
import pytest

from aeolus_rams.data_loader import discover_farms, load_all_event_tables
from aeolus_rams.validation import (
    confirm_status_legend,
    empirical_status_check_around_events,
    event_description_value_counts,
    run_step0_checks,
)


def test_confirm_status_legend_no_unknown_codes(care_root):
    farms = discover_farms(care_root)
    reports = confirm_status_legend(farms, sample_files_per_farm=None)
    for farm_id, report in reports.items():
        # Synthetic fixture only ever emits codes 0,1,3,4,5, all of which
        # are documented in config.STATUS_LEGEND.
        assert report.unknown_ids == set(), f"farm {farm_id}: {report.unknown_ids}"
        assert report.is_clean


def test_confirm_status_legend_flags_unknown_code(care_root, monkeypatch):
    import aeolus_rams.config as config

    farms = discover_farms(care_root)
    # Temporarily shrink the documented legend so a real, observed code
    # (status 0) becomes "unknown" — proves the cross-check actually
    # looks at the data rather than always passing.
    monkeypatch.setattr(config, "STATUS_LEGEND", {
        k: v for k, v in config.STATUS_LEGEND.items() if k != 0
    })
    reports = confirm_status_legend(farms, sample_files_per_farm=1)
    assert 0 in reports["A"].unknown_ids
    assert not reports["A"].is_clean


def test_confirm_status_legend_sampling_reduces_files_scanned(care_root):
    farms = discover_farms(care_root)
    full = confirm_status_legend(farms, sample_files_per_farm=None)
    sampled = confirm_status_legend(farms, sample_files_per_farm=1)
    # Sampling 1 of 2 files must see <= the row counts the exhaustive scan sees.
    total_full = sum(full["A"].counts.values())
    total_sampled = sum(sampled["A"].counts.values())
    assert total_sampled <= total_full


def test_event_description_value_counts_counts_repeats(care_root):
    farms = discover_farms(care_root)
    event_tables = load_all_event_tables(farms)
    counts = event_description_value_counts(event_tables)
    # The fixture deliberately repeats the first description twice for
    # every farm (see conftest._make_event_info) — the WHOLE POINT of
    # Section 1.1.2 is that this must show up as count=2, not be
    # collapsed away like `.unique()` would.
    farm_a_counts = counts["A"]
    assert farm_a_counts["Transformer failure"] == 2


def test_event_description_value_counts_excludes_normal_events(care_root):
    farms = discover_farms(care_root)
    event_tables = load_all_event_tables(farms)
    counts = event_description_value_counts(event_tables)
    # Normal-labeled events have NaN descriptions in the fixture and must
    # not appear in the anomaly-only value_counts output.
    assert counts["A"].index.isna().sum() == 0


def test_empirical_status_check_returns_tidy_frame(care_root):
    farms = discover_farms(care_root)
    events = load_all_event_tables(farms)["A"]
    result = empirical_status_check_around_events(farms["A"], events)
    assert isinstance(result, pd.DataFrame)
    if not result.empty:
        for col in ("farm", "event_id", "status_id", "count_in_window"):
            assert col in result.columns


def test_run_step0_checks_bundles_everything(care_root):
    farms = discover_farms(care_root)
    event_tables = load_all_event_tables(farms)
    report = run_step0_checks(farms, event_tables, sample_files_per_farm=1)
    assert report.all_inventory_reconciles
    assert set(report.status_legend.keys()) == {"A", "B", "C"}
    assert set(report.description_counts.keys()) == {"A", "B", "C"}
    assert len(report.summary_lines()) > 0
