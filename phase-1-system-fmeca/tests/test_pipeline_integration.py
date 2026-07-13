from __future__ import annotations

import pandas as pd
import pytest

from aeolus_rams import taxonomy as tax
from aeolus_rams.pipeline import run_phase1, main


def test_run_phase1_end_to_end_in_memory(care_root):
    result = run_phase1(care_root, write_outputs=False)

    assert len(result.inventory) == 3
    assert result.inventory["counts_reconcile"].all()

    assert not result.tagged_events.empty
    assert "component_primary" in result.tagged_events.columns

    assert len(result.fmeca_table) == len(tax.COMPONENT_NAMES)
    assert result.fmeca_table["rpn"].is_monotonic_decreasing

    assert isinstance(result.report_markdown, str)
    assert "AEOLUS-RAMS — Phase 1 Report" in result.report_markdown
    assert "Definition of Done" in result.report_markdown


def test_run_phase1_writes_expected_files(care_root, tmp_path):
    out_dir = tmp_path / "phase1_out"
    run_phase1(care_root, output_dir=out_dir, sample_files_per_farm=1)

    assert (out_dir / "tagged_events.csv").exists()
    assert (out_dir / "fmeca_table.csv").exists()
    assert (out_dir / "inventory_summary.csv").exists()
    assert (out_dir / "phase1_report.md").exists()

    tagged = pd.read_csv(out_dir / "tagged_events.csv")
    assert "component_primary" in tagged.columns

    fmeca = pd.read_csv(out_dir / "fmeca_table.csv")
    assert len(fmeca) == len(tax.COMPONENT_NAMES)
    assert list(fmeca["rank"]) == list(range(1, len(fmeca) + 1))


def test_run_phase1_resolves_all_fixture_events(care_root):
    # Because we manually resolved 'high temperature' and 'WEC in failure...',
    # the synthetic dataset used in testing should now result in zero 
    # unclassified events.
    result = run_phase1(care_root, write_outputs=False)
    assert len(result.unclassified) == 0


def test_run_phase1_occurrence_reflects_repeated_descriptions(care_root):
    # conftest repeats each farm's first description twice — confirm that
    # signal survives tagging + aggregation into the FMECA table's
    # distinct_incidents_observed for the corresponding component.
    result = run_phase1(care_root, write_outputs=False)
    transformer_row = result.fmeca_table.loc[
        result.fmeca_table["component"] == "Transformer"
    ].iloc[0]
    assert transformer_row["distinct_incidents_observed"] == 2


def test_cli_main_returns_zero_on_success(care_root, tmp_path, capsys):
    out_dir = tmp_path / "cli_out"
    rc = main([
        "--data-root", str(care_root),
        "--output-dir", str(out_dir),
        "--sample-files", "1",
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "PHASE 1 COMPLETE" in captured.out
    assert (out_dir / "fmeca_table.csv").exists()


def test_cli_main_returns_nonzero_on_bad_data_root(tmp_path, capsys):
    rc = main(["--data-root", str(tmp_path / "nope")])
    assert rc == 1


def test_run_phase1_is_idempotent(care_root, tmp_path):
    out_dir = tmp_path / "run1"
    r1 = run_phase1(care_root, output_dir=out_dir, sample_files_per_farm=1)
    r2 = run_phase1(care_root, output_dir=out_dir, sample_files_per_farm=1)
    pd.testing.assert_frame_equal(
        r1.fmeca_table.reset_index(drop=True),
        r2.fmeca_table.reset_index(drop=True),
    )
