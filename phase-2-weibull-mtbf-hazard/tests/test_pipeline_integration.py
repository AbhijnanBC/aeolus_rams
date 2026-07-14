from __future__ import annotations

import pandas as pd
import pytest

from aeolus_rams_phase2.pipeline import run_phase2, main
from aeolus_rams_phase2.linkage import UNIQUE_MATCH


def test_run_phase2_end_to_end_in_memory(phase2_project):
    data_root, phase1_dir, ground_truth = phase2_project
    result = run_phase2(data_root, phase1_dir, write_outputs=False, n_boot=200)

    # Perfect linkage on well-separated synthetic episodes.
    assert set(result.linkage_counts.index) == {UNIQUE_MATCH}
    assert result.linkage_counts[UNIQUE_MATCH] == len(ground_truth)


def test_run_phase2_linkage_recovers_true_asset_ids(phase2_project):
    data_root, phase1_dir, ground_truth = phase2_project
    result = run_phase2(data_root, phase1_dir, write_outputs=False, n_boot=200)

    merged = result.linked_events.merge(
        ground_truth[["event_id", "_true_asset_id"]], on="event_id",
    )
    assert (merged["asset_id"] == merged["_true_asset_id"]).all()


def test_run_phase2_pitch_system_is_tier_a(phase2_project):
    data_root, phase1_dir, _ = phase2_project
    result = run_phase2(data_root, phase1_dir, write_outputs=False, n_boot=200)

    assert "Pitch System" in result.tier_a_fits
    fit = result.tier_a_fits["Pitch System"]
    assert fit.n_used == 9  # 8 intervals on asset0 + 1 on asset1
    assert "Pitch System" in result.tier_a_bootstrap


def test_run_phase2_gearbox_downgrades_from_a_to_b(phase2_project):
    data_root, phase1_dir, _ = phase2_project
    result = run_phase2(data_root, phase1_dir, write_outputs=False, n_boot=200)

    downgrade = [d for d in result.tier_downgrades if d["component"] == "Gearbox"]
    assert len(downgrade) == 1
    assert downgrade[0]["from_tier"] == "A"
    assert downgrade[0]["to_tier"] == "B"
    assert "Gearbox" in result.tier_b_fits
    assert "Gearbox" not in result.tier_a_fits


def test_run_phase2_hydraulic_system_is_tier_b(phase2_project):
    data_root, phase1_dir, _ = phase2_project
    result = run_phase2(data_root, phase1_dir, write_outputs=False, n_boot=200)

    assert "Hydraulic System" in result.tier_b_fits
    assert result.tier_b_fits["Hydraulic System"].n_used == 5


def test_run_phase2_remaining_components_are_tier_c_with_priors(phase2_project):
    data_root, phase1_dir, _ = phase2_project
    result = run_phase2(data_root, phase1_dir, write_outputs=False, n_boot=200)

    expected_tier_c = {
        "Main/Rotor Bearing", "Generator", "Mechanical Brake", "Converter",
        "Transformer", "Yaw System", "SCADA/Communication",
        "Electrical Safety System", "Cooling System",
        "Grounding/Lightning Protection",
    }
    assert set(result.tier_c_priors.keys()) == expected_tier_c
    # Every Tier C row must carry a source string, even the not_yet_sourced ones.
    for prior in result.tier_c_priors.values():
        assert prior.source.strip() != ""


def test_run_phase2_mtbf_table_has_all_thirteen_components(phase2_project):
    data_root, phase1_dir, _ = phase2_project
    result = run_phase2(data_root, phase1_dir, write_outputs=False, n_boot=200)
    assert len(result.mtbf_table) == 13
    assert set(result.mtbf_table["tier"]) == {"A", "B", "C"}


def test_run_phase2_writes_expected_files(phase2_project, tmp_path):
    data_root, phase1_dir, _ = phase2_project
    out_dir = tmp_path / "phase2_out"
    run_phase2(data_root, phase1_dir, output_dir=out_dir, n_boot=200)

    assert (out_dir / "linked_events.csv").exists()
    assert (out_dir / "tbf_table.csv").exists()
    assert (out_dir / "mtbf_table.csv").exists()
    assert (out_dir / "phase2_report.md").exists()
    assert (out_dir / "hazard_tier_a_b.png").exists()
    assert (out_dir / "hazard_system_illustrative.png").exists()

    mtbf = pd.read_csv(out_dir / "mtbf_table.csv")
    assert len(mtbf) == 13


def test_run_phase2_report_contains_key_sections(phase2_project):
    data_root, phase1_dir, _ = phase2_project
    result = run_phase2(data_root, phase1_dir, write_outputs=False, n_boot=200)

    assert "AEOLUS-RAMS — Phase 2 Report" in result.report_markdown
    assert "Definition of Done" in result.report_markdown
    assert "Tier A" in result.report_markdown
    assert "Tier C" in result.report_markdown


def test_run_phase2_missing_phase1_dir_raises(phase2_project, tmp_path):
    data_root, _, _ = phase2_project
    with pytest.raises(FileNotFoundError):
        run_phase2(data_root, tmp_path / "nonexistent_phase1", write_outputs=False)


def test_cli_main_returns_zero_on_success(phase2_project, tmp_path, capsys):
    data_root, phase1_dir, _ = phase2_project
    out_dir = tmp_path / "cli_out"
    rc = main([
        "--data-root", str(data_root),
        "--phase1-dir", str(phase1_dir),
        "--output-dir", str(out_dir),
        "--n-boot", "200",
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "PHASE 2 COMPLETE" in captured.out
    assert (out_dir / "mtbf_table.csv").exists()


def test_cli_main_returns_nonzero_on_bad_data_root(tmp_path):
    rc = main(["--data-root", str(tmp_path / "nope"), "--phase1-dir", str(tmp_path)])
    assert rc == 1


def test_run_phase2_is_idempotent(phase2_project, tmp_path):
    data_root, phase1_dir, _ = phase2_project
    out_dir = tmp_path / "run1"
    r1 = run_phase2(data_root, phase1_dir, output_dir=out_dir, n_boot=200, seed=7)
    r2 = run_phase2(data_root, phase1_dir, output_dir=out_dir, n_boot=200, seed=7)
    pd.testing.assert_frame_equal(
        r1.mtbf_table.reset_index(drop=True),
        r2.mtbf_table.reset_index(drop=True),
    )
