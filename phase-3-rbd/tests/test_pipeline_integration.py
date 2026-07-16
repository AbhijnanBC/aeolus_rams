from __future__ import annotations

import math
import pandas as pd
import pytest

from aeolus_rams_phase3 import config
from aeolus_rams_phase3.pipeline import run_phase3, main


# ---------------------------------------------------------------------------
# End-to-end pipeline (in-memory)
# ---------------------------------------------------------------------------

def test_run_phase3_completes_in_memory(mtbf_table_path):
    result = run_phase3(mtbf_table_path=mtbf_table_path, write_outputs=False)
    assert result is not None


def test_run_phase3_loads_13_components(mtbf_table_path):
    result = run_phase3(mtbf_table_path=mtbf_table_path, write_outputs=False)
    assert len(result.components) == 13


def test_run_phase3_no_nan_in_components(mtbf_table_path):
    result = run_phase3(mtbf_table_path=mtbf_table_path, write_outputs=False)
    for name, c in result.components.items():
        assert not math.isnan(c.mtbf_days), f"{name} has NaN MTBF"
        assert not math.isnan(c.lambda_per_day), f"{name} has NaN lambda"


def test_run_phase3_r_turbine_1yr_bounded(mtbf_table_path):
    result = run_phase3(mtbf_table_path=mtbf_table_path, write_outputs=False)
    r = result.R_turbine_1yr
    assert 0.0 < r < 0.90
    assert not math.isnan(r)


def test_run_phase3_r_turbine_5yr_very_low(mtbf_table_path):
    """With Pitch (MTBF≈1936d) and Hydraulic (≈1845d) in series, R(5yr) << 0.5."""
    result = run_phase3(mtbf_table_path=mtbf_table_path, write_outputs=False)
    r = result.R_turbine_5yr
    assert 0.0 < r < 0.25


def test_run_phase3_lambda_sys_positive(mtbf_table_path):
    result = run_phase3(mtbf_table_path=mtbf_table_path, write_outputs=False)
    assert result.lambda_sys > 0
    assert not math.isnan(result.lambda_sys)


def test_run_phase3_top_two_includes_pitch_or_hydraulic(mtbf_table_path):
    result = run_phase3(mtbf_table_path=mtbf_table_path, write_outputs=False)
    top_two = result.top_two_components
    assert len(top_two) == 2
    assert "Pitch System" in top_two or "Hydraulic System" in top_two


def test_run_phase3_importance_table_has_all_components(mtbf_table_path):
    result = run_phase3(mtbf_table_path=mtbf_table_path, write_outputs=False)
    assert set(result.imp_table["component"]) == set(result.components.keys())


def test_run_phase3_IC_sums_to_one(mtbf_table_path):
    result = run_phase3(mtbf_table_path=mtbf_table_path, write_outputs=False)
    ic_cols = [c for c in result.imp_table.columns if c.startswith("IC_")]
    for col in ic_cols:
        total = result.imp_table[col].sum()
        # Rounded to 6dp × 13 components → up to ~6.5e-6 accumulated error
        assert abs(total - 1.0) < 1e-4, f"IC column {col} sums to {total} ≠ 1"


def test_run_phase3_report_contains_key_sections(mtbf_table_path):
    result = run_phase3(mtbf_table_path=mtbf_table_path, write_outputs=False)
    report = result.report_markdown
    assert "AEOLUS-RAMS — Phase 3 Report" in report
    assert "Definition of Done" in report
    assert "Birnbaum" in report
    assert "Criticality" in report
    assert "sensitivity" in report.lower()
    assert "Phase 4" in report


def test_run_phase3_system_table_has_5_mission_times(mtbf_table_path):
    result = run_phase3(mtbf_table_path=mtbf_table_path, write_outputs=False)
    assert len(result.system_rt_table) == len(config.MISSION_TIMES_DAYS)


def test_run_phase3_farm_table_has_5_rows(mtbf_table_path):
    result = run_phase3(mtbf_table_path=mtbf_table_path, write_outputs=False)
    assert len(result.farm_table) == len(config.MISSION_TIMES_DAYS)


# ---------------------------------------------------------------------------
# File outputs
# ---------------------------------------------------------------------------

def test_run_phase3_writes_expected_files(mtbf_table_path, tmp_path):
    out = tmp_path / "p3_outputs"
    run_phase3(mtbf_table_path=mtbf_table_path, output_dir=out)
    assert (out / "component_rt_table.csv").exists()
    assert (out / "system_reliability_table.csv").exists()
    assert (out / "farm_reliability_table.csv").exists()
    assert (out / "importance_table.csv").exists()
    assert (out / "importance_rank.csv").exists()
    assert (out / "sensitivity_aggregate.csv").exists()
    assert (out / "phase3_report.md").exists()
    assert (out / "turbine_rbd.png").exists()
    assert (out / "farm_rbd.png").exists()
    assert (out / "sensitivity_band.png").exists()


def test_run_phase3_exports_are_non_empty(mtbf_table_path, tmp_path):
    out = tmp_path / "p3_outputs"
    run_phase3(mtbf_table_path=mtbf_table_path, output_dir=out)
    comp_tbl = pd.read_csv(out / "component_rt_table.csv")
    assert len(comp_tbl) == 13
    imp_tbl = pd.read_csv(out / "importance_table.csv")
    assert len(imp_tbl) == 13


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_run_phase3_is_idempotent(mtbf_table_path, tmp_path):
    out = tmp_path / "run1"
    r1 = run_phase3(mtbf_table_path=mtbf_table_path, output_dir=out)
    r2 = run_phase3(mtbf_table_path=mtbf_table_path, output_dir=out)
    assert r1.lambda_sys == pytest.approx(r2.lambda_sys, rel=1e-12)
    assert r1.R_turbine_5yr == pytest.approx(r2.R_turbine_5yr, rel=1e-12)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_main_returns_zero_on_success(mtbf_table_path, tmp_path, capsys):
    out = tmp_path / "cli_out"
    rc = main(["--mtbf-table", mtbf_table_path, "--output-dir", str(out)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "PHASE 3 COMPLETE" in captured.out
    assert (out / "phase3_report.md").exists()


def test_cli_main_returns_nonzero_on_bad_path(tmp_path, capsys):
    rc = main(["--mtbf-table", str(tmp_path / "nonexistent.csv")])
    assert rc == 1


def test_cli_output_contains_top_components(mtbf_table_path, tmp_path, capsys):
    out = tmp_path / "cli_chk"
    main(["--mtbf-table", mtbf_table_path, "--output-dir", str(out)])
    captured = capsys.readouterr()
    # Console output mentions the top IC components
    assert "Pitch" in captured.out or "Hydraulic" in captured.out
