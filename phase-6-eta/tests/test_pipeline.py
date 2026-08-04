"""
Integration tests for aeolus_rams_phase6.pipeline
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aeolus_rams_phase6 import config as _cfg
from aeolus_rams_phase6.pipeline import run_pipeline, Phase6Result


@pytest.fixture(name="phase5_dir")
def fixture_phase5_dir(gate_Q_csv, ccf_csv, tmp_path) -> Path:
    """Create a mock phase5 outputs directory with required CSVs."""
    d = tmp_path / "phase5_outputs"
    d.mkdir()
    shutil.copy(gate_Q_csv, d / "gate_Q_table.csv")
    shutil.copy(ccf_csv, d / "ccf_sensitivity.csv")
    return d


@pytest.fixture(name="pipeline_result")
def fixture_pipeline_result(phase5_dir, tmp_path) -> Phase6Result:
    output_dir = tmp_path / "phase6_outputs"
    return run_pipeline(
        phase5_dir=phase5_dir,
        output_dir=output_dir,
        lambda_ie=_cfg.LAMBDA_IE_CENTRAL,
    )


class TestPipelineOutputs:
    def test_pipeline_completes_without_error(self, pipeline_result):
        assert pipeline_result is not None

    def test_consequence_frequency_table_csv_exists(self, pipeline_result):
        cft_path = pipeline_result.output_dir / "consequence_frequency_table.csv"
        assert cft_path.exists()

    def test_consequence_frequency_table_has_48_rows(self, pipeline_result):
        cft_path = pipeline_result.output_dir / "consequence_frequency_table.csv"
        df = pd.read_csv(cft_path)
        assert len(df) == 48   # 6 λ_IE × 4 branches × 2 CCF scenarios

    def test_risk_matrix_png_exists(self, pipeline_result):
        rm_path = pipeline_result.output_dir / "risk_matrix.png"
        assert rm_path.exists()
        assert rm_path.stat().st_size > 5_000

    def test_bowtie_png_exists(self, pipeline_result):
        bt_path = pipeline_result.output_dir / "bowtie.png"
        assert bt_path.exists()
        assert bt_path.stat().st_size > 5_000

    def test_alarp_table_csv_exists(self, pipeline_result):
        alarp_path = pipeline_result.output_dir / "alarp_table.csv"
        assert alarp_path.exists()

    def test_alarp_table_has_three_rows(self, pipeline_result):
        alarp_path = pipeline_result.output_dir / "alarp_table.csv"
        df = pd.read_csv(alarp_path)
        assert len(df) == 3

    def test_phase6_report_md_exists(self, pipeline_result):
        report_path = pipeline_result.output_dir / "phase6_report.md"
        assert report_path.exists()

    def test_phase6_report_contains_key_sections(self, pipeline_result):
        report_path = pipeline_result.output_dir / "phase6_report.md"
        text = report_path.read_text(encoding="utf-8")
        for section in [
            "Executive Summary",
            "Phase 5 Inheritance",
            "Initiating Events",
            "Event Tree Structure",
            "Risk Matrix",
            "ALARP Argument",
            "Bridge to Phase 7",
        ]:
            assert section in text, f"Missing section: {section}"

    def test_branches_in_result_sum_to_one(self, pipeline_result):
        total = sum(b.P_branch_given_IE for b in pipeline_result.branches)
        assert abs(total - 1.0) < 1e-9

    def test_ccf_uplift_loaded_correctly(self, pipeline_result):
        from tests.conftest import CCF_UPLIFT
        assert abs(pipeline_result.ccf_uplift - CCF_UPLIFT) < 1e-6

    def test_risk_matrix_ccf_png_exists(self, pipeline_result):
        rm_ccf_path = pipeline_result.output_dir / "risk_matrix_ccf.png"
        assert rm_ccf_path.exists()

    def test_risk_matrix_alarp_option1_png_exists(self, pipeline_result):
        rm_opt1_path = pipeline_result.output_dir / "risk_matrix_alarp_option1.png"
        assert rm_opt1_path.exists()