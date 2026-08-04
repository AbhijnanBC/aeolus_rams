"""
Tests for aeolus_rams_phase6.risk_matrix
"""
from __future__ import annotations

import pytest
from pathlib import Path

from aeolus_rams_phase6 import config as _cfg
from aeolus_rams_phase6.risk_matrix import (
    plot_risk_matrix,
    build_risk_points,
    CELL_COLORS,
    _MATRIX_CELLS,
    _SEV_IDX,
    _FREQ_IDX,
)


class TestRiskMatrixStructure:
    def test_matrix_is_5x5(self):
        assert len(_MATRIX_CELLS) == 5
        for row in _MATRIX_CELLS:
            assert len(row) == 5

    def test_all_cells_are_valid_labels(self):
        valid = {"UNACCEPTABLE", "ALARP", "ACCEPTABLE"}
        for row in _MATRIX_CELLS:
            for cell in row:
                assert cell in valid

    def test_cell_colors_keys_match_labels(self):
        assert set(CELL_COLORS.keys()) == {"UNACCEPTABLE", "ALARP", "ACCEPTABLE"}

    def test_cat_A_row_has_no_acceptable_at_F1_F2_F3(self):
        # First three F-columns for Cat A must be UNACCEPTABLE
        row_A = _MATRIX_CELLS[_SEV_IDX["A"]]
        assert row_A[0] == "UNACCEPTABLE"
        assert row_A[1] == "UNACCEPTABLE"
        assert row_A[2] == "UNACCEPTABLE"

    def test_cat_A_F4_is_ALARP(self):
        row_A = _MATRIX_CELLS[_SEV_IDX["A"]]
        assert row_A[_FREQ_IDX["F4"]] == "ALARP"

    def test_cat_A_F5_is_ACCEPTABLE(self):
        row_A = _MATRIX_CELLS[_SEV_IDX["A"]]
        assert row_A[_FREQ_IDX["F5"]] == "ACCEPTABLE"

    def test_cat_E_all_acceptable(self):
        row_E = _MATRIX_CELLS[_SEV_IDX["E"]]
        assert all(c == "ACCEPTABLE" for c in row_E)


class TestBuildRiskPoints:
    def test_build_risk_points_returns_five_points_with_bop(self, cft):
        pts = build_risk_points(cft, lambda_IE=_cfg.LAMBDA_IE_CENTRAL, ccf_adjusted=False)
        assert len(pts) == 5  # 4 branches + 1 BoP

    def test_build_risk_points_returns_four_without_bop(self, cft):
        pts = build_risk_points(cft, lambda_IE=_cfg.LAMBDA_IE_CENTRAL,
                                ccf_adjusted=False, include_bop=False)
        assert len(pts) == 4

    def test_each_point_has_required_keys(self, cft):
        pts = build_risk_points(cft, lambda_IE=_cfg.LAMBDA_IE_CENTRAL)
        for pt in pts:
            assert "label" in pt
            assert "severity" in pt
            assert "freq_cat" in pt

    def test_catastrophic_point_is_category_A(self, cft):
        pts = build_risk_points(cft, lambda_IE=_cfg.LAMBDA_IE_CENTRAL, include_bop=False)
        cat4_pts = [p for p in pts if "CATASTROPHIC" in p["label"] or "B4" in p["label"]]
        assert len(cat4_pts) == 1
        assert cat4_pts[0]["severity"] == "A"


class TestPlotRiskMatrix:
    def test_plot_risk_matrix_creates_file(self, tmp_path, cft):
        pts = build_risk_points(cft, lambda_IE=_cfg.LAMBDA_IE_CENTRAL)
        out = tmp_path / "risk_matrix_test.png"
        result = plot_risk_matrix(pts, out)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 5_000