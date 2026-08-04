"""
Tests for aeolus_rams_phase6.event_tree
"""
from __future__ import annotations

import numpy as np
import pytest

from aeolus_rams_phase6 import config as _cfg
from aeolus_rams_phase6.event_tree import (
    build_event_tree,
    consequence_frequency_table,
    load_gate_Q_values,
    load_basic_event_Q_values,
    load_ccf_uplift,
    EventTreeBranch,
)
from tests.conftest import Q_G1, Q_G2, Q_G3, Q_TOP, CCF_UPLIFT, GATE_Q


# ── Branch probability invariants ─────────────────────────────────────────

class TestBranchProbabilities:
    def test_branches_sum_to_one_independent(self, branches):
        total = sum(b.P_branch_given_IE for b in branches)
        assert abs(total - 1.0) < 1e-9, f"Branches sum to {total}, not 1.0"

    def test_branches_sum_to_one_ccf(self, branches_ccf):
        total = sum(b.P_branch_given_IE for b in branches_ccf)
        assert abs(total - 1.0) < 1e-9

    def test_branch_count_is_four(self, branches):
        assert len(branches) == 4

    def test_branch_labels(self, branches):
        labels = {b.label for b in branches}
        expected = {
            "Branch_1_SAFE_controlled",
            "Branch_2_SAFE_brake",
            "Branch_3_NEAR_MISS_SCADA",
            "Branch_4_CATASTROPHIC",
        }
        assert labels == expected

    def test_catastrophic_branch_probability_matches_Q_top(self, branches):
        cat4 = next(b for b in branches if "CATASTROPHIC" in b.label)
        assert abs(cat4.P_branch_given_IE - Q_TOP) < 1e-12

    def test_branch1_probability_is_1_minus_Q_G1(self, branches):
        b1 = next(b for b in branches if "controlled" in b.label)
        assert abs(b1.P_branch_given_IE - (1.0 - Q_G1)) < 1e-12

    def test_branch2_probability(self, branches):
        b2 = next(b for b in branches if "brake" in b.label)
        expected = Q_G1 * (1.0 - Q_G2)
        assert abs(b2.P_branch_given_IE - expected) < 1e-12

    def test_branch3_probability(self, branches):
        b3 = next(b for b in branches if "NEAR_MISS" in b.label)
        expected = Q_G1 * Q_G2 * (1.0 - Q_G3)
        assert abs(b3.P_branch_given_IE - expected) < 1e-12

    def test_ccf_increases_catastrophic_branch(self, branches, branches_ccf):
        cat4_indep = next(b for b in branches     if "CATASTROPHIC" in b.label)
        cat4_ccf   = next(b for b in branches_ccf if "CATASTROPHIC" in b.label)
        assert cat4_ccf.P_branch_given_IE > cat4_indep.P_branch_given_IE

    def test_ccf_reduces_near_miss_branch(self, branches, branches_ccf):
        b3_indep = next(b for b in branches     if "NEAR_MISS" in b.label)
        b3_ccf   = next(b for b in branches_ccf if "NEAR_MISS" in b.label)
        assert b3_ccf.P_branch_given_IE <= b3_indep.P_branch_given_IE


# ── Derived quantities ────────────────────────────────────────────────────

class TestDerivedQuantities:
    def test_lambda_outcome_per_turbine(self, branches):
        for b in branches:
            expected = _cfg.LAMBDA_IE_CENTRAL * b.P_branch_given_IE
            assert abs(b.lambda_outcome_per_turbine - expected) < 1e-14

    def test_lambda_per_farm_scales_by_N_turbines(self, branches):
        for b in branches:
            assert abs(b.lambda_outcome_per_farm - b.lambda_outcome_per_turbine * _cfg.N_TURBINES) < 1e-14

    def test_catastrophic_frequency_value(self, branches):
        cat4 = next(b for b in branches if "CATASTROPHIC" in b.label)
        expected = _cfg.LAMBDA_IE_CENTRAL * Q_TOP
        assert abs(cat4.lambda_outcome_per_turbine - expected) < 1e-14

    def test_frequency_category_catastrophic_is_F4(self, branches):
        cat4 = next(b for b in branches if "CATASTROPHIC" in b.label)
        assert cat4.frequency_category() == "F4"

    def test_frequency_category_branch1_is_F1(self, branches):
        b1 = next(b for b in branches if "controlled" in b.label)
        assert b1.frequency_category() == "F1"

    def test_risk_acceptability_catastrophic_is_ALARP(self, branches):
        cat4 = next(b for b in branches if "CATASTROPHIC" in b.label)
        assert cat4.risk_acceptability() == "ALARP"

    def test_severity_categories(self, branches):
        sev_map = {
            "Branch_1_SAFE_controlled":  "E",
            "Branch_2_SAFE_brake":       "D",
            "Branch_3_NEAR_MISS_SCADA":  "C",
            "Branch_4_CATASTROPHIC":     "A",
        }
        for b in branches:
            assert b.severity_category == sev_map[b.label]

    def test_P_one_or_more_in_design_life_is_between_0_and_1(self, branches):
        for b in branches:
            p = b.P_one_or_more_in_design_life
            assert 0.0 <= p <= 1.0


# ── CSV loading ───────────────────────────────────────────────────────────

class TestCSVLoading:
    def test_load_gate_Q_values_returns_four_gates(self, gate_Q_csv):
        gq = load_gate_Q_values(gate_Q_csv, t_filter="365d")
        assert len(gq) == 4

    def test_load_gate_Q_values_strips_whitespace(self, gate_Q_csv):
        gq = load_gate_Q_values(gate_Q_csv, t_filter="365d")
        assert "G1: Pitch System fails to feather" in gq

    def test_load_gate_Q_values_no_basic_events(self, gate_Q_csv):
        gq = load_gate_Q_values(gate_Q_csv, t_filter="365d")
        assert not any("G1a" in k or "G2a" in k for k in gq)

    def test_load_gate_Q_values_correct_Q_G1(self, gate_Q_csv):
        gq = load_gate_Q_values(gate_Q_csv, t_filter="365d")
        assert abs(gq[_cfg.GATE_NAMES["G1"]] - Q_G1) < 1e-12

    def test_load_basic_event_Q_values_returns_seven_events(self, gate_Q_csv):
        beq = load_basic_event_Q_values(gate_Q_csv, t_filter="365d")
        assert len(beq) == 7

    def test_load_ccf_uplift_matches_phase5(self, ccf_csv):
        uplift = load_ccf_uplift(ccf_csv, beta=0.10)
        assert abs(uplift - CCF_UPLIFT) < 1e-6

    def test_load_ccf_uplift_fallback_on_missing_file(self, tmp_path):
        uplift = load_ccf_uplift(tmp_path / "nonexistent.csv", beta=0.10)
        assert uplift == _cfg.CCF_UPLIFT_CENTRAL


# ── Consequence frequency table ───────────────────────────────────────────

class TestConsequenceFrequencyTable:
    def test_cft_has_expected_columns(self, cft):
        required = {"lambda_IE", "branch_label", "P_branch",
                    "lambda_per_turbine", "freq_category",
                    "risk_acceptability", "ccf_adjusted"}
        assert required.issubset(set(cft.columns))

    def test_cft_row_count(self, cft):
        # 6 λ_IE values × 4 branches × 2 (ccf/no-ccf) = 48 rows
        assert len(cft) == 48

    def test_cft_branch_probabilities_sum_to_one_per_scenario(self, cft):
        for (lam, ccf), grp in cft.groupby(["lambda_IE", "ccf_adjusted"]):
            total = grp["P_branch"].sum()
            assert abs(total - 1.0) < 1e-9, (
                f"λ_IE={lam}, ccf={ccf}: P_branch sums to {total}"
            )