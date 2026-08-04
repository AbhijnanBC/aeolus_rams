"""
Tests for aeolus_rams_phase6.alarp
"""
from __future__ import annotations

import numpy as np
import pytest

from aeolus_rams_phase6 import config as _cfg
from aeolus_rams_phase6.alarp import (
    compute_option_1,
    compute_option_2,
    compute_option_3,
    compute_all_options,
    alarp_table,
    _or_gate,
    _BASELINE_Q,
)
from tests.conftest import Q_G1, Q_G2, Q_G3, Q_TOP, GATE_Q


class TestOrGate:
    def test_single_event(self):
        assert abs(_or_gate([0.1]) - 0.1) < 1e-12

    def test_two_independent_events(self):
        # OR(0.1, 0.2) = 1 - 0.9*0.8 = 0.28
        assert abs(_or_gate([0.1, 0.2]) - 0.28) < 1e-12

    def test_zero_events_gives_zero(self):
        assert _or_gate([0.0, 0.0]) == 0.0

    def test_baseline_Q_G1_matches_phase5(self):
        Q_G1_computed = _or_gate([
            _BASELINE_Q["G1a: Control/Encoder fault"],
            _BASELINE_Q["G1b: Battery/Electrical supply fault"],
            _BASELINE_Q["G1c: Actuator mechanical failure"],
        ])
        assert abs(Q_G1_computed - Q_G1) < 1e-6, (
            f"_or_gate(G1 sub-events) = {Q_G1_computed:.6f}, expected {Q_G1:.6f}"
        )

    def test_baseline_Q_G2_matches_phase5(self):
        Q_G2_computed = _or_gate([
            _BASELINE_Q["G2a: Hydraulic supply loss"],
            _BASELINE_Q["G2b: Brake hardware fault"],
        ])
        assert abs(Q_G2_computed - Q_G2) < 1e-6

    def test_baseline_Q_G3_matches_phase5(self):
        Q_G3_computed = _or_gate([
            _BASELINE_Q["G3a: Communication/fieldbus loss"],
            _BASELINE_Q["G3b: Safety relay/electrical fault"],
        ])
        assert abs(Q_G3_computed - Q_G3) < 1e-6


class TestALARPOptions:
    def test_option_1_reduces_Q_top_significantly(self, gate_Q):
        opt = compute_option_1(gate_Q[_cfg.GATE_NAMES["TE"]])
        reduction = opt.Q_top_reduction_pct
        assert reduction > 75.0, f"Option 1 should reduce Q_top by >75%, got {reduction:.1f}%"

    def test_option_1_moves_risk_to_acceptable(self, gate_Q):
        opt = compute_option_1(gate_Q[_cfg.GATE_NAMES["TE"]])
        assert opt.risk_matrix_moves_to_acceptable, (
            f"Option 1 should reach ACCEPTABLE, got {opt.risk_acceptability_after}"
        )

    def test_option_1_targets_G3(self, gate_Q):
        opt = compute_option_1(gate_Q[_cfg.GATE_NAMES["TE"]])
        assert opt.target_gate == "G3"
        assert opt.fv_rank == 1

    def test_option_1_Q_gate_after_is_less_than_before(self, gate_Q):
        opt = compute_option_1(gate_Q[_cfg.GATE_NAMES["TE"]])
        assert opt.Q_gate_after < opt.Q_gate_before

    def test_option_2_reduces_Q_top(self, gate_Q):
        opt = compute_option_2(gate_Q[_cfg.GATE_NAMES["TE"]])
        assert opt.Q_top_after < opt.Q_top_before

    def test_option_2_targets_G2(self, gate_Q):
        opt = compute_option_2(gate_Q[_cfg.GATE_NAMES["TE"]])
        assert opt.target_gate == "G2"
        assert opt.fv_rank == 2

    def test_option_3_reduces_Q_top(self, gate_Q):
        opt = compute_option_3(gate_Q[_cfg.GATE_NAMES["TE"]])
        assert opt.Q_top_after < opt.Q_top_before

    def test_option_3_targets_G1(self, gate_Q):
        opt = compute_option_3(gate_Q[_cfg.GATE_NAMES["TE"]])
        assert opt.target_gate == "G1"
        assert opt.fv_rank == 3

    def test_option_3_insufficient_alone_for_acceptable(self, gate_Q):
        opt = compute_option_3(gate_Q[_cfg.GATE_NAMES["TE"]])
        assert not opt.risk_matrix_moves_to_acceptable

    def test_Q_top_after_equals_gate_product(self, gate_Q):
        """Q_top_after must equal Q_G1 × Q_G2_improved × Q_G3 for Option 2."""
        opt = compute_option_2(gate_Q[_cfg.GATE_NAMES["TE"]])
        Q_G2_after_v = _or_gate([
            _BASELINE_Q["G2a: Hydraulic supply loss"] / 5.0,
            _BASELINE_Q["G2b: Brake hardware fault"],
        ])
        expected = Q_G1 * Q_G2_after_v * Q_G3
        assert abs(opt.Q_top_after - expected) < 1e-10

    def test_FV_rank_ordering(self, gate_Q):
        opts = compute_all_options(gate_Q)
        ranks = [o.fv_rank for o in opts]
        assert ranks == [1, 2, 3]

    def test_alarp_table_has_correct_columns(self, gate_Q):
        opts = compute_all_options(gate_Q)
        df = alarp_table(opts)
        required_cols = {
            "option", "FV_rank", "Q_top_before", "Q_top_after",
            "Q_top_reduction_pct", "risk_matrix_before", "risk_matrix_after",
            "moves_to_acceptable",
        }
        assert required_cols.issubset(set(df.columns))

    def test_alarp_table_has_three_rows(self, gate_Q):
        opts = compute_all_options(gate_Q)
        df = alarp_table(opts)
        assert len(df) == 3