"""
Tests for fault_tree.py.

All gate Q values verified by hand against OR/AND formulas before
these tests were written. Tests serve as regression guards, not
as the primary verification.
"""
from __future__ import annotations

import numpy as np
import pytest

from aeolus_rams_phase5.fault_tree import (
    BasicEvent, Gate,
    build_overspeed_fault_tree,
    gate_Q_table,
    _allocate_subcause_Q,
)
from aeolus_rams_phase5 import config as _cfg


# ── OR gate exact formula ─────────────────────────────────────────────────

def test_or_gate_exact_formula():
    """OR gate Q = 1 - ∏(1 - Qᵢ), NOT rare-event approximation."""
    g1 = BasicEvent("A", 0.10, confidence="test")
    g2 = BasicEvent("B", 0.20, confidence="test")
    gate = Gate("OR_test", "OR", (g1, g2))
    expected = 1 - (1 - 0.10) * (1 - 0.20)   # = 0.28
    assert abs(gate.Q - expected) < 1e-12, (
        f"OR gate Q={gate.Q:.12f} ≠ exact {expected:.12f}"
    )


def test_or_gate_never_uses_sum_approximation():
    """OR gate Q must be strictly less than sum of individual Q values."""
    g1 = BasicEvent("A", 0.30, confidence="test")
    g2 = BasicEvent("B", 0.40, confidence="test")
    gate = Gate("OR_test", "OR", (g1, g2))
    assert gate.Q < 0.30 + 0.40, (
        "OR gate Q must be < sum of Q values (rare-event approx not allowed)"
    )


def test_and_gate_product_formula():
    """AND gate Q = ∏ Qᵢ."""
    g1 = BasicEvent("A", 0.10, confidence="test")
    g2 = BasicEvent("B", 0.20, confidence="test")
    g3 = BasicEvent("C", 0.05, confidence="test")
    gate = Gate("AND_test", "AND", (g1, g2, g3))
    expected = 0.10 * 0.20 * 0.05   # = 0.001
    assert abs(gate.Q - expected) < 1e-15


# ── G1 normalisation ──────────────────────────────────────────────────────

def test_G1_Q_equals_pitch_system_Q(top_gate_1yr, Q_1yr):
    """G1 OR gate Q must exactly equal Pitch System Q from Phase 3."""
    gates = top_gate_1yr.collect_gates()
    G1 = gates["G1: Pitch System fails to feather"]
    expected = Q_1yr["Pitch System"]   # 0.171903
    assert abs(G1.Q - expected) < 1e-9, (
        f"G1.Q={G1.Q:.9f} ≠ Q_pitch={expected:.9f} after normalisation"
    )


def test_allocate_subcause_Q_or_matches_target():
    """_allocate_subcause_Q: OR-gate of result must equal target."""
    for Q_target in [0.05, 0.10, 0.17, 0.50]:
        fracs = [0.45, 0.35, 0.20]
        Q_subs = _allocate_subcause_Q(Q_target, fracs)
        Q_or = 1 - np.prod([1 - q for q in Q_subs])
        assert abs(Q_or - Q_target) < 1e-10, (
            f"target={Q_target:.6f}, OR(result)={Q_or:.6f}, delta={abs(Q_or-Q_target):.2e}"
        )


# ── Tree structure ────────────────────────────────────────────────────────

def test_top_gate_is_AND(top_gate_1yr):
    assert top_gate_1yr.gate_type == "AND"


def test_top_gate_has_three_subgates(top_gate_1yr):
    assert len(top_gate_1yr.inputs) == 3
    for child in top_gate_1yr.inputs:
        assert isinstance(child, Gate)
        assert child.gate_type == "OR"


def test_seven_basic_events(top_gate_1yr):
    all_be = top_gate_1yr.collect_basic_events()
    assert len(all_be) == 7, f"Expected 7 basic events, got {len(all_be)}: {list(all_be.keys())}"


def test_Q_values_in_range(top_gate_1yr):
    """All Q values must be in [0, 1]."""
    assert 0.0 <= top_gate_1yr.Q <= 1.0
    for gate in top_gate_1yr.collect_gates().values():
        assert 0.0 <= gate.Q <= 1.0
    for be in top_gate_1yr.collect_basic_events().values():
        assert 0.0 <= be.Q <= 1.0


# ── Phase 3 bridge check ─────────────────────────────────────────────────

def test_Q_top_matches_expected(top_gate_1yr):
    """Q_top must be within 5% of the pre-computed expected value."""
    Q_top = top_gate_1yr.Q
    expected = _cfg.EXPECTED_Q_TOP_1YR
    error = abs(Q_top - expected) / expected
    assert error < _cfg.EXPECTED_Q_TOP_TOLERANCE, (
        f"Q_top={Q_top:.6e} deviates {error*100:.1f}% from expected {expected:.6e} "
        f"(tolerance {_cfg.EXPECTED_Q_TOP_TOLERANCE*100:.0f}%)"
    )


def test_Q_top_1yr_less_than_Q_top_5yr(top_gate_1yr, top_gate_5yr):
    """Higher t → higher Q (unreliability is monotone in time)."""
    assert top_gate_1yr.Q < top_gate_5yr.Q, (
        f"Q_top(1yr)={top_gate_1yr.Q:.4e} should be < Q_top(5yr)={top_gate_5yr.Q:.4e}"
    )


# ── with_basic_event_Q ────────────────────────────────────────────────────

def test_with_Q_zero_reduces_top_event():
    """Setting any basic event Q=0 must reduce or maintain Q_top."""
    Q = _cfg.FALLBACK_Q_1YR.copy()
    tree = build_overspeed_fault_tree(Q)
    Q_original = tree.Q
    all_be = tree.collect_basic_events()
    for name in all_be:
        tree_mod = tree.with_basic_event_Q(name, 0.0)
        assert tree_mod.Q <= Q_original + 1e-12, (
            f"Setting {name} Q=0 increased Q_top: {tree_mod.Q:.4e} > {Q_original:.4e}"
        )


def test_with_Q_one_increases_top_event():
    """Setting any basic event Q=1 must increase or maintain Q_top."""
    Q = _cfg.FALLBACK_Q_1YR.copy()
    tree = build_overspeed_fault_tree(Q)
    Q_original = tree.Q
    all_be = tree.collect_basic_events()
    for name in all_be:
        tree_mod = tree.with_basic_event_Q(name, 1.0)
        assert tree_mod.Q >= Q_original - 1e-12, (
            f"Setting {name} Q=1 decreased Q_top: {tree_mod.Q:.4e} < {Q_original:.4e}"
        )