"""Tests for mcs.py."""
from __future__ import annotations

import pytest

from aeolus_rams_phase5.mcs import extract_minimal_cut_sets, mcs_table
from aeolus_rams_phase5.fault_tree import BasicEvent, Gate


def test_mcs_count_is_12(top_gate_1yr):
    """There must be exactly 12 MCS (3 × 2 × 2 from AND top with OR sub-gates)."""
    mcs_list = extract_minimal_cut_sets(top_gate_1yr)
    assert len(mcs_list) == 12, f"Expected 12 MCS, got {len(mcs_list)}"


def test_all_mcs_order_3(top_gate_1yr):
    """All MCS must be of order 3 (AND gate at top requires one from each OR sub-gate)."""
    mcs_list = extract_minimal_cut_sets(top_gate_1yr)
    for mcs in mcs_list:
        assert len(mcs) == 3, (
            f"MCS {mcs} has order {len(mcs)}, expected 3"
        )


def test_mcs_sorted_by_Q_descending(top_gate_1yr):
    """MCS must be sorted by Q value, highest first."""
    mcs_list = extract_minimal_cut_sets(top_gate_1yr)
    all_be = top_gate_1yr.collect_basic_events()

    def _q(mcs):
        q = 1.0
        for name in mcs:
            q *= all_be[name].Q
        return q

    q_vals = [_q(m) for m in mcs_list]
    for i in range(len(q_vals) - 1):
        assert q_vals[i] >= q_vals[i + 1] - 1e-15, (
            f"MCS Q not sorted: q[{i}]={q_vals[i]:.4e} < q[{i+1}]={q_vals[i+1]:.4e}"
        )


def test_mcs_are_minimal():
    """No MCS should be a superset of another."""
    from aeolus_rams_phase5 import config as _cfg
    from aeolus_rams_phase5.fault_tree import build_overspeed_fault_tree
    tree = build_overspeed_fault_tree(_cfg.FALLBACK_Q_1YR)
    mcs_list = extract_minimal_cut_sets(tree)
    for i, mcs_i in enumerate(mcs_list):
        for j, mcs_j in enumerate(mcs_list):
            if i != j:
                assert not (mcs_j < mcs_i), (
                    f"MCS {mcs_i} is a superset of {mcs_j} — not minimal"
                )


def test_mcs_cover_all_events(top_gate_1yr):
    """Every basic event should appear in at least one MCS."""
    mcs_list = extract_minimal_cut_sets(top_gate_1yr)
    all_be = top_gate_1yr.collect_basic_events()
    covered = set().union(*mcs_list)
    assert covered == set(all_be.keys()), (
        f"Events not in any MCS: {set(all_be.keys()) - covered}"
    )


def test_mcs_table_columns(top_gate_1yr):
    """mcs_table must have required columns."""
    df = mcs_table(top_gate_1yr, "365d")
    for col in ["rank", "mcs_id", "events", "order", "Q_mcs", "FV", "t"]:
        assert col in df.columns
    assert len(df) == 12
    assert (df["FV"] >= 0).all()
    assert abs(df["FV"].sum() - 1.0) < 0.01  # FV should sum to ~1 (rare-event approx)


def test_simple_tree_mcs():
    """Unit test: simple 2-layer tree should produce predictable MCS."""
    # AND(OR(A,B), OR(C,D)) → MCS: {A,C}, {A,D}, {B,C}, {B,D}
    A = BasicEvent("A", 0.1, confidence="test")
    B = BasicEvent("B", 0.2, confidence="test")
    C = BasicEvent("C", 0.3, confidence="test")
    D = BasicEvent("D", 0.4, confidence="test")
    G1 = Gate("G1", "OR", (A, B))
    G2 = Gate("G2", "OR", (C, D))
    TE = Gate("TE", "AND", (G1, G2))

    mcs_list = extract_minimal_cut_sets(TE)
    assert len(mcs_list) == 4
    expected = [
        frozenset(["A", "C"]), frozenset(["A", "D"]),
        frozenset(["B", "C"]), frozenset(["B", "D"]),
    ]
    assert set(mcs_list) == set(expected)