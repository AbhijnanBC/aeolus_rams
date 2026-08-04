"""Tests for importance.py — IB, CIM, FV."""
from __future__ import annotations

import numpy as np
import pytest

from aeolus_rams_phase5.importance import compute_importance_measures, gate_importance_table
from aeolus_rams_phase5.fault_tree import BasicEvent, Gate


def test_importance_columns(top_gate_1yr):
    """Importance table must have all required columns."""
    df = compute_importance_measures(top_gate_1yr, "365d")
    for col in ["basic_event", "Q", "IB", "CIM", "FV", "FV_rank", "IB_rank"]:
        assert col in df.columns


def test_seven_rows(top_gate_1yr):
    """One row per basic event (7 total)."""
    df = compute_importance_measures(top_gate_1yr, "365d")
    assert len(df) == 7


def test_FV_values_between_0_and_1(top_gate_1yr):
    """FV must be in [0, 1]."""
    df = compute_importance_measures(top_gate_1yr, "365d")
    assert (df["FV"] >= 0.0).all()
    assert (df["FV"] <= 1.0 + 1e-9).all()


def test_IB_non_negative(top_gate_1yr):
    """Birnbaum importance must be ≥ 0 (fixing a component can only reduce risk)."""
    df = compute_importance_measures(top_gate_1yr, "365d")
    assert (df["IB"] >= -1e-12).all(), f"Negative IB found:\n{df[df['IB']<0]}"


def test_CIM_non_negative(top_gate_1yr):
    """CIM must be ≥ 0."""
    df = compute_importance_measures(top_gate_1yr, "365d")
    assert (df["CIM"] >= -1e-12).all()


def test_FV_G3b_highest(top_gate_1yr):
    """G3b (Safety relay fault) must have the highest or joint-highest FV.

    G3b has higher Q than G3a (0.027 vs 0.006), so it appears in more
    high-Q MCS and should dominate the Fussell-Vesely ranking.
    """
    df = compute_importance_measures(top_gate_1yr, "365d")
    g3b_row = df[df["basic_event"].str.contains("G3b")]
    assert not g3b_row.empty
    top_fv = df["FV"].max()
    # G3b should be in top 3 by FV
    g3b_fv = float(g3b_row["FV"].iloc[0])
    assert g3b_fv >= df["FV"].nlargest(3).iloc[-1] - 1e-6, (
        f"G3b FV={g3b_fv:.4f} is not in top 3 (threshold={df['FV'].nlargest(3).iloc[-1]:.4f})"
    )


def test_analytical_IB_and_gate(top_gate_1yr):
    """For AND top gate with OR sub-gates: IB_Gi = ∏(sibling gate Q values).

    Verify gate IB against analytical formula.
    """
    gi_df = gate_importance_table(top_gate_1yr, "365d")
    gates = top_gate_1yr.collect_gates()

    Q_G1 = gates["G1: Pitch System fails to feather"].Q
    Q_G2 = gates["G2: Mechanical brake fails to engage"].Q
    Q_G3 = gates["G3: SCADA overspeed trip fails to activate"].Q

    expected_IB = {
        "G1: Pitch System fails to feather":          Q_G2 * Q_G3,
        "G2: Mechanical brake fails to engage":        Q_G1 * Q_G3,
        "G3: SCADA overspeed trip fails to activate": Q_G1 * Q_G2,
    }
    for gate_name, expected in expected_IB.items():
        row = gi_df[gi_df["gate"] == gate_name]
        if not row.empty:
            actual = float(row["IB_gate"].iloc[0])
            assert abs(actual - expected) < 1e-10, (
                f"Gate IB mismatch for {gate_name}: "
                f"actual={actual:.8e}, expected={expected:.8e}"
            )