"""
Shared fixtures for Phase 6 tests.
Uses synthetic Q values that match Phase 5 gate_Q_table.csv exactly
so tests remain deterministic without requiring Phase 5 to be installed.
"""
from __future__ import annotations

import io
import numpy as np
import pandas as pd
from pathlib import Path
import pytest

from aeolus_rams_phase6 import config as _cfg
from aeolus_rams_phase6.event_tree import build_event_tree, consequence_frequency_table


# ── Verified Phase 5 values (from gate_Q_table.csv @ 365d) ────────────────

Q_G1 = 0.17190299999991898
Q_G2 = 0.13620740491919991
Q_G3 = 0.032843124307199933
Q_TOP = 7.690040705441519e-4
CCF_UPLIFT = 1.5771754474755542

GATE_Q: dict[str, float] = {
    "G1: Pitch System fails to feather":                        Q_G1,
    "G2: Mechanical brake fails to engage":                     Q_G2,
    "G3: SCADA overspeed trip fails to activate":               Q_G3,
    "TE: Turbine Overspeed → Catastrophic Structural Failure":  Q_TOP,
}


@pytest.fixture(name="gate_Q")
def fixture_gate_Q() -> dict[str, float]:
    return dict(GATE_Q)


@pytest.fixture(name="branches")
def fixture_branches(gate_Q):
    return build_event_tree(
        lambda_IE=_cfg.LAMBDA_IE_CENTRAL,
        Q_G1=gate_Q[_cfg.GATE_NAMES["G1"]],
        Q_G2=gate_Q[_cfg.GATE_NAMES["G2"]],
        Q_G3=gate_Q[_cfg.GATE_NAMES["G3"]],
        ccf_adjusted=False,
        ccf_uplift=CCF_UPLIFT,
    )


@pytest.fixture(name="branches_ccf")
def fixture_branches_ccf(gate_Q):
    return build_event_tree(
        lambda_IE=_cfg.LAMBDA_IE_CENTRAL,
        Q_G1=gate_Q[_cfg.GATE_NAMES["G1"]],
        Q_G2=gate_Q[_cfg.GATE_NAMES["G2"]],
        Q_G3=gate_Q[_cfg.GATE_NAMES["G3"]],
        ccf_adjusted=True,
        ccf_uplift=CCF_UPLIFT,
    )


@pytest.fixture(name="cft")
def fixture_cft(gate_Q):
    return consequence_frequency_table(gate_Q, include_ccf=True, ccf_uplift=CCF_UPLIFT)


@pytest.fixture(name="gate_Q_csv")
def fixture_gate_Q_csv(tmp_path) -> "Path":
    """Write a minimal gate_Q_table.csv matching Phase 5 format."""
    content = (
        "node,node_type,gate_type,Q,confidence,parent_component,fraction_estimate,t\n"
        f"TE: Turbine Overspeed → Catastrophic Structural Failure,"
        f"Gate,AND,{Q_TOP},computed,—,1.0,365d\n"
        f"  G1: Pitch System fails to feather,"
        f"Gate,OR,{Q_G1},computed,—,1.0,365d\n"
        f"    G1a: Control/Encoder fault,"
        f"BasicEvent,—,0.08201660687363882,fitted_tier_a,Pitch System,0.45,365d\n"
        f"    G1b: Battery/Electrical supply fault,"
        f"BasicEvent,—,0.06379069423505243,fitted_tier_a,Pitch System,0.35,365d\n"
        f"    G1c: Actuator mechanical failure,"
        f"BasicEvent,—,0.03645182527717281,fitted_tier_a,Pitch System,0.2,365d\n"
        f"  G2: Mechanical brake fails to engage,"
        f"Gate,OR,{Q_G2},computed,—,1.0,365d\n"
        f"    G2a: Hydraulic supply loss,"
        f"BasicEvent,—,0.10777799999999997,fitted_tier_b,Hydraulic System,0.6,365d\n"
        f"    G2b: Brake hardware fault,"
        f"BasicEvent,—,0.03186360000000001,assumed_placeholder,Mechanical Brake,0.4,365d\n"
        f"  G3: SCADA overspeed trip fails to activate,"
        f"Gate,OR,{Q_G3},computed,—,1.0,365d\n"
        f"    G3a: Communication/fieldbus loss,"
        f"BasicEvent,—,0.005870400000000009,posterior_informed,SCADA/Communication,0.6,365d\n"
        f"    G3b: Safety relay/electrical fault,"
        f"BasicEvent,—,0.02713199999999998,assumed_placeholder,Electrical Safety System,0.4,365d\n"
    )
    p = tmp_path / "gate_Q_table.csv"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture(name="ccf_csv")
def fixture_ccf_csv(tmp_path) -> "Path":
    """Write a minimal ccf_sensitivity.csv matching Phase 5 format."""
    content = (
        "beta_G1G2,Q_top_independent,Q_top_ccf_adjusted,Q_CCF_G1G2,ccf_risk_uplift,return_period_ccf\n"
        f"0.0,{Q_TOP},{Q_TOP},0.0,1.0,1300.38\n"
        f"0.1,{Q_TOP},{Q_TOP * CCF_UPLIFT},0.017963,{CCF_UPLIFT},824.50\n"
    )
    p = tmp_path / "ccf_sensitivity.csv"
    p.write_text(content, encoding="utf-8")
    return p