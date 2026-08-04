"""Phase 5 test fixtures — minimal and fast."""
from __future__ import annotations

import pytest

from aeolus_rams_phase5 import config as _cfg
from aeolus_rams_phase5.fault_tree import build_overspeed_fault_tree, Gate


@pytest.fixture
def Q_1yr() -> dict[str, float]:
    """Fallback Q values from Phase 3 (used when CSV not available in tests)."""
    return _cfg.FALLBACK_Q_1YR.copy()


@pytest.fixture
def Q_5yr() -> dict[str, float]:
    return _cfg.FALLBACK_Q_5YR.copy()


@pytest.fixture
def top_gate_1yr(Q_1yr) -> Gate:
    """Fully-built overspeed fault tree at t=1yr."""
    return build_overspeed_fault_tree(Q_1yr)


@pytest.fixture
def top_gate_5yr(Q_5yr) -> Gate:
    return build_overspeed_fault_tree(Q_5yr)