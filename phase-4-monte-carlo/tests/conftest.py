"""
Phase 4 test fixtures.

Minimal parameter sets for fast unit tests — not the full 13-component
baseline. The key fixture is a two-component turbine (Pitch + Hydraulic)
which exercises all code paths while running in milliseconds.
"""
from __future__ import annotations

import numpy as np
import pytest

from aeolus_rams_phase4.component_sampler import ComponentParams


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic generator for all tests. Same seed = same results."""
    return np.random.default_rng(42)


@pytest.fixture
def pitch_params() -> ComponentParams:
    """Pitch System (Tier A, highest-criticality turbine component)."""
    return ComponentParams(
        name="Pitch System",
        mtbf_days=1936.377,
        mttr_raw_days=3.0,
        access_fraction=0.60,
        confidence="fitted_tier_a",
    )


@pytest.fixture
def hydraulic_params() -> ComponentParams:
    """Hydraulic System (Tier B, highest-lambda turbine component)."""
    return ComponentParams(
        name="Hydraulic System",
        mtbf_days=1844.703,
        mttr_raw_days=5.0,
        access_fraction=0.60,
        confidence="fitted_tier_b",
    )


@pytest.fixture
def two_component_params(pitch_params, hydraulic_params) -> list[ComponentParams]:
    """Minimal two-component turbine — fast tests only."""
    return [pitch_params, hydraulic_params]


@pytest.fixture
def perfect_bop_params() -> list[ComponentParams]:
    """BoP with MTBF → ∞ (never fails). Used to isolate k-of-N tests."""
    return [
        ComponentParams(
            name="Perfect Substation",
            mtbf_days=1e12,
            mttr_raw_days=0.001,
            access_fraction=1.0,
            confidence="test_only",
            is_bop=True,
        ),
        ComponentParams(
            name="Perfect Cable",
            mtbf_days=1e12,
            mttr_raw_days=0.001,
            access_fraction=1.0,
            confidence="test_only",
            is_bop=True,
        ),
    ]