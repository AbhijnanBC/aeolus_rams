"""Phase 7 test fixtures."""
from __future__ import annotations

import numpy as np
import pytest

from aeolus_rams_phase7.config import (
    WeibullParams, ReplacementCosts,
    LITERATURE_WEIBULL, REPLACEMENT_COSTS,
    LAMBDA_G3b, PHASE5_Q_G3b, TAU_IMPLIED_G3b_DAYS,
)


@pytest.fixture
def bearing_params() -> WeibullParams:
    return LITERATURE_WEIBULL["Main/Rotor Bearing"]


@pytest.fixture
def gearbox_params() -> WeibullParams:
    return LITERATURE_WEIBULL["Gearbox"]


@pytest.fixture
def generator_params() -> WeibullParams:
    return LITERATURE_WEIBULL["Generator"]


@pytest.fixture
def bearing_costs() -> ReplacementCosts:
    return REPLACEMENT_COSTS["Main/Rotor Bearing"]


@pytest.fixture
def gearbox_costs() -> ReplacementCosts:
    return REPLACEMENT_COSTS["Gearbox"]


@pytest.fixture
def beta_lt1_params() -> WeibullParams:
    """Pitch System parameters: β < 1, no T* should be returned."""
    from aeolus_rams_phase7.config import PITCH_BETA, PITCH_ETA
    return WeibullParams(
        component="Pitch System",
        beta=PITCH_BETA, eta=PITCH_ETA,
        source="fitted_tier_a",
        citation="Phase 2 mtbf_table.csv",
    )


@pytest.fixture
def beta_lt1_costs() -> ReplacementCosts:
    return ReplacementCosts("Pitch System", Cp=50_000, Cf=200_000)


@pytest.fixture
def lambda_G3b() -> float:
    return LAMBDA_G3b