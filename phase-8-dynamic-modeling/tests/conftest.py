"""
Phase 8 test fixtures.

All fixtures are self-contained — they do not depend on upstream
Phase 2 or Phase 7 CSV files being present on disk. This ensures
tests pass in isolated CI environments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aeolus_rams_phase8 import config
from aeolus_rams_phase8 import synthetic_data


# ---------------------------------------------------------------------------
# Minimal bearing failure fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bearing_failures_small() -> pd.DataFrame:
    """10 rows: 7 failures (batches 1/2/3) + 3 right-censored. Fixed seed."""
    rng = np.random.default_rng(99)
    rows = []
    for i in range(7):
        rows.append({
            "failure_id":      i + 1,
            "event_time_days": float(rng.exponential(5000.0)),
            "event_observed":  1,
            "batch":           (i % 3) + 1,
            "synthetic":       True,
        })
    for j in range(3):
        rows.append({
            "failure_id":      8 + j,
            "event_time_days": float(rng.uniform(1000, 1825)),
            "event_observed":  0,
            "batch":           0,
            "synthetic":       True,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Minimal Cox PH covariate fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def covariates_small() -> pd.DataFrame:
    """50-row synthetic covariate dataset for fast Cox PH tests."""
    return synthetic_data.generate_covariates(seed=777)[:50].copy()


# ---------------------------------------------------------------------------
# Phase 7 cost fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def phase7_pitch_costs() -> dict:
    return {
        "Cp": config.PITCH_Cp,
        "Cf": config.PITCH_Cf,
    }


@pytest.fixture
def phase2_bearing_params() -> dict:
    return {
        "beta": config.BEARING_BETA,
        "eta":  config.BEARING_ETA,
    }


# ---------------------------------------------------------------------------
# Pre-computed posterior grid at low resolution (fast tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def posterior_grid_small(bearing_failures_small) -> dict:
    """30×30 posterior grid for fast numerical tests."""
    from aeolus_rams_phase8 import bayesian_update

    fail_mask = bearing_failures_small["event_observed"] == 1
    fail_times = bearing_failures_small[fail_mask]["event_time_days"].values
    cens_times = bearing_failures_small[~fail_mask]["event_time_days"].values

    return bayesian_update.compute_posterior_grid(
        fail_times=fail_times,
        censor_times=cens_times,
        n_beta=30,
        n_eta=30,
    )