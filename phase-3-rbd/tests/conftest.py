"""
Synthetic fixtures for Phase 3.

The fixture mtbf_table_path() writes a minimal mtbf_table.csv whose
schema is byte-for-byte identical to what aeolus_rams_phase2 actually
produces:

  component, tier, n_incidents_phase1, n_tbf_used, mtbf_days,
  beta, eta, lambda, ci_low_days, ci_high_days, confidence, source

It contains all 13 Phase 1 taxonomy components:
  - Pitch System:     Tier A, MTBF=1936.38, source includes "preferred over
                      exponential: False" (the AIC-preferred-exponential flag)
  - Hydraulic System: Tier B, MTBF=1844.70
  - 5 Tier C with posterior_informed MTBFs (Bayesian-updated)
  - 6 Tier C with mtbf_days=NaN (not_yet_sourced) → trigger Option A injection

Real values from the actual Phase 2 run (Abhijnan's execution plan):
  Pitch: β=0.728, η=1586.6, MTBF=1936.38 (Weibull MLE, AIC→exp)
  Hydraulic: λ=0.000542, MTBF=1844.70
  Gearbox posterior:       28033.15 days
  Main/Rotor Bearing:      29293.57 days
  SCADA/Communication:     37150.25 days
  Converter:               37289.56 days
  Generator:               42203.02 days

Using these real numbers rather than synthetic round numbers ensures that
tests which compute known R(t) values are verified against the actual
Phase 2 output, not a test-only stand-in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# Exact rows matching Phase 2 schema — do NOT round the MTBF values;
# tests assert to 4 decimal places and will catch drift.
_MTBF_TABLE_ROWS = [
    # Tier A
    {
        "component": "Pitch System",
        "tier": "A",
        "n_incidents_phase1": 11,
        "n_tbf_used": 9,
        "mtbf_days": 1936.3776678104537,
        "beta": 0.728,
        "eta": 1586.6,
        "lambda": None,
        "ci_low_days": 900.0,
        "ci_high_days": 3200.0,
        "confidence": "fitted_tier_a",
        "source": "Weibull MLE fit (9 TBF intervals); preferred over exponential: False",
    },
    # Tier B
    {
        "component": "Hydraulic System",
        "tier": "B",
        "n_incidents_phase1": 6,
        "n_tbf_used": 5,
        "mtbf_days": 1844.7037037037037,
        "beta": None,
        "eta": None,
        "lambda": 0.000542,
        "ci_low_days": 1100.0,
        "ci_high_days": 3400.0,
        "confidence": "fitted_tier_b",
        "source": "Exponential MLE fit (5 TBF intervals)",
    },
    # Tier C — posterior_informed
    {
        "component": "Gearbox",
        "tier": "C",
        "n_incidents_phase1": 8,
        "n_tbf_used": 6,
        "mtbf_days": 28033.147886957817,
        "beta": None, "eta": None, "lambda": None,
        "ci_low_days": None, "ci_high_days": None,
        "confidence": "posterior_informed",
        "source": "Gamma-Poisson posterior; Carroll et al. (2016)",
    },
    {
        "component": "Main/Rotor Bearing",
        "tier": "C",
        "n_incidents_phase1": 2,
        "n_tbf_used": 0,
        "mtbf_days": 29293.568135823352,
        "beta": None, "eta": None, "lambda": None,
        "ci_low_days": None, "ci_high_days": None,
        "confidence": "posterior_informed",
        "source": "Gamma-Poisson posterior; Hart et al. (2020)",
    },
    {
        "component": "SCADA/Communication",
        "tier": "C",
        "n_incidents_phase1": 1,
        "n_tbf_used": 0,
        "mtbf_days": 37150.250660361286,
        "beta": None, "eta": None, "lambda": None,
        "ci_low_days": None, "ci_high_days": None,
        "confidence": "posterior_informed",
        "source": "Gamma-Poisson posterior; Walgern et al. (2026)",
    },
    {
        "component": "Converter",
        "tier": "C",
        "n_incidents_phase1": 4,
        "n_tbf_used": 0,
        "mtbf_days": 37289.56293551928,
        "beta": None, "eta": None, "lambda": None,
        "ci_low_days": None, "ci_high_days": None,
        "confidence": "posterior_informed",
        "source": "Gamma-Poisson posterior; Walgern et al. (2026)",
    },
    {
        "component": "Generator",
        "tier": "C",
        "n_incidents_phase1": 3,
        "n_tbf_used": 0,
        "mtbf_days": 42203.017130032,
        "beta": None, "eta": None, "lambda": None,
        "ci_low_days": None, "ci_high_days": None,
        "confidence": "posterior_informed",
        "source": "Gamma-Poisson posterior; Carroll et al. (2016)",
    },
    # Tier C — not_yet_sourced (NaN mtbf → Option A injection)
    *[
        {
            "component": comp,
            "tier": "C",
            "n_incidents_phase1": 0,
            "n_tbf_used": 0,
            "mtbf_days": float("nan"),
            "beta": None, "eta": None, "lambda": None,
            "ci_low_days": None, "ci_high_days": None,
            "confidence": "not_yet_sourced",
            "source": "Not yet sourced — see literature_priors.py",
        }
        for comp in [
            "Mechanical Brake",
            "Transformer",
            "Yaw System",
            "Electrical Safety System",
            "Grounding/Lightning Protection",
            "Cooling System",
        ]
    ],
]

# Expected MTBF after Option A injection for the 6 NaN components
EXPECTED_PLACEHOLDER_MTBF = {
    "Mechanical Brake": 4_400.0,
    "Transformer": 6_000.0,
    "Yaw System": 4_300.0,
    "Electrical Safety System": 5_200.0,
    "Grounding/Lightning Protection": 14_600.0,
    "Cooling System": 11_000.0,
}


@pytest.fixture
def mtbf_table_path(tmp_path) -> str:
    """Write the synthetic mtbf_table.csv and return its path as a string."""
    df = pd.DataFrame(_MTBF_TABLE_ROWS)
    path = tmp_path / "mtbf_table.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def components(mtbf_table_path):
    """Fully-loaded and placeholder-injected ComponentRT dict."""
    from aeolus_rams_phase3.component_rt import load_all_components
    return load_all_components(mtbf_table_path)


@pytest.fixture
def mtbf_table_path_with_bad_nan(tmp_path) -> str:
    """mtbf_table.csv with a NaN component NOT in PLACEHOLDER_MTBF — triggers error."""
    rows = [
        r for r in _MTBF_TABLE_ROWS
        if r["component"] != "Mechanical Brake"
    ]
    rows.append({
        "component": "Mystery Component",
        "tier": "C", "n_incidents_phase1": 0, "n_tbf_used": 0,
        "mtbf_days": float("nan"), "beta": None, "eta": None, "lambda": None,
        "ci_low_days": None, "ci_high_days": None,
        "confidence": "not_yet_sourced", "source": "",
    })
    df = pd.DataFrame(rows)
    path = tmp_path / "bad_mtbf.csv"
    df.to_csv(path, index=False)
    return str(path)
