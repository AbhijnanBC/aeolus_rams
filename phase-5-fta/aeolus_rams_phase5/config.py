"""
aeolus_rams_phase5.config
==========================
Central configuration for Phase 5 FTA.

Q values are loaded at import time from Phase 3's component_rt_table.csv.
No Q value is hard-coded in any other module — all modules import from here.

Sub-cause fractions are engineering estimates derived from the relative
frequency of event description types in Phase 1's tagged_events.csv.
These are explicitly marked as estimates and swept in ccf_sensitivity.csv.

Phase 3 cross-validation (verified by hand calculation):
    λ_system = 0.002184 /day  [from phase-3-rbd/outputs/system_reliability_table.csv]
    Q_pitch(1yr)  = 0.171903  [from component_rt_table.csv Q_365d, confirmed: 1-0.828097]
    Q_hyd(1yr)    = 0.179630  [1-0.820370]
    Q_brake(1yr)  = 0.079659  [1-0.920341]
    Q_scada(1yr)  = 0.009784  [1-0.990216]
    Q_esys(1yr)   = 0.067830  [1-0.932170]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Upstream file paths
# ---------------------------------------------------------------------------

PHASE3_RT_TABLE = Path("../phase-3-rbd/outputs/component_rt_table.csv")
DEFAULT_OUTPUT_DIR = Path("outputs")


# ---------------------------------------------------------------------------
# Q-value loader
# ---------------------------------------------------------------------------

# Required component names — must match Phase 3 component_rt_table.csv exactly
REQUIRED_COMPONENTS = [
    "Pitch System",
    "Hydraulic System",
    "Mechanical Brake",
    "SCADA/Communication",
    "Electrical Safety System",
    "Yaw System",
    "Transformer",
    "Gearbox",
    "Main/Rotor Bearing",
    "Cooling System",
    "Grounding/Lightning Protection",
    "Converter",
    "Generator",
]

# Column names for unreliability at 1-year and 5-year horizons.
# In component_rt_table.csv: R_365d and R_1825d exist, so Q = 1 - R.
# The CSV does not have a Q column — we derive it here.
_RT_R_1YR_COL = "R_365d"
_RT_R_5YR_COL = "R_1825d"


def load_Q_values(
    rt_table_path: str | Path = PHASE3_RT_TABLE,
) -> tuple[dict[str, float], dict[str, float]]:
    """Load Q(1yr) and Q(5yr) from Phase 3's component_rt_table.csv.

    Returns
    -------
    (Q_1yr, Q_5yr) : tuple of dicts mapping component name → Q(t).

    Raises
    ------
    FileNotFoundError
        If the Phase 3 output file cannot be found.
    KeyError
        If a required component is absent from the table.
    ValueError
        If any Q value is outside [0, 1].
    """
    path = Path(rt_table_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Phase 3 component_rt_table.csv not found at {path.resolve()}. "
            "Run aeolus-rams-phase3 first."
        )

    df = pd.read_csv(path)
    df = df.set_index("component")

    # Validate all required components are present
    missing = [c for c in REQUIRED_COMPONENTS if c not in df.index]
    if missing:
        raise KeyError(f"Missing components in Phase 3 RT table: {missing}")

    Q_1yr: dict[str, float] = {}
    Q_5yr: dict[str, float] = {}

    for comp in REQUIRED_COMPONENTS:
        r1 = float(df.loc[comp, _RT_R_1YR_COL])
        r5 = float(df.loc[comp, _RT_R_5YR_COL])
        q1, q5 = 1.0 - r1, 1.0 - r5

        if not (0.0 <= q1 <= 1.0):
            raise ValueError(f"{comp}: Q(1yr) = {q1} out of [0,1]")
        if not (0.0 <= q5 <= 1.0):
            raise ValueError(f"{comp}: Q(5yr) = {q5} out of [0,1]")

        Q_1yr[comp] = q1
        Q_5yr[comp] = q5

    return Q_1yr, Q_5yr


# ---------------------------------------------------------------------------
# Fallback hard-coded Q values (for tests and offline runs)
# Sourced from phase-3-rbd/outputs/component_rt_table.csv
# ---------------------------------------------------------------------------

FALLBACK_Q_1YR: dict[str, float] = {
    "Pitch System":               0.171903,   # 1 - 0.828097
    "Hydraulic System":           0.179630,   # 1 - 0.820370
    "Mechanical Brake":           0.079659,   # 1 - 0.920341
    "SCADA/Communication":        0.009784,   # 1 - 0.990216
    "Electrical Safety System":   0.067830,   # 1 - 0.932170
    "Yaw System":                 0.081434,   # 1 - 0.918566
    "Transformer":                0.059059,   # 1 - 0.940941
    "Gearbox":                    0.012945,   # 1 - 0.987055
    "Main/Rotor Bearing":         0.012391,   # 1 - 0.987609
    "Cooling System":             0.032659,   # 1 - 0.967341
    "Grounding/Lightning Protection": 0.024707, # 1 - 0.975293
    "Converter":                  0.009747,   # 1 - 0.990253
    "Generator":                  0.008617,   # 1 - 0.991383
}

FALLBACK_Q_5YR: dict[str, float] = {
    "Pitch System":               0.610391,   # 1 - 0.389609
    "Hydraulic System":           0.628220,   # 1 - 0.371780
    "Mechanical Brake":           0.339547,   # 1 - 0.660453
    "SCADA/Communication":        0.047944,   # 1 - 0.952056
    "Electrical Safety System":   0.296023,   # 1 - 0.703977
    "Yaw System":                 0.345888,   # 1 - 0.654112
    "Transformer":                0.262293,   # 1 - 0.737707
    "Gearbox":                    0.063036,   # 1 - 0.936964
    "Main/Rotor Bearing":         0.060407,   # 1 - 0.939593
    "Cooling System":             0.152896,   # 1 - 0.847104
    "Grounding/Lightning Protection": 0.117518, # 1 - 0.882482
    "Converter":                  0.047769,   # 1 - 0.952231
    "Generator":                  0.042327,   # 1 - 0.957673
}


# ---------------------------------------------------------------------------
# Sub-cause fraction definitions
# ---------------------------------------------------------------------------
# Fractions are engineering estimates from Phase 1 tagged_events.csv
# event-description frequency analysis. Explicitly flagged as estimates —
# the sensitivity sweep in ccf.py varies these ±50%.
#
# For G1 (Pitch System, OR gate — 3 sub-causes):
#   G1a: Control/Encoder — Beckhoff card fault, axis encoder signal lost, slip ring
#   G1b: Battery/Electrical — Hub battery charger, DC-link fault, rewiring fault
#   G1c: Actuator Mechanical — Pitch motor seized, bearing jam, grease collector
#   Source: fraction of pitch failure-mode descriptions in tagged_events.csv
G1_FRACTIONS = {"G1a": 0.45, "G1b": 0.35, "G1c": 0.20}

# For G2 (Mechanical Brake, OR gate — 2 sub-causes):
#   G2a: Hydraulic Supply Loss — feeds from Hydraulic System Q (CCF pathway with G1)
#        Phase 1 FMECA: "Rotorbrake and Hydraulic problemes" confirms shared path
#   G2b: Brake Hardware Fault — caliper/disc/24VAC from Mechanical Brake Q
G2_FRACTIONS = {"G2a_hyd": 0.60, "G2b_brake": 0.40}

# For G3 (SCADA/Controller trip, OR gate — 2 sub-causes):
#   G3a: Communication/Fieldbus — BK1120 module, NC300 fault (real CARE events)
#        Q from SCADA/Communication × 0.60
#   G3b: Safety Relay/Electrical — Safety chain relay, RCD fault (CARE events)
#        Q from Electrical Safety System × 0.40
G3_FRACTIONS = {"G3a_scada": 0.60, "G3b_esys": 0.40}


# ---------------------------------------------------------------------------
# CCF β-factor parameters (IEC 61508 β-factor method)
# ---------------------------------------------------------------------------

#: Central estimate of β_G1G2 — fraction of G1/G2 simultaneous failures
#: caused by shared hydraulic infrastructure (Hydraulic System failure
#: can disable both pitch actuators and brake caliper).
#: Evidence: Phase 1 FMECA entry "Rotorbrake and Hydraulic problemes".
#: Range: 0.05 (low CCF) to 0.20 (high CCF per IEC 61508 Annex D).
CCF_BETA_CENTRAL = 0.10
CCF_BETA_RANGE = np.linspace(0.0, 0.20, 21)   # 0.0, 0.01, ..., 0.20


# ---------------------------------------------------------------------------
# Phase 2 bootstrap CI for Pitch System MTBF (for uncertainty propagation)
# Phase 3 report: "Bootstrap 95% CI on β spans [0.46, 2.81]"
# MTBF point and CI: from Phase 2 mtbf_table.csv (Pitch System, fitted_tier_a)
# ---------------------------------------------------------------------------

PITCH_MTBF_POINT = 1936.377    # days (Phase 3 component_rt_table.csv)
PITCH_MTBF_CI_LOW = 1017.0     # days (approx from β=2.81 Weibull upper end)
PITCH_MTBF_CI_HIGH = 2659.0    # days (approx from β=0.46 lower end)

# For placeholder components (assumed_placeholder confidence):
# Use ±50% range on MTBF to reflect literature-prior uncertainty.
PLACEHOLDER_MTBF_UNCERTAINTY_FRACTION = 0.50

# Samples for probabilistic FTA
N_PROBABILISTIC_SAMPLES = 50_000
PROBABILISTIC_SEED = 42

# Expected Q_top (pre-computed, for pipeline sanity check)
# Verified by hand calculation (see Section 0 preamble)
EXPECTED_Q_TOP_1YR = 7.690e-4    # independent assumption, 1yr
EXPECTED_Q_TOP_TOLERANCE = 0.05    # 5% tolerance for sanity check