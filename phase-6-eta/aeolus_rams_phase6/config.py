"""
aeolus_rams_phase6.config
==========================
Phase 6 ETA configuration.

Design principle: Q values are NEVER hard-coded here. They are read at
runtime from Phase 5's gate_Q_table.csv by load_gate_Q_values() in
event_tree.py. This file contains only structural constants (gate names,
IEC matrix definition, λ_IE sweep range) that cannot be derived from
Phase 5 outputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ── Phase 5 output paths (resolved relative to this package's parent dir) ──
_THIS_DIR = Path(__file__).parent.parent          # phase-6-eta/
PHASE5_OUTPUT_DIR   = _THIS_DIR / ".." / "phase-5-fta" / "outputs"
PHASE5_GATE_Q_TABLE = PHASE5_OUTPUT_DIR / "gate_Q_table.csv"
PHASE5_CCF_CSV      = PHASE5_OUTPUT_DIR / "ccf_sensitivity.csv"
PHASE5_IMPORTANCE   = PHASE5_OUTPUT_DIR / "importance_table.csv"
DEFAULT_OUTPUT_DIR  = _THIS_DIR / "outputs"

# ── Initiating event frequencies (per turbine per year) ───────────────────
LAMBDA_IE_CENTRAL  = 0.50   # Grid fault / load rejection — central estimate
LAMBDA_IE_LOW      = 0.10   # Low grid-fault environment
LAMBDA_IE_HIGH     = 2.00   # High grid-fault environment (pessimistic)
LAMBDA_IE_GUST     = 0.05   # IE-2: Extreme gust beyond cut-out
LAMBDA_IE_RANGE    = np.array([0.05, 0.10, 0.20, 0.50, 1.00, 2.00])

# ── CCF parameters (loaded from Phase 5 ccf_sensitivity.csv at runtime) ──
CCF_BETA_CENTRAL   = 0.10
# Fallback if ccf_sensitivity.csv is unavailable — exact value from Phase 5
CCF_UPLIFT_CENTRAL = 1.5772   # ccf_risk_uplift at beta=0.10 (Phase 5 verified)

# ── Farm parameters (Phase 4) ─────────────────────────────────────────────
N_TURBINES  = 22
DESIGN_LIFE = 25   # years

# ── Gate node names — must match Phase 5 gate_Q_table.csv exactly ─────────
# Verified against committed CSV: leading whitespace stripped by load function.
GATE_NAMES: dict[str, str] = {
    "G1": "G1: Pitch System fails to feather",
    "G2": "G2: Mechanical brake fails to engage",
    "G3": "G3: SCADA overspeed trip fails to activate",
    "TE": "TE: Turbine Overspeed → Catastrophic Structural Failure",
}

# ── IEC 61400-1 Ed.4 Annex K risk matrix ─────────────────────────────────
# Frequency category lower boundaries (events per turbine per year)
FREQ_LOWER_BOUNDS: dict[str, float] = {
    "F1": 1e-1,
    "F2": 1e-2,
    "F3": 1e-3,
    "F4": 1e-4,
    "F5": 0.0,
}

FREQ_LABELS: list[str] = ["F1", "F2", "F3", "F4", "F5"]
SEV_LABELS:  list[str] = ["A", "B", "C", "D", "E"]

# Consequence category assigned to each ETA branch
BRANCH_SEVERITY: dict[str, str] = {
    "Branch_1_SAFE_controlled":  "E",   # Negligible — production interruption only
    "Branch_2_SAFE_brake":       "D",   # Minor — brake wear, inspection needed
    "Branch_3_NEAR_MISS_SCADA":  "C",   # Significant — structural fatigue inspection
    "Branch_4_CATASTROPHIC":     "A",   # Extreme — blade ejection / tower collapse
}

# IEC 61400-1 risk matrix cell lookup: (severity, freq_category) → acceptability
RISK_MATRIX: dict[tuple[str, str], str] = {
    ("A", "F1"): "UNACCEPTABLE", ("A", "F2"): "UNACCEPTABLE",
    ("A", "F3"): "UNACCEPTABLE", ("A", "F4"): "ALARP",
    ("A", "F5"): "ACCEPTABLE",
    ("B", "F1"): "UNACCEPTABLE", ("B", "F2"): "UNACCEPTABLE",
    ("B", "F3"): "ALARP",        ("B", "F4"): "ACCEPTABLE",
    ("B", "F5"): "ACCEPTABLE",
    ("C", "F1"): "UNACCEPTABLE", ("C", "F2"): "ALARP",
    ("C", "F3"): "ACCEPTABLE",   ("C", "F4"): "ACCEPTABLE",
    ("C", "F5"): "ACCEPTABLE",
    ("D", "F1"): "ALARP",        ("D", "F2"): "ACCEPTABLE",
    ("D", "F3"): "ACCEPTABLE",   ("D", "F4"): "ACCEPTABLE",
    ("D", "F5"): "ACCEPTABLE",
    ("E", "F1"): "ACCEPTABLE",   ("E", "F2"): "ACCEPTABLE",
    ("E", "F3"): "ACCEPTABLE",   ("E", "F4"): "ACCEPTABLE",
    ("E", "F5"): "ACCEPTABLE",
}

# BoP cable failure point (Phase 4) — placed on risk matrix for comparison
# λ_cable = 365 / MTBF_cable_days ≈ 365/1300 ≈ 0.281/yr → F1–F2 boundary
BOP_CABLE_LAMBDA_PER_TURBINE = 365.0 / 1300.0   # Phase 4: cable MTBF ≈ 1300 days
BOP_CABLE_SEVERITY = "D"                          # Economic loss only, no safety risk


def freq_category(lambda_per_turbine: float) -> str:
    """Map an absolute frequency to an IEC 61400-1 frequency category label."""
    for cat in FREQ_LABELS:
        if lambda_per_turbine >= FREQ_LOWER_BOUNDS[cat]:
            return cat
    return "F5"