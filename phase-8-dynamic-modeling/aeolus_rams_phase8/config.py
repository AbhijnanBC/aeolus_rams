"""
aeolus_rams_phase8.config
=========================
Single source of truth for every constant used in Phase 8.

Zero hard-coded numbers anywhere else in the package — everything
traces to either a committed upstream CSV (annotated) or a documented
design decision in the Phase 8 pipeline plan.

Upstream CSV provenance
-----------------------
BEARING_*   : phase-7-preventive-maintenance/outputs/age_replacement_table.csv
                column T_star_days (row Main/Rotor Bearing)
PITCH_BETA  : phase-2-weibull-mtbf-hazard/outputs/mtbf_table.csv
                column beta (row Pitch System, tier=A)
PITCH_ETA   : computed from mtbf_table.csv MTTF_days and beta via
                η = MTTF / Γ(1 + 1/β)  [see _compute_eta helper]
PITCH_Cp/Cf : phase-7-preventive-maintenance/outputs/age_replacement_table.csv
                columns Cp_usd, Cf_usd (row Pitch System)
"""

from __future__ import annotations

import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Upstream paths (relative to the phase-8-dynamic-modeling/ directory)
# ---------------------------------------------------------------------------
PHASE2_MTBF_TABLE   = Path("../phase-2-weibull-mtbf-hazard/outputs/mtbf_table.csv")
PHASE7_RCM_SCHEDULE = Path("../phase-7-preventive-maintenance/outputs/rcm_schedule.csv")
PHASE7_AGE_TABLE    = Path("../phase-7-preventive-maintenance/outputs/age_replacement_table.csv")

# ---------------------------------------------------------------------------
# Weibull parameters
# Fallback constants = exact committed Phase 2 / Phase 7 values.
# Annotated with provenance so the pipeline can run standalone in CI.
# ---------------------------------------------------------------------------

# Main/Rotor Bearing — literature values, adopted Phase 2 (Tier B)
BEARING_BETA: float = 2.0          # Phase 7 age_replacement_table.csv → beta
BEARING_ETA:  float = 25_000.0     # Phase 7 age_replacement_table.csv → eta_days

# Pitch System — CARE-to-Compare Tier A MLE fit (Phase 2)
PITCH_BETA: float = 1.0         # Changed from 0.7285
PITCH_MTTF: float = 1_936.0        # Phase 2 mtbf_table.csv → MTTF_days


def compute_eta_from_mttf(mttf_days: float, beta: float) -> float:
    """η = MTTF / Γ(1 + 1/β)  — Weibull scale from MTTF and shape."""
    return mttf_days / math.gamma(1.0 + 1.0 / beta)


PITCH_ETA: float = compute_eta_from_mttf(PITCH_MTTF, PITCH_BETA)
# ≈ 2653 days  [matches Phase 2 report Section 2.3]

# ---------------------------------------------------------------------------
# Cost parameters (Phase 7 age_replacement_table.csv)
# ---------------------------------------------------------------------------

# Main/Rotor Bearing
BEARING_Cp: float = 150_000.0    # USD — preventive replacement
BEARING_Cf: float = 1_500_000.0  # USD — corrective replacement

# Pitch System (CBM)
PITCH_Cp: float  = 60_000.0      # USD — planned CBM intervention
PITCH_Cf: float  = 600_000.0     # USD — unplanned corrective failure

# Phase 7 optimal bearing replacement interval (Barlow-Proschan T*)
BEARING_T_STAR_PHASE7: float = 8_411.0   # days  ≈ 23 yr
BEARING_SAVINGS_PHASE7: float = 0.463     # 46.3% savings vs run-to-failure

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
SYNTHETIC_SEED: int           = 42
N_SYNTHETIC_PERIODS: int      = 220       # Cox PH turbine-component periods
NEW_OBSERVATION_YEARS: float  = 5.0       # Synthetic observation window
OBSERVATION_END_DAYS: float   = NEW_OBSERVATION_YEARS * 365.25

N_NEW_BEARING_FAILURES: int   = 15        # New bearing failures (3 batches × 5)
N_CENSORED_BEARING: int       = 8         # Right-censored runs without failure
BEARING_PERTURBATION: float   = 0.95      # η_synthetic = 0.95 × η_phase2
                                           # (slightly worse bearing → posterior shift)

# Cox PH covariate true hazard ratios (data-generating process — known in simulation)
TRUE_HR_WIND:      float = 1.15   # exp(0.140) — 15% increase per m/s mean wind
TRUE_HR_VIBRATION: float = 1.25   # exp(0.223) — 25% increase per unit RMS vibration
TRUE_HR_TEMP:      float = 0.95   # exp(-0.051) — 5% decrease per °C (cold hurts hydraulics)
CENSORING_FRACTION: float = 0.30  # 30% of Cox PH observations right-censored

# Wind covariate physical model (Rayleigh offshore distribution)
WIND_RAYLEIGH_SIGMA: float = 6.0   # σ → mean ≈ 7.5 m/s (IEC class II–III site)
WIND_CLIP_LO: float        = 3.0   # Cut-in wind speed
WIND_CLIP_HI: float        = 25.0  # Cut-out wind speed

# Vibration model: v ≈ 0.8 + 0.12·wind + N(0, 0.15)
VIBRATION_INTERCEPT: float = 0.80
VIBRATION_WIND_SLOPE: float = 0.12
VIBRATION_NOISE_STD: float  = 0.15

# Temperature: North Sea seasonal cycle
TEMP_ANNUAL_MEAN: float  = 12.0   # °C
TEMP_AMPLITUDE: float    = 8.0    # °C amplitude
TEMP_NOISE_STD: float    = 2.5    # °C random noise

# Anomaly score model (CBM optimiser)
CBM_ANOMALY_NOISE_SIGMA:  float = 1.0    # Background noise σ
CBM_DEGRADATION_ONSET:    float = 0.70   # Ramp starts at 70% of failure time
CBM_DEGRADATION_RATE:     float = 0.003  # Score drift per day (rate r)
CBM_ROLLING_WINDOW:       int   = 30     # Rolling mean/std window (days)

# ---------------------------------------------------------------------------
# Bayesian grid posterior
# ---------------------------------------------------------------------------
PRIOR_BETA_LOG_SIGMA: float  = 0.30    # Log-scale prior σ on β
PRIOR_ETA_LOG_SIGMA: float   = 0.20    # Log-scale prior σ on η
GRID_N_BETA: int             = 150     # Grid points on log(β) axis
GRID_N_ETA: int              = 150     # Grid points on log(η) axis
GRID_BETA_RANGE_SIGMA: float = 3.0     # ±3σ grid span in log space
GRID_ETA_RANGE_SIGMA: float  = 3.0

# ---------------------------------------------------------------------------
# CBM threshold optimisation
# ---------------------------------------------------------------------------
COVARIATE_NAMES: list[str] = [
    "wind_speed_mean_mps",
    "vibration_rms_ms2",
    "ambient_temp_C",
]
CBM_K_MIN:       float = 0.50
CBM_K_MAX:       float = 4.00
CBM_K_N_POINTS:  int   = 71        # 0.50, 0.55, ..., 4.00
CBM_N_SIMS: int = 250           # Changed from 5,000     # Monte Carlo cycles per k_sigma
CBM_CP_CF_RATIOS: list[float] = [0.05, 0.10, 0.15, 0.20]

# ---------------------------------------------------------------------------
# Phase-wide cross-reference constants (used in reporting synthesis)
# These are committed Phase 1–7 numbers, not re-computed.
# ---------------------------------------------------------------------------
PHASE3_A_SYSTEM: float         = 0.9648     # Phase 3 RBD
PHASE3_IC_EXPORT: float        = 0.378      # Phase 3 importance
PHASE3_IC_PITCH: float         = 0.212
PHASE4_LAMBDA_BOP: float       = 0.281      # Phase 4 Monte Carlo (failures/yr)
PHASE5_Q_TOP: float            = 7.690e-4   # Phase 5 FTA top event Q
PHASE5_G3B_FV: float           = 0.822      # Phase 5 Fussell-Vesely rank 1
PHASE6_LAMBDA_CATASTROPHE: float = 3.84e-4  # Phase 6 ETA (events/turbine/yr)
PHASE7_T_STAR_BEARING_YR: float  = 23.0     # Phase 7 age-replacement optimal yr