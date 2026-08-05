"""
aeolus_rams_phase8.synthetic_data
==================================
Reproducible synthetic data generator.  This is the single source of
randomness in Phase 8; all other modules consume its outputs deterministically.

SYNTHETIC DATA NOTICE
---------------------
All data produced by this module is synthetically generated from the
Phase 2 Weibull posterior (Main/Rotor Bearing: β=2.0, η=25000d;
Pitch System: β=0.728, η=2653d) using physically motivated covariate
distributions for a North Sea offshore wind site.

No new real SCADA failure observations exist beyond the CARE-to-Compare
dataset used in Phases 1–7.  The synthetic observations are drawn from
the Phase 2 fitted distribution with a small deterministic perturbation
(bearing η × 0.95) that creates a directionally interesting posterior
shift while remaining physically plausible.

Every output file carries column `synthetic=True` and all plots carry
the annotation `[SYNTHETIC — generated from Phase 2 Weibull parameters
+ physical forward model]`.

Reproducibility
---------------
Every public function accepts a `seed` argument (default = config.SYNTHETIC_SEED).
`np.random.seed(seed)` is called at the TOP of each function — not globally —
so unit tests can call individual functions without cross-contaminating state.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import weibull_min

from . import config


# ---------------------------------------------------------------------------
# Helper: Weibull survival function (vectorised)
# ---------------------------------------------------------------------------

def _weibull_survival(t: np.ndarray, beta: float, eta: float) -> np.ndarray:
    """R(t; β, η) = exp(−(t/η)^β)  — element-wise, safe for t=0."""
    return np.exp(-((np.maximum(t, 1e-12) / eta) ** beta))


def _weibull_sample(n: int, beta: float, eta: float, rng: np.random.Generator) -> np.ndarray:
    """
    Draw n samples from Weibull(β, η) using the numpy generator API.
    scipy.stats.weibull_min.rvs uses shape=β, scale=η, loc=0.
    We replicate that here for numpy-Generator compatibility.
    """
    # Weibull quantile function: η × (−ln U)^(1/β)
    u = rng.uniform(0.0, 1.0, size=n)
    return eta * (-np.log(u + 1e-15)) ** (1.0 / beta)


# ---------------------------------------------------------------------------
# Output 1: Bearing failure dataset for Bayesian updating
# ---------------------------------------------------------------------------

def generate_bearing_failures(seed: int = config.SYNTHETIC_SEED) -> pd.DataFrame:
    """
    Generate synthetic Main/Rotor Bearing failure events for Bayesian updating.

    Design choices
    --------------
    - β=2.0, η=25000d × 0.95 (slightly worse bearing) → posterior η̂ < η_phase2
    - 15 failures split into 3 batches of 5 (sequential update demo)
    - 8 right-censored observations (ran full 5-yr window without failure)
    - All event times > 0; censored at OBSERVATION_END_DAYS

    Returns
    -------
    DataFrame with columns:
        failure_id      int    unique row index
        event_time_days float  observed time to event (failure or censoring)
        event_observed  int    1 = failure, 0 = right-censored
        batch           int    1/2/3 for failures; 0 for censored
        synthetic       bool   always True (SYNTHETIC DATA NOTICE)
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    eta_synthetic = config.BEARING_ETA * config.BEARING_PERTURBATION
    beta = config.BEARING_BETA

    # Draw failure times
    fail_times = _weibull_sample(
        config.N_NEW_BEARING_FAILURES, beta, eta_synthetic, rng
    )
    # Clip to plausible physical range: at least 90 days, at most 60 years
    fail_times = np.clip(fail_times, 90.0, 60.0 * 365.25)

    batch_labels = np.repeat([1, 2, 3], config.N_NEW_BEARING_FAILURES // 3)

    failure_rows = []
    for i, (t, b) in enumerate(zip(fail_times, batch_labels)):
        failure_rows.append({
            "failure_id":      i + 1,
            "event_time_days": float(t),
            "event_observed":  1,
            "batch":           int(b),
            "synthetic":       True,
        })

    # Right-censored observations
    for j in range(config.N_CENSORED_BEARING):
        censored_time = rng.uniform(
            0.70 * config.OBSERVATION_END_DAYS,
            config.OBSERVATION_END_DAYS,
        )
        failure_rows.append({
            "failure_id":      config.N_NEW_BEARING_FAILURES + j + 1,
            "event_time_days": float(censored_time),
            "event_observed":  0,
            "batch":           0,   # censored not in a batch
            "synthetic":       True,
        })

    df = pd.DataFrame(failure_rows)
    return df.sort_values("failure_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Output 2: Cox PH covariate dataset for Pitch System survival analysis
# ---------------------------------------------------------------------------

def generate_covariates(seed: int = config.SYNTHETIC_SEED) -> pd.DataFrame:
    """
    Generate synthetic Pitch System operational dataset for Cox PH fitting.

    Covariate distributions (physically motivated, North Sea offshore site)
    -----------------------------------------------------------------------
    wind_speed_mean_mps  ~ Rayleigh(σ=6.0), clipped [3, 25]  (mean ≈ 7.5 m/s)
    vibration_rms_ms2    ~ 0.8 + 0.12·wind + N(0, 0.15)      (correlated with load)
    ambient_temp_C       ~ 12 + 8·sin(2π·φ) + N(0, 2.5)      (North Sea seasonal)

    Cox PH acceleration
    -------------------
    Covariates are standardised; the true log-hazard linear predictor is:
        lp = log(HR_wind)·wind_z + log(HR_vib)·vib_z + log(HR_temp)·temp_z
    Accelerated failure time: t_actual = t_baseline · exp(−lp)

    Returns
    -------
    DataFrame with columns:
        turbine_id, component, duration_days, event_observed,
        wind_speed_mean_mps, vibration_rms_ms2, ambient_temp_C,
        synthetic (always True)
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    n = config.N_SYNTHETIC_PERIODS

    # --- Covariates ---
    # Wind: Rayleigh  →  X ~ Rayleigh(σ)  ↔  X = σ · sqrt(-2 ln U)
    u_wind = rng.uniform(0.0, 1.0, n)
    wind = config.WIND_RAYLEIGH_SIGMA * np.sqrt(-2.0 * np.log(u_wind + 1e-15))
    wind = np.clip(wind, config.WIND_CLIP_LO, config.WIND_CLIP_HI)

    # Vibration: physically correlated with wind load
    vibration = (
        config.VIBRATION_INTERCEPT
        + config.VIBRATION_WIND_SLOPE * wind
        + rng.normal(0.0, config.VIBRATION_NOISE_STD, n)
    )
    vibration = np.maximum(vibration, 0.1)  # physical floor

    # Temperature: seasonal North Sea cycle
    seasonal_phase = rng.uniform(0.0, 1.0, n)  # random phase per turbine-period
    temp = (
        config.TEMP_ANNUAL_MEAN
        + config.TEMP_AMPLITUDE * np.sin(2.0 * math.pi * seasonal_phase)
        + rng.normal(0.0, config.TEMP_NOISE_STD, n)
    )

    # Standardise for Cox PH model
    wind_z = (wind - wind.mean()) / wind.std()
    vib_z  = (vibration - vibration.mean()) / vibration.std()
    temp_z = (temp - temp.mean()) / temp.std()

    # True log-hazard linear predictor
    log_hr_wind = math.log(config.TRUE_HR_WIND)
    log_hr_vib  = math.log(config.TRUE_HR_VIBRATION)
    log_hr_temp = math.log(config.TRUE_HR_TEMP)
    lp = log_hr_wind * wind_z + log_hr_vib * vib_z + log_hr_temp * temp_z

    # Baseline Pitch System lifetime from Weibull(β, η)
    t_baseline = _weibull_sample(n, config.PITCH_BETA, config.PITCH_ETA, rng)

    # Accelerated failure time: shorter life in high-hazard environments
    t_actual = t_baseline * np.exp(-lp)
    t_actual = np.maximum(t_actual, 1.0)

    # Right-censoring: 30% randomly censored
    censored_mask = rng.uniform(0.0, 1.0, n) < config.CENSORING_FRACTION
    # Censoring time drawn uniformly between 20% and 100% of actual time
    censor_time = t_actual * rng.uniform(0.20, 1.00, n)
    duration = np.where(censored_mask, censor_time, t_actual)
    event_obs = np.where(censored_mask, 0, 1).astype(int)

    df = pd.DataFrame({
        "turbine_id":          np.arange(1, n + 1),
        "component":           "Pitch System",
        "duration_days":       duration,
        "event_observed":      event_obs,
        "wind_speed_mean_mps": wind,
        "vibration_rms_ms2":   vibration,
        "ambient_temp_C":      temp,
        "synthetic":           True,
    })
    return df


# ---------------------------------------------------------------------------
# Output 3: Anomaly score trajectory (on-demand, for CBM optimiser)
# ---------------------------------------------------------------------------

def generate_anomaly_trajectory(
    t_failure: float,
    seed_offset: int = 0,
    seed: int = config.SYNTHETIC_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate one anomaly score trajectory S(t) for t = 0..t_failure.

    Model
    -----
    S(t) = background_noise(t) + degradation_signal(t)
    background_noise(t) ~ iid N(0, σ_noise)
    degradation_signal(t) = 0 for t < t_onset
                          = rate × (t − t_onset) for t >= t_onset
    where t_onset = 0.70 × t_failure

    Returns
    -------
    (time_array, score_array) both length ceil(t_failure)+1
    """
    rng = np.random.default_rng(seed + seed_offset)
    t_max = int(math.ceil(t_failure)) + 1
    t_arr = np.arange(t_max, dtype=float)

    # Background noise
    noise = rng.normal(0.0, config.CBM_ANOMALY_NOISE_SIGMA, t_max)

    # Degradation ramp
    t_onset = config.CBM_DEGRADATION_ONSET * t_failure
    ramp = np.maximum(0.0, config.CBM_DEGRADATION_RATE * (t_arr - t_onset))

    return t_arr, noise + ramp


# ---------------------------------------------------------------------------
# Master generator
# ---------------------------------------------------------------------------

def generate_all(
    seed: int = config.SYNTHETIC_SEED,
    output_dir: "Path | None" = None,
) -> dict[str, pd.DataFrame]:
    """
    Generate all synthetic datasets and optionally write to CSVs.

    Returns dict with keys: 'bearing_failures', 'covariates'
    """
    bearing_df   = generate_bearing_failures(seed=seed)
    covariates_df = generate_covariates(seed=seed)

    if output_dir is not None:
        from pathlib import Path
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        bearing_df.to_csv(out / "synthetic_failures.csv", index=False)
        covariates_df.to_csv(out / "synthetic_covariates.csv", index=False)

    return {"bearing_failures": bearing_df, "covariates": covariates_df}