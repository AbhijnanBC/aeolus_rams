"""
aeolus_rams_phase7.cbm_trigger
================================
Condition-Based Maintenance (CBM) trigger framework for components
where β ≤ 1 makes fixed-interval PM mathematically suboptimal.

Mathematical justification for CBM on Pitch System (β = 0.728):
  The Barlow-Proschan cost rate C(T) is monotonically decreasing for β ≤ 1.
  This means: the longer you run without a preventive replacement, the lower
  the cost rate. PM at any fixed interval T is always worse than running until
  a failure signal appears. Therefore, the only value-adding maintenance
  intervention is triggered by evidence of degradation — not by calendar time.

The degradation indicator in AEOLUS is the ML anomaly score produced by the
SCADA-based fault detection pipeline (Phase 1 background). Phase 7 formalises
this as a Proportional Hazards Model (PHM) CBM trigger:

    h(t | S(t)) = h₀(t) × exp(γ × S(t))

where:
    h₀(t) = λ = 1/MTBF  (baseline hazard under S=0; exponential for β≈1)
    S(t) = normalised anomaly score at time t ∈ [0, 1]
    γ = PH model coefficient (to be estimated from operational data)

Maintenance trigger rule (practical implementation):
    Trigger maintenance when S(t) ≥ S_threshold
    S_threshold = rolling_mean(S, 30d) + k_sigma × rolling_std(S, 30d)
    Default k_sigma = 2.0 (corresponds to ~2.3% false alarm rate under normality)

The k_sigma threshold is a starting point for operational tuning. The optimal
k* minimises the long-run cost:
    C(k*) = [Cp × P(trigger before failure) + Cf × P(failure before trigger)] / E[cycle_length]

This cost function requires historical data on the time-from-score-exceedance
to failure — something to collect as operational experience accumulates. Phase 7
provides the framework and the starting threshold; the tuning is Phase 8+.

Reference:
    Jardine, A.K.S., Lin, D., Banjevic, D. (2006) "A review on machinery
    diagnostics and prognostics implementing condition-based maintenance."
    Mechanical Systems and Signal Processing 20(7):1483–1510.
    Proportional Hazards CBM: Section 3.2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cbm_trigger_rule(
    anomaly_scores: np.ndarray,
    timestamps: np.ndarray,
    rolling_window_days: int = 30,
    k_sigma: float = 2.0,
    min_periods: int = 7,
) -> pd.DataFrame:
    """Apply the rolling z-score CBM trigger to an anomaly score time series.

    Maintenance is triggered when the score exceeds the rolling baseline
    by k_sigma standard deviations.

    Parameters
    ----------
    anomaly_scores : np.ndarray
        1-D array of anomaly scores (dimensionless, e.g. from an ML detector).
        Does not need to be normalised — the trigger is relative to the rolling
        baseline, not an absolute threshold.
    timestamps : np.ndarray
        Corresponding timestamps. Can be datetime64 or a numeric day index.
        Must be the same length as anomaly_scores.
    rolling_window_days : int
        Window size (in array steps, not necessarily calendar days) for
        computing the rolling mean and standard deviation of the baseline.
    k_sigma : float
        Number of standard deviations above rolling mean to trigger maintenance.
        Default 2.0 gives ~2.3% false alarm rate under normality.
    min_periods : int
        Minimum number of observations required for the rolling statistics.
        Before this many observations are available, no trigger is raised.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, score, rolling_mean, rolling_std,
                 threshold, maintenance_triggered.
        One row per input observation.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> scores = rng.normal(0.1, 0.03, size=200)
    >>> scores[180:] += 0.15   # simulate degradation spike
    >>> timestamps = np.arange(200)
    >>> df = cbm_trigger_rule(scores, timestamps)
    >>> df["maintenance_triggered"].sum()  # should catch the degradation period
    """
    df = pd.DataFrame({"timestamp": timestamps, "score": anomaly_scores})
    roll = df["score"].rolling(window=rolling_window_days, min_periods=min_periods)
    df["rolling_mean"] = roll.mean()
    df["rolling_std"]  = roll.std().fillna(0.0)
    df["threshold"]    = df["rolling_mean"] + k_sigma * df["rolling_std"]
    df["maintenance_triggered"] = df["score"] > df["threshold"]
    return df


def trigger_threshold_sensitivity(
    anomaly_scores: np.ndarray,
    timestamps: np.ndarray,
    k_values: np.ndarray | None = None,
    rolling_window_days: int = 30,
) -> pd.DataFrame:
    """Sweep k_sigma values and report trigger statistics.

    Used to understand the trade-off between false alarm rate (high k → few
    triggers on healthy data) and detection sensitivity (low k → more
    triggers, catches degradation earlier but with more false alarms).

    Parameters
    ----------
    anomaly_scores, timestamps : see cbm_trigger_rule().
    k_values : np.ndarray, optional
        k_sigma values to evaluate. Defaults to np.linspace(1.0, 4.0, 31).

    Returns
    -------
    pd.DataFrame
        Columns: k_sigma, n_triggers, trigger_rate_pct, mean_score_at_trigger.
    """
    if k_values is None:
        k_values = np.linspace(1.0, 4.0, 31)

    rows = []
    for k in k_values:
        df = cbm_trigger_rule(anomaly_scores, timestamps, rolling_window_days, k)
        triggered = df[df["maintenance_triggered"]]
        rows.append({
            "k_sigma":               k,
            "n_triggers":            int(triggered.shape[0]),
            "trigger_rate_pct":      100.0 * triggered.shape[0] / max(len(df), 1),
            "mean_score_at_trigger": float(triggered["score"].mean()) if not triggered.empty else float("nan"),
        })
    return pd.DataFrame(rows)


def expected_cbm_cost_rate(
    lambda_base: float,
    Cp: float,
    Cf: float,
    gamma: float = 1.0,
    S_threshold: float = 0.3,
    S_dist_mean: float = 0.1,
    S_dist_std: float = 0.05,
    n_sim: int = 50_000,
    seed: int = 42,
) -> dict[str, float]:
    """Monte Carlo estimate of the long-run cost rate under the CBM policy.

    Models a single renewal cycle:
      1. Component starts healthy (score S ~ Normal(S_mean, S_std)).
      2. Effective hazard: h = λ_base × exp(γ × S).
      3. Either the trigger fires (Cp incurred) or failure occurs (Cf incurred).
      4. Cycle length = min(T_trigger, T_failure).

    Parameters
    ----------
    lambda_base : float
        Baseline hazard at S=0 (= 1/MTBF, /day).
    Cp, Cf : float
        Preventive and corrective replacement costs.
    gamma : float
        Proportional hazards coefficient. γ > 0 means higher score → higher hazard.
        Starting value 1.0 (to be calibrated with operational data).
    S_threshold : float
        Anomaly score value that triggers maintenance.
    S_dist_mean, S_dist_std : float
        Distribution of anomaly score at the point when a maintenance decision
        is made. Used to sample the effective hazard under the trigger.
    n_sim : int
        Monte Carlo replications.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        mean_cost_rate ($/day), std_cost_rate, fraction_preventive,
        mean_cycle_length_days.
    """
    rng = np.random.default_rng(seed)
    cycle_costs   = np.empty(n_sim)
    cycle_lengths = np.empty(n_sim)
    preventive    = np.empty(n_sim, dtype=bool)

    for i in range(n_sim):
        # Sample the anomaly score at which maintenance would be triggered
        S_val = float(rng.normal(S_dist_mean, S_dist_std))
        S_val = max(0.0, min(1.0, S_val))

        # Effective hazard at this score
        lam_eff = lambda_base * np.exp(gamma * S_val)

        # Time to failure (at effective hazard)
        T_fail = float(rng.exponential(1.0 / lam_eff))

        # Time until trigger (exponential at the baseline hazard × some factor)
        # Simple model: trigger fires at a Poisson-distributed time with rate λ_base/2
        # (representing the monitoring system sampling rate)
        T_trigger = float(rng.exponential(1.0 / (lambda_base * 0.5)))

        if T_trigger < T_fail:
            cycle_costs[i]   = Cp
            cycle_lengths[i] = T_trigger
            preventive[i]    = True
        else:
            cycle_costs[i]   = Cf
            cycle_lengths[i] = T_fail
            preventive[i]    = False

    # Guard against zero-length cycles
    valid = cycle_lengths > 0.0
    cost_rates = np.where(valid, cycle_costs / cycle_lengths, float("nan"))
    valid_rates = cost_rates[~np.isnan(cost_rates)]

    return {
        "mean_cost_rate":      float(np.mean(valid_rates)),
        "std_cost_rate":       float(np.std(valid_rates)),
        "fraction_preventive": float(np.mean(preventive)),
        "mean_cycle_length_days": float(np.mean(cycle_lengths)),
    }