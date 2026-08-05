"""
aeolus_rams_phase8.cox_survival
================================
Cox Proportional Hazards survival analysis — Pitch System.

Closes Phase 7 open item: γ in h(t|S) = h₀(t)·exp(γ·S(t)) was a stub.
The fitted vibration hazard ratio from this module provides γ_estimated.

SYNTHETIC DATA NOTICE: all input data is synthetically generated from
the Phase 2 Weibull posterior.  The fitted hazard ratios reflect the
data-generating true HRs (TRUE_HR_WIND=1.15, TRUE_HR_VIB=1.25,
TRUE_HR_TEMP=0.95) plus sampling noise.

Phase 7 linkage
---------------
Phase 7 CBM trigger defined:
    h(t | S(t)) = h₀(t) × exp(γ × S(t))
where S(t) = normalised anomaly score ∈ [0,1].

The Cox PH fit provides:
    exp(β_vibration) = HR for 1 SD increase in standardised vibration

Since vibration is the primary SCADA-observable proxy for the anomaly
score, the CBM coefficient is:
    γ_estimated = β_vibration = log(HR_vibration)

This is explicitly reported in the report (Section 4.4) with a numbered
equation and explanation.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test
    _LIFELINES_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LIFELINES_AVAILABLE = False


def _check_lifelines() -> None:
    if not _LIFELINES_AVAILABLE:
        raise ImportError(
            "lifelines is required for Cox PH analysis. "
            "Install with: pip install lifelines"
        )


def _standardise(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    Standardise columns to zero mean, unit variance.

    Returns the transformed DataFrame and a dict of {col: (mean, std)}.
    """
    df = df.copy()
    stats: dict[str, tuple[float, float]] = {}
    for col in cols:
        mu  = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mu) / std
        stats[col] = (mu, std)
    return df, stats


def fit_cox(covariates_df: pd.DataFrame) -> dict:
    """
    Fit Cox PH model to the Pitch System covariate dataset.

    Parameters
    ----------
    covariates_df : output of synthetic_data.generate_covariates()

    Returns
    -------
    dict with keys:
        cph            : fitted CoxPHFitter object
        hazard_ratios  : DataFrame (cox_hazard_ratios.csv schema)
        gamma_estimated : float  — log(HR_vibration) → Phase 7 γ
        concordance    : float   — Harrell's C-index
        covariate_stats : dict of (mean, std) per covariate
        df_std         : standardised DataFrame used for fitting
    """
    _check_lifelines()

    # Standardise covariates
    df_std, cov_stats = _standardise(covariates_df, config.COVARIATE_NAMES)

    # Fit
    cph = CoxPHFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(
            df_std[config.COVARIATE_NAMES + ["duration_days", "event_observed"]],
            duration_col="duration_days",
            event_col="event_observed",
        )

    summary = cph.summary.copy()

    # Build hazard ratios DataFrame
    hr_rows = []
    for cov in config.COVARIATE_NAMES:
        row = summary.loc[cov]
        hr_rows.append({
            "covariate":          cov,
            "exp_coef":           float(row["exp(coef)"]),
            "exp_coef_lower_95":  float(row["exp(coef) lower 95%"]),
            "exp_coef_upper_95":  float(row["exp(coef) upper 95%"]),
            "coef":               float(row["coef"]),
            "z_score":            float(row["z"]),
            "p_value":            float(row["p"]),
            "significant_at_0.05": bool(row["p"] < 0.05),
        })
    hazard_ratios_df = pd.DataFrame(hr_rows)

    # γ_estimated: coefficient for vibration (on standardised scale)
    vib_row = hazard_ratios_df[hazard_ratios_df["covariate"] == "vibration_rms_ms2"].iloc[0]
    gamma_estimated = float(vib_row["coef"])   # log(HR_vibration) = β_vibration

    concordance = float(cph.concordance_index_)

    return {
        "cph":              cph,
        "hazard_ratios":    hazard_ratios_df,
        "gamma_estimated":  gamma_estimated,
        "concordance":      concordance,
        "covariate_stats":  cov_stats,
        "df_std":           df_std,
    }


def compute_survival_curves(
    cph_result: dict,
    covariates_df: pd.DataFrame,
) -> dict:
    """
    Compute Kaplan-Meier curves stratified by wind speed quartile,
    plus Cox-predicted survival for average vs high-stress turbine.
    """
    _check_lifelines()

    df = covariates_df.copy()
    wind_q1 = df["wind_speed_mean_mps"].quantile(0.25)
    wind_q3 = df["wind_speed_mean_mps"].quantile(0.75)

    df_low  = df[df["wind_speed_mean_mps"] <= wind_q1].copy()
    df_high = df[df["wind_speed_mean_mps"] >= wind_q3].copy()

    km_low  = KaplanMeierFitter()
    km_high = KaplanMeierFitter()

    km_low.fit(df_low["duration_days"],   event_observed=df_low["event_observed"],  label="Low wind (Q1)")
    km_high.fit(df_high["duration_days"], event_observed=df_high["event_observed"], label="High wind (Q4)")

    # Log-rank test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr_result = logrank_test(
            df_low["duration_days"],
            df_high["duration_days"],
            event_observed_A=df_low["event_observed"],
            event_observed_B=df_high["event_observed"],
        )

    cph = cph_result["cph"]
    cov_stats = cph_result["covariate_stats"]

    def _make_profile(wind_offset_sd: float = 0.0, vib_offset_sd: float = 0.0) -> pd.DataFrame:
        """Build a single-row covariate profile (standardised)."""
        row = {}
        for cov in config.COVARIATE_NAMES:
            row[cov] = 0.0   # average profile
        row["wind_speed_mean_mps"]  += wind_offset_sd
        row["vibration_rms_ms2"]    += vib_offset_sd
        return pd.DataFrame([row])

    profile_avg    = _make_profile(0, 0)
    profile_stress = _make_profile(2, 2)

    sf_avg    = cph.predict_survival_function(profile_avg)
    sf_stress = cph.predict_survival_function(profile_stress)

    return {
        "km_low":       km_low,
        "km_high":      km_high,
        "lr_p_value":   float(lr_result.p_value),
        "sf_avg":       sf_avg,
        "sf_stress":    sf_stress,
        "wind_q1":      wind_q1,
        "wind_q3":      wind_q3,
    }


def plot_survival(survival_result: dict, output_dir: Path) -> None:
    """Generate cox_survival_curves.png (2-panel)."""
    _check_lifelines()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Pitch System Survival Analysis — Cox PH & Kaplan-Meier\n"
        "[SYNTHETIC — generated from Phase 2 Weibull parameters + physical forward model]",
        fontsize=10,
    )

    # Left: Kaplan-Meier stratified by wind speed quartile
    ax = axes[0]
    km_lo = survival_result["km_low"]
    km_hi = survival_result["km_high"]
    km_lo.plot_survival_function(ax=ax, color="#3b82f6", ci_show=True)
    km_hi.plot_survival_function(ax=ax, color="#ef4444", ci_show=True)
    ax.set_xlabel("Time (days)", fontsize=9)
    ax.set_ylabel("Survival probability S(t)", fontsize=9)
    ax.set_title(
        f"KM Curves by Wind Speed Quartile\n"
        f"(Log-rank p = {survival_result['lr_p_value']:.4f})",
        fontsize=10,
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)

    # Right: Cox-predicted survival
    ax = axes[1]
    sf_avg    = survival_result["sf_avg"]
    sf_stress = survival_result["sf_stress"]
    ax.plot(sf_avg.index,    sf_avg.values,    label="Average turbine",            color="#3b82f6", linewidth=1.8)
    ax.plot(sf_stress.index, sf_stress.values, label="High-stress (+2σ wind+vib)", color="#ef4444", linewidth=1.8, linestyle="--")
    ax.set_xlabel("Time (days)", fontsize=9)
    ax.set_ylabel("Predicted survival probability S(t)", fontsize=9)
    ax.set_title("Cox PH Predicted Survival", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "cox_survival_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(cph_result: dict, output_dir: Path) -> None:
    """Generate cox_ph_diagnostics.png — Schoenfeld + martingale residuals."""
    _check_lifelines()

    cph = cph_result["cph"]
    df_std = cph_result["df_std"]
    
    # Isolate only the numerical columns used during fitting
    df_fit = df_std[config.COVARIATE_NAMES + ["duration_days", "event_observed"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Cox PH Assumption Diagnostics [SYNTHETIC DATA]", fontsize=10)

    # Schoenfeld residuals — one line per covariate
    ax = axes[0]
    try:
        sch_res = cph.compute_residuals(df_fit, kind="schoenfeld")
        for col in config.COVARIATE_NAMES:
            if col in sch_res.columns:
                ax.scatter(
                    sch_res.index, sch_res[col],
                    alpha=0.5, s=15, label=col.replace("_", " "),
                )
        ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Time (days)", fontsize=9)
        ax.set_ylabel("Schoenfeld residuals", fontsize=9)
        ax.set_title("PH Assumption Check (Schoenfeld)", fontsize=10)
        ax.legend(fontsize=7)
    except Exception as exc:  # pragma: no cover
        ax.text(0.5, 0.5, f"Residuals unavailable:\n{exc}", ha="center", va="center", transform=ax.transAxes)

    # Martingale residuals vs fitted linear predictor
    ax = axes[1]
    try:
        mart_res = cph.compute_residuals(df_fit, kind="martingale")
        lp = cph.predict_log_partial_hazard(df_fit)
        
        # lp is a Series of length N.
        x_vals = lp.values
        
        # Extract ONLY the martingale column to match the exact size of x_vals (220 elements)
        if "martingale" in mart_res.columns:
            y_vals = mart_res["martingale"].values
        else:
            y_vals = mart_res.iloc[:, -1].values  # Fallback for version differences
            
        ax.scatter(x_vals, y_vals, alpha=0.4, s=15, color="#6b7280")
        ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Log partial hazard (linear predictor)", fontsize=9)
        ax.set_ylabel("Martingale residuals", fontsize=9)
        ax.set_title("Functional Form Check (Martingale)", fontsize=10)
    except Exception as exc:  # pragma: no cover
        ax.text(0.5, 0.5, f"Residuals unavailable:\n{exc}", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    fig.savefig(output_dir / "cox_ph_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(
    covariates_df: pd.DataFrame,
    output_dir: Path,
    no_plots: bool = False,
) -> dict:
    """
    Full Cox PH pipeline.

    Returns
    -------
    dict with keys: 'cph_result', 'survival_result', 'gamma_estimated'
    """
    _check_lifelines()
    output_dir.mkdir(parents=True, exist_ok=True)

    cph_result      = fit_cox(covariates_df)
    survival_result = compute_survival_curves(cph_result, covariates_df)

    # Write hazard ratios CSV
    cph_result["hazard_ratios"].to_csv(output_dir / "cox_hazard_ratios.csv", index=False)

    if not no_plots:
        plot_survival(survival_result, output_dir)
        plot_diagnostics(cph_result, output_dir)

    return {
        "cph_result":      cph_result,
        "survival_result": survival_result,
        "gamma_estimated": cph_result["gamma_estimated"],
    }