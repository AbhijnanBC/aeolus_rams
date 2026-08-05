"""
aeolus_rams_phase8.bayesian_update
===================================
Grid-based Bayesian posterior for Weibull (β, η) — Main/Rotor Bearing.

Mathematical framework
----------------------
Prior (LogNormal on β and η, independent):
    log β ~ N(log β₀, σ_β²)
    log η ~ N(log η₀, σ_η²)

where β₀ = BEARING_BETA, η₀ = BEARING_ETA (Phase 2 / Phase 7 committed values).

Weibull likelihood with right-censoring:
    L(β, η | data) = ∏ f(tᵢ; β, η) × ∏ R(cⱼ; β, η)
                    failures i       censored j

    f(t; β, η) = (β/η)(t/η)^(β−1) exp(−(t/η)^β)   [Weibull PDF]
    R(t; β, η) = exp(−(t/η)^β)                       [Weibull reliability]

Log-posterior (numerically stable):
    log π = Σᵢ [log β − log η + (β−1)log(tᵢ/η) − (tᵢ/η)^β]
           + Σⱼ [−(cⱼ/η)^β]
           − (log β − log β₀)²/(2σ_β²)
           − (log η − log η₀)²/(2σ_η²)

Grid computation uses the log-sum-exp trick for numerical stability.

Maintenance interval update
---------------------------
Phase 7 derives the optimal replacement interval T* via the
Barlow-Proschan cost-rate minimisation:
    C(T) = [Cp × F(T) + Cf × R(T)] / ∫₀ᵀ R(t) dt

where F(T) = 1 − R(T), R(T) = exp(−(T/η)^β).

Phase 8 evaluates C(T) at the posterior MAP (β̂, η̂) to show how T*
shifts as new bearing failure data arrives.

SYNTHETIC DATA NOTICE: all failure data passed to this module is
synthetically generated from the Phase 2 Weibull posterior.
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
import matplotlib.cm as cm

from . import config


# ---------------------------------------------------------------------------
# Log-posterior computation
# ---------------------------------------------------------------------------

def _log_likelihood_grid(
    log_beta_grid: np.ndarray,   # (N_beta,)
    log_eta_grid: np.ndarray,    # (N_eta,)
    fail_times: np.ndarray,      # (n_failures,)
    censor_times: np.ndarray,    # (n_censored,)
) -> np.ndarray:
    """
    Compute log-likelihood on a 2D grid.

    Returns array of shape (N_beta, N_eta).
    Vectorised: no Python loops.
    """
    beta = np.exp(log_beta_grid)   # (N_beta,)
    eta  = np.exp(log_eta_grid)    # (N_eta,)

    # Broadcast: beta → (N_beta, 1), eta → (1, N_eta)
    beta2d = beta[:, None]
    eta2d  = eta[None, :]

    log_ll = np.zeros((len(beta), len(eta)))

    # Failure contribution: log f(t; β, η)
    for t in fail_times:
        t = max(t, 1e-3)
        log_ll += (
            np.log(beta2d)
            - np.log(eta2d)
            + (beta2d - 1.0) * (math.log(t) - np.log(eta2d))
            - (t / eta2d) ** beta2d
        )

    # Censored contribution: log R(c; β, η) = −(c/η)^β
    for c in censor_times:
        c = max(c, 1e-3)
        log_ll -= (c / eta2d) ** beta2d

    return log_ll


def _log_prior_grid(
    log_beta_grid: np.ndarray,
    log_eta_grid: np.ndarray,
    log_beta0: float,
    log_eta0: float,
) -> np.ndarray:
    """
    Log of the independent LogNormal prior on (β, η).

    Returns array of shape (N_beta, N_eta).
    """
    sigma_beta = config.PRIOR_BETA_LOG_SIGMA
    sigma_eta  = config.PRIOR_ETA_LOG_SIGMA

    log_prior_beta = -0.5 * ((log_beta_grid - log_beta0) / sigma_beta) ** 2
    log_prior_eta  = -0.5 * ((log_eta_grid  - log_eta0)  / sigma_eta)  ** 2

    # Broadcast to 2D
    return log_prior_beta[:, None] + log_prior_eta[None, :]


def _normalise_log_posterior(log_post: np.ndarray) -> np.ndarray:
    """
    Numerically stable normalisation via log-sum-exp.
    Returns a proper probability grid (sums to 1.0).
    """
    # Subtract maximum for numerical stability
    lp_shifted = log_post - log_post.max()
    post = np.exp(lp_shifted)
    post /= post.sum()
    return post


def compute_posterior_grid(
    fail_times: np.ndarray,
    censor_times: np.ndarray,
    beta0: float = config.BEARING_BETA,
    eta0: float  = config.BEARING_ETA,
    n_beta: int  = config.GRID_N_BETA,
    n_eta: int   = config.GRID_N_ETA,
) -> dict:
    """
    Compute the normalised 2D posterior grid over (log β, log η).

    Parameters
    ----------
    fail_times    : observed failure times (days)
    censor_times  : right-censored observation times (days)
    beta0, eta0   : prior centre (Phase 2 / Phase 7 values)
    n_beta, n_eta : grid resolution

    Returns
    -------
    dict with keys:
        log_beta_grid  (n_beta,)
        log_eta_grid   (n_eta,)
        posterior      (n_beta, n_eta)  — normalised, sums to 1
        log_posterior  (n_beta, n_eta)  — unnormalised log-posterior
        beta_map, eta_map               — MAP estimates
        marginal_beta  (n_beta,)
        marginal_eta   (n_eta,)
        ci90_beta      (lo, hi)
        ci90_eta       (lo, hi)
    """
    log_beta0 = math.log(beta0)
    log_eta0  = math.log(eta0)
    sigma_beta = config.PRIOR_BETA_LOG_SIGMA
    sigma_eta  = config.PRIOR_ETA_LOG_SIGMA

    # Grid bounds: ±3σ in log space
    lb_lo = log_beta0 - config.GRID_BETA_RANGE_SIGMA * sigma_beta
    lb_hi = log_beta0 + config.GRID_BETA_RANGE_SIGMA * sigma_beta
    le_lo = log_eta0  - config.GRID_ETA_RANGE_SIGMA  * sigma_eta
    le_hi = log_eta0  + config.GRID_ETA_RANGE_SIGMA  * sigma_eta

    log_beta_grid = np.linspace(lb_lo, lb_hi, n_beta)
    log_eta_grid  = np.linspace(le_lo, le_hi, n_eta)

    fail_arr   = np.asarray(fail_times,   dtype=float)
    censor_arr = np.asarray(censor_times, dtype=float)

    log_ll    = _log_likelihood_grid(log_beta_grid, log_eta_grid, fail_arr, censor_arr)
    log_prior = _log_prior_grid(log_beta_grid, log_eta_grid, log_beta0, log_eta0)

    log_post = log_ll + log_prior
    post     = _normalise_log_posterior(log_post)

    # MAP
    idx_map  = np.unravel_index(np.argmax(post), post.shape)
    beta_map = float(np.exp(log_beta_grid[idx_map[0]]))
    eta_map  = float(np.exp(log_eta_grid[idx_map[1]]))

    # Marginals
    marginal_beta = post.sum(axis=1)   # sum over η
    marginal_eta  = post.sum(axis=0)   # sum over β

    def _ci90(marginal: np.ndarray, grid: np.ndarray) -> tuple[float, float]:
        cdf = np.cumsum(marginal)
        cdf /= cdf[-1]
        lo = float(np.exp(grid[np.searchsorted(cdf, 0.05)]))
        hi = float(np.exp(grid[np.searchsorted(cdf, 0.95)]))
        return lo, hi

    ci90_beta = _ci90(marginal_beta, log_beta_grid)
    ci90_eta  = _ci90(marginal_eta,  log_eta_grid)

    return {
        "log_beta_grid": log_beta_grid,
        "log_eta_grid":  log_eta_grid,
        "posterior":     post,
        "log_posterior": log_post,
        "beta_map":      beta_map,
        "eta_map":       eta_map,
        "marginal_beta": marginal_beta,
        "marginal_eta":  marginal_eta,
        "ci90_beta":     ci90_beta,
        "ci90_eta":      ci90_eta,
    }


# ---------------------------------------------------------------------------
# Barlow-Proschan cost-rate minimisation (Phase 7 formula)
# ---------------------------------------------------------------------------

def _barlow_proschan_t_star(
    beta: float,
    eta: float,
    Cp: float = config.BEARING_Cp,
    Cf: float = config.BEARING_Cf,
    t_max_years: float = 80.0,
    n_points: int = 5_000,
) -> float:
    """
    Find T* that minimises the Barlow-Proschan cost rate:
        C(T) = [Cp × (1−R(T)) + Cf × R(T)] / ∫₀ᵀ R(t) dt

    Uses Simpson's rule for the integral; grid search for minimum.
    Returns T* in days.
    """
    t_grid = np.linspace(1.0, t_max_years * 365.25, n_points)

    def R(t: np.ndarray) -> np.ndarray:
        return np.exp(-((t / eta) ** beta))

    r_vals = R(t_grid)

    # Trapezoidal integral of R from 0 to T for each T
    integrals = np.cumsum(
        0.5 * (r_vals[:-1] + r_vals[1:]) * np.diff(t_grid)
    )
    integrals = np.concatenate([[t_grid[0]], integrals])  # pad left

    cost_rate = (Cp * r_vals + Cf * (1.0 - r_vals)) / np.maximum(integrals, 1e-6)

    idx_min = int(np.argmin(cost_rate))
    return float(t_grid[idx_min])


# ---------------------------------------------------------------------------
# Sequential update
# ---------------------------------------------------------------------------

def run_sequential_update(
    bearing_df: pd.DataFrame,
) -> tuple[list[dict], pd.DataFrame]:
    """
    Run four-stage sequential Bayesian update:
        Stage 0 — prior only
        Stage 1 — prior + batch 1
        Stage 2 — prior + batches 1–2
        Stage 3 — prior + all batches + censored

    Returns
    -------
    posteriors : list of 4 posterior dicts (from compute_posterior_grid)
    summary_df : DataFrame with one row per stage (bayesian_update_summary.csv)
    """
    # Separate failures by batch; censored observations
    def _get_data(stages: list[int]) -> tuple[np.ndarray, np.ndarray]:
        fail_mask = bearing_df["event_observed"] == 1
        fail_df   = bearing_df[fail_mask]
        cens_df   = bearing_df[bearing_df["event_observed"] == 0]

        if not stages:
            return np.array([]), np.array([])

        selected_failures = fail_df[fail_df["batch"].isin(stages)]["event_time_days"].values
        # Include censored only in final stage (stage 3, which uses all batches)
        selected_censored = (
            cens_df["event_time_days"].values if max(stages) == 3 else np.array([])
        )
        return selected_failures, selected_censored

    stage_configs = [
        ([], "Stage 0: prior only"),
        ([1], "Stage 1: + batch 1 (5 failures)"),
        ([1, 2], "Stage 2: + batch 2 (10 failures)"),
        ([1, 2, 3], "Stage 3: + batch 3 + censored"),
    ]

    posteriors: list[dict] = []
    rows: list[dict] = []

    for stage_idx, (batches, label) in enumerate(stage_configs):
        fail_arr, cens_arr = _get_data(batches)

        result = compute_posterior_grid(
            fail_times   = fail_arr,
            censor_times = cens_arr,
        )
        result["stage_label"] = label
        posteriors.append(result)

        # Barlow-Proschan T* at MAP
        t_star = _barlow_proschan_t_star(result["beta_map"], result["eta_map"])
        savings = 1.0 - (result["beta_map"] / config.BEARING_BETA)  # heuristic; see note

        # Proper savings: C(T*_MAP) vs C(∞) [run-to-failure] at MAP params
        r_inf = 0.0  # R(∞) = 0 → cost = Cf per cycle of length MTTF
        mttf_map = result["eta_map"] * math.gamma(1.0 + 1.0 / result["beta_map"])
        cost_rtf = config.BEARING_Cf / mttf_map
        # cost at T*
        beta_m, eta_m = result["beta_map"], result["eta_map"]
        r_tstar = math.exp(-((t_star / eta_m) ** beta_m))
        integral_tstar = _barlow_proschan_integral(beta_m, eta_m, t_star)
        cost_tstar = (
            (config.BEARING_Cp * r_tstar + config.BEARING_Cf * (1.0 - r_tstar))
            / max(integral_tstar, 1e-6)
        )
        savings_proper = 1.0 - cost_tstar / cost_rtf

        rows.append({
            "update_stage":       stage_idx,
            "stage_label":        label,
            "n_failures":         len(fail_arr),
            "n_censored":         len(cens_arr),
            "beta_map":           round(result["beta_map"], 4),
            "eta_map_days":       round(result["eta_map"], 1),
            "beta_ci90_lo":       round(result["ci90_beta"][0], 4),
            "beta_ci90_hi":       round(result["ci90_beta"][1], 4),
            "eta_ci90_lo_days":   round(result["ci90_eta"][0], 1),
            "eta_ci90_hi_days":   round(result["ci90_eta"][1], 1),
            "T_star_updated_days": round(t_star, 1),
            "T_star_updated_years": round(t_star / 365.25, 2),
            "savings_updated_fraction": round(max(savings_proper, 0.0), 4),
        })

    summary_df = pd.DataFrame(rows)
    return posteriors, summary_df


def _barlow_proschan_integral(beta: float, eta: float, T: float, n: int = 2_000) -> float:
    """∫₀ᵀ R(t; β, η) dt via trapezoid."""
    t = np.linspace(0.0, T, n)
    R = np.exp(-((t / eta) ** beta))
    return float(np.trapezoid(R, t))  


# ---------------------------------------------------------------------------
# Posterior grid DataFrame export
# ---------------------------------------------------------------------------

def posterior_to_dataframe(result: dict) -> pd.DataFrame:
    """
    Flatten the 2D posterior grid to a long-form DataFrame for CSV export.

    Columns: beta, eta, log_prior, log_posterior, posterior_normalised
    """
    n_beta = len(result["log_beta_grid"])
    n_eta  = len(result["log_eta_grid"])

    beta_vals = np.exp(result["log_beta_grid"])
    eta_vals  = np.exp(result["log_eta_grid"])

    rows = []
    for i in range(n_beta):
        for j in range(n_eta):
            rows.append({
                "beta":                 beta_vals[i],
                "eta":                  eta_vals[j],
                "log_posterior":        result["log_posterior"][i, j],
                "posterior_normalised": result["posterior"][i, j],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _cost_rate_curve(
    beta: float,
    eta: float,
    label: str,
    color: str,
    ax: plt.Axes,
    t_max_yr: float = 50.0,
    n_points: int = 500,
) -> None:
    """Plot C(T) vs T (years) on ax."""
    t_grid = np.linspace(30.0, t_max_yr * 365.25, n_points)
    R = np.exp(-((t_grid / eta) ** beta))
    integrals = np.array([
        _barlow_proschan_integral(beta, eta, T, n=500) for T in t_grid
    ])
    cost_rate = (config.BEARING_Cp * R + config.BEARING_Cf * (1 - R)) / np.maximum(integrals, 1.0)
    t_yr = t_grid / 365.25
    ax.plot(t_yr, cost_rate / 1_000, label=label, color=color, linewidth=1.8)
    t_star = _barlow_proschan_t_star(beta, eta)
    r_ts = math.exp(-((t_star / eta) ** beta))
    cr_ts = (
        (config.BEARING_Cp * r_ts + config.BEARING_Cf * (1 - r_ts))
        / _barlow_proschan_integral(beta, eta, t_star, n=500)
    )
    ax.axvline(x=t_star / 365.25, color=color, linestyle="--", alpha=0.6, linewidth=1.0)
    ax.scatter([t_star / 365.25], [cr_ts / 1_000], color=color, s=60, zorder=5)


def plot_bayesian_update(
    posteriors: list[dict],
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Generate bayesian_update_plot.png (2×2 figure) and
    sequential_update_plot.png (4-stage contour overlay).
    """
    colours = ["#6b7280", "#3b82f6", "#f59e0b", "#10b981"]   # gray, blue, amber, green

    # ---- Figure 1: 2×2 diagnostic ----------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        "Bayesian Weibull Updating — Main/Rotor Bearing\n"
        "[SYNTHETIC — generated from Phase 2 Weibull parameters + physical forward model]",
        fontsize=11, y=0.98,
    )

    # Top-left: 2D posterior contour (prior vs. final)
    ax = axes[0, 0]
    for stage_idx, label, alpha in [(0, "Prior (Stage 0)", 0.4), (3, "Posterior (Stage 3)", 1.0)]:
        result = posteriors[stage_idx]
        beta_grid = np.exp(result["log_beta_grid"])
        eta_grid  = np.exp(result["log_eta_grid"]) / 1_000  # display in kdays
        post_2d   = result["posterior"]
        levels    = np.percentile(post_2d[post_2d > 0], [10, 40, 70, 90, 99])
        ax.contour(
            eta_grid, beta_grid, post_2d,
            levels=levels,
            alpha=alpha,
            colors=[colours[stage_idx]],
        )
    ax.scatter(
        [posteriors[3]["eta_map"] / 1_000],
        [posteriors[3]["beta_map"]],
        color="#10b981", s=80, zorder=10, label="MAP (Stage 3)",
    )
    ax.scatter(
        [config.BEARING_ETA / 1_000],
        [config.BEARING_BETA],
        color="#6b7280", s=60, marker="x", zorder=10, label="Phase 2 prior",
    )
    ax.set_xlabel("η (× 10³ days)", fontsize=9)
    ax.set_ylabel("β (shape)", fontsize=9)
    ax.set_title("2D Posterior: Prior vs Stage 3", fontsize=10)
    ax.legend(fontsize=8)

    # Top-right: Marginal P(β)
    ax = axes[0, 1]
    for stage_idx in range(4):
        result = posteriors[stage_idx]
        beta_grid = np.exp(result["log_beta_grid"])
        marginal  = result["marginal_beta"]
        marginal  = marginal / np.trapezoid(marginal, beta_grid) 
        ax.plot(
            beta_grid, marginal,
            color=colours[stage_idx],
            linewidth=1.8,
            linestyle="--" if stage_idx == 0 else "-",
            label=f"Stage {stage_idx}",
            alpha=0.9,
        )
    ax.axvline(x=config.BEARING_BETA, color="k", linestyle=":", linewidth=1.0, label="β₀ (Phase 2)")
    ax.set_xlabel("β (shape parameter)", fontsize=9)
    ax.set_ylabel("Marginal density", fontsize=9)
    ax.set_title("Marginal Posterior P(β)", fontsize=10)
    ax.legend(fontsize=8)

    # Bottom-left: Marginal P(η)
    ax = axes[1, 0]
    for stage_idx in range(4):
        result = posteriors[stage_idx]
        eta_grid  = np.exp(result["log_eta_grid"])
        marginal  = result["marginal_eta"]
        marginal  = marginal / np.trapezoid(marginal, eta_grid) 
        ax.plot(
            eta_grid / 1_000, marginal * 1_000,
            color=colours[stage_idx],
            linewidth=1.8,
            linestyle="--" if stage_idx == 0 else "-",
            label=f"Stage {stage_idx}",
            alpha=0.9,
        )
    ax.axvline(x=config.BEARING_ETA / 1_000, color="k", linestyle=":", linewidth=1.0, label="η₀ (Phase 2)")
    ax.set_xlabel("η (× 10³ days)", fontsize=9)
    ax.set_ylabel("Marginal density (×10⁻³)", fontsize=9)
    ax.set_title("Marginal Posterior P(η)", fontsize=10)
    ax.legend(fontsize=8)

    # Bottom-right: Updated C(T) curve
    ax = axes[1, 1]
    _cost_rate_curve(
        config.BEARING_BETA, config.BEARING_ETA,
        label="Phase 2 prior (β=2.0, η=25000d)", color="#6b7280", ax=ax,
    )
    row3 = summary_df[summary_df["update_stage"] == 3].iloc[0]
    _cost_rate_curve(
        row3["beta_map"], row3["eta_map_days"],
        label=f"Stage 3 MAP (β={row3['beta_map']:.3f}, η={row3['eta_map_days']:.0f}d)",
        color="#10b981", ax=ax,
    )
    ax.axvline(
        x=config.BEARING_T_STAR_PHASE7 / 365.25,
        color="#6b7280", linestyle=":", linewidth=1.0,
        label=f"Phase 7 T*={config.BEARING_T_STAR_PHASE7/365.25:.0f} yr",
    )
    ax.set_xlabel("Replacement interval T (years)", fontsize=9)
    ax.set_ylabel("Cost rate C(T) (USD/day, ×10³)", fontsize=9)
    ax.set_title("Barlow-Proschan Cost Rate: Phase 2 vs Stage 3", fontsize=10)
    ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(output_dir / "bayesian_update_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: Sequential update overlay --------------------------------
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    fig2.suptitle(
        "Sequential Bayesian Update — 4 Posterior Stages\n"
        "[SYNTHETIC — generated from Phase 2 Weibull parameters + physical forward model]",
        fontsize=10,
    )
    alphas = [1.0, 0.75, 0.55, 0.35]
    for stage_idx in range(4):
        result    = posteriors[stage_idx]
        beta_grid = np.exp(result["log_beta_grid"])
        eta_grid  = np.exp(result["log_eta_grid"]) / 1_000
        post_2d   = result["posterior"]
        # Use a single level at 50% of maximum for each stage
        levels = [0.5 * post_2d.max()]
        cs = ax2.contour(
            eta_grid, beta_grid, post_2d,
            levels=levels,
            colors=[colours[stage_idx]],
            alpha=alphas[3 - stage_idx],
            linewidths=2.0,
        )
        ax2.scatter(
            [result["eta_map"] / 1_000],
            [result["beta_map"]],
            color=colours[stage_idx],
            s=50, zorder=10,
            label=f"Stage {stage_idx}: β̂={result['beta_map']:.3f}, η̂={result['eta_map']:.0f}d",
        )
    ax2.set_xlabel("η (× 10³ days)", fontsize=10)
    ax2.set_ylabel("β (shape parameter)", fontsize=10)
    ax2.set_title("Progressive Posterior Narrowing (Stages 0→3)", fontsize=11)
    ax2.legend(fontsize=9, loc="upper right")
    fig2.savefig(output_dir / "sequential_update_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    bearing_df: pd.DataFrame,
    output_dir: Path,
    no_plots: bool = False,
) -> dict:
    """
    Full Bayesian update pipeline.

    Parameters
    ----------
    bearing_df  : output of synthetic_data.generate_bearing_failures()
    output_dir  : where to write CSV and PNG files

    Returns
    -------
    dict with keys: 'posteriors', 'summary_df', 'posterior_df' (final stage)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    posteriors, summary_df = run_sequential_update(bearing_df)

    # Write summary CSV
    summary_df.to_csv(output_dir / "bayesian_update_summary.csv", index=False)

    # Write posterior grid CSV (final stage)
    posterior_df = posterior_to_dataframe(posteriors[3])
    posterior_df.to_csv(output_dir / "posterior_grid.csv", index=False)

    if not no_plots:
        plot_bayesian_update(posteriors, summary_df, output_dir)

    return {
        "posteriors":   posteriors,
        "summary_df":   summary_df,
        "posterior_df": posterior_df,
    }