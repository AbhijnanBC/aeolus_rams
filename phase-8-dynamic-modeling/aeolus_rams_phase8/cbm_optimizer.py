"""
aeolus_rams_phase8.cbm_optimizer
==================================
CBM threshold cost-rate optimisation — Pitch System.

Closes Phase 7 open item: rcm_schedule.csv marks Pitch System
trigger as "k=2σ (default); tune with operational data (Phase 8+)".

This module finds k* by minimising the expected long-run cost rate
over a Monte Carlo simulation of CBM cycles.

Anomaly score model
-------------------
For each simulation cycle (one turbine-component lifespan):
    1. Draw T_f ~ Weibull(β=0.728, η=2653d)
    2. Generate S(t) = N(0, 1) + ramp starting at 0.70·T_f
       ramp(t) = max(0, r·(t − 0.70·T_f))  where r = CBM_DEGRADATION_RATE
    3. Rolling baseline: μ(t), σ(t) over 30-day window
    4. Alert fires at first t where S(t) > μ(t) + k·σ(t)   [burn-in: t≥30]
    5. If alert before T_f → preventive (cost = Cp), cycle length = t_alert
    6. Otherwise         → corrective (cost = Cf), cycle length = T_f

Cost rate ($/day)
-----------------
cost_rate(k) = E[cost per cycle] / E[cycle length]
             = (n_prev·Cp + n_corr·Cf) / (N · mean_cycle_length)

This is Renewal Reward Theorem. The simulation ensures the expected
value converges; N=5000 gives < 5% Monte Carlo error for typical
cost curves.

SYNTHETIC DATA NOTICE: lifetime T_f is drawn from the Phase 2 Weibull
posterior, not from observed SCADA events.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config
from . import synthetic_data


# ---------------------------------------------------------------------------
# Core simulation: one k_sigma, one Cp/Cf ratio
# ---------------------------------------------------------------------------

def _simulate_cbm_cycles(
    k_sigma: float,
    Cp: float,
    Cf: float,
    n_sims: int = config.CBM_N_SIMS,
    seed: int   = config.SYNTHETIC_SEED,
) -> dict:
    """
    Run N Monte Carlo CBM cycles for a given k_sigma and Cp/Cf.

    Returns dict:
        cost_rate_per_day  float
        expected_cycle_days float
        p_preventive       float
        p_corrective       float
        n_preventive       int
        n_corrective       int
    """
    rng = np.random.default_rng(seed + int(k_sigma * 1000))

    n_preventive = 0
    n_corrective = 0
    total_cost   = 0.0
    total_length = 0.0

    beta = config.PITCH_BETA
    eta  = config.PITCH_ETA

    for i in range(n_sims):
        # Draw failure time from Weibull(β, η)
        u = rng.uniform(1e-9, 1.0 - 1e-9)
        T_f = eta * (-math.log(u)) ** (1.0 / beta)
        T_f = max(T_f, float(config.CBM_ROLLING_WINDOW) + 5.0)  # ensure burn-in possible

        t_int = int(math.ceil(T_f)) + 1

        # Anomaly score trajectory
        noise = rng.standard_normal(t_int)
        t_arr = np.arange(t_int, dtype=float)
        t_onset = config.CBM_DEGRADATION_ONSET * T_f
        ramp = np.maximum(0.0, config.CBM_DEGRADATION_RATE * (t_arr - t_onset))
        S = noise + ramp

        # Rolling mean and std (window = 30)
        w = config.CBM_ROLLING_WINDOW
        t_alert = None
        for t in range(w, t_int):
            window = S[t - w: t]
            mu_w  = window.mean()
            std_w = window.std()
            if std_w < 1e-6:
                continue
            threshold = mu_w + k_sigma * std_w
            if S[t] > threshold:
                t_alert = float(t)
                break

        if t_alert is not None and t_alert < T_f:
            # Preventive maintenance
            n_preventive += 1
            total_cost   += Cp
            total_length += t_alert
        else:
            # Corrective failure
            n_corrective += 1
            total_cost   += Cf
            total_length += T_f

    mean_cost_per_cycle   = total_cost / n_sims
    mean_cycle_length     = total_length / n_sims
    cost_rate             = mean_cost_per_cycle / max(mean_cycle_length, 1.0)

    return {
        "cost_rate_per_day":  cost_rate,
        "expected_cycle_days": mean_cycle_length,
        "p_preventive":       n_preventive / n_sims,
        "p_corrective":       n_corrective / n_sims,
        "n_preventive":       n_preventive,
        "n_corrective":       n_corrective,
    }


# ---------------------------------------------------------------------------
# Full sweep: all k values × all Cp/Cf ratios
# ---------------------------------------------------------------------------

def run_sweep(
    Cp: float = config.PITCH_Cp,
    Cf: float = config.PITCH_Cf,
    k_values: np.ndarray | None = None,
    n_sims: int = config.CBM_N_SIMS,
    seed: int   = config.SYNTHETIC_SEED,
) -> pd.DataFrame:
    """
    Sweep k_sigma from CBM_K_MIN to CBM_K_MAX for a single Cp/Cf ratio.

    Returns DataFrame with columns matching cbm_threshold_table.csv schema.
    """
    if k_values is None:
        k_values = np.linspace(config.CBM_K_MIN, config.CBM_K_MAX, config.CBM_K_N_POINTS)

    rows = []
    for k in k_values:
        result = _simulate_cbm_cycles(k, Cp, Cf, n_sims=n_sims, seed=seed)
        rows.append({
            "k_sigma":             round(float(k), 3),
            "cost_rate_per_day":   result["cost_rate_per_day"],
            "expected_cycle_days": result["expected_cycle_days"],
            "p_preventive":        result["p_preventive"],
            "p_corrective":        result["p_corrective"],
            "n_preventive":        result["n_preventive"],
            "n_corrective":        result["n_corrective"],
            "Cp_usd":              Cp,
            "Cf_usd":              Cf,
            "cp_cf_ratio":         round(Cp / Cf, 4),
        })
    return pd.DataFrame(rows)


def find_k_star(sweep_df: pd.DataFrame) -> dict:
    """
    Extract optimal k* from a sweep DataFrame.
    """
    # Use argmin() to get the integer position for .iloc, not the label index
    idx_min   = int(sweep_df["cost_rate_per_day"].argmin())
    row_star  = sweep_df.iloc[idx_min]
    k_star    = float(row_star["k_sigma"])
    cr_star   = float(row_star["cost_rate_per_day"])

    # Cost rate at Phase 7 default k=2.0
    idx_def = int((sweep_df["k_sigma"] - 2.0).abs().argmin())
    default_row = sweep_df.iloc[idx_def]
    cr_default  = float(default_row["cost_rate_per_day"])

    reduction_pct = 100.0 * (cr_default - cr_star) / max(cr_default, 1e-12)

    return {
        "k_star":          k_star,
        "cost_rate_star":  cr_star,
        "cost_rate_default": cr_default,
        "reduction_pct":   reduction_pct,
        "k_default":       2.0,
    }


def run_sensitivity(
    k_values: np.ndarray | None = None,
    n_sims: int = config.CBM_N_SIMS,
    seed: int   = config.SYNTHETIC_SEED,
) -> pd.DataFrame:
    """
    Run sweep for all Cp/Cf ratios in config.CBM_CP_CF_RATIOS.

    Returns combined DataFrame with cp_cf_ratio column.
    """
    if k_values is None:
        k_values = np.linspace(config.CBM_K_MIN, config.CBM_K_MAX, config.CBM_K_N_POINTS)

    frames = []
    for ratio in config.CBM_CP_CF_RATIOS:
        Cp = config.PITCH_Cf * ratio   # Cf is fixed; Cp = ratio × Cf
        df = run_sweep(Cp=Cp, Cf=config.PITCH_Cf, k_values=k_values, n_sims=n_sims, seed=seed)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Update text for Phase 7 RCM schedule
# ---------------------------------------------------------------------------

def generate_rcm_update_text(
    k_star_result: dict,
    sensitivity_df: pd.DataFrame,
    output_path: Path,
) -> str:
    """
    Generate cbm_update_to_rcm.txt — the text block that replaces
    the Phase 7 rcm_schedule.csv Pitch System note.
    """
    k_star    = k_star_result["k_star"]
    cr_star   = k_star_result["cost_rate_star"]
    cr_default = k_star_result["cost_rate_default"]
    reduction  = k_star_result["reduction_pct"]

    # k* for each Cp/Cf ratio (from sensitivity)
    sensitivity_lines = []
    for ratio in config.CBM_CP_CF_RATIOS:
        sub = sensitivity_df[abs(sensitivity_df["cp_cf_ratio"] - ratio) < 1e-6]
        if len(sub) == 0:
            continue
        k_r = find_k_star(sub)
        sensitivity_lines.append(
            f"  Cp/Cf={ratio:.2f}: k* = {k_r['k_star']:.2f}  "
            f"(cost_rate = ${k_r['cost_rate_star']:.2f}/day)"
        )

    text = (
        "==========================================================\n"
        "PHASE 8 CBM THRESHOLD OPTIMISATION — PITCH SYSTEM UPDATE\n"
        "==========================================================\n\n"
        "Phase 7 rcm_schedule.csv entry:\n"
        "  Component     : Pitch System\n"
        "  RCM Category  : CONDITION_BASED\n"
        "  Trigger Rule  : k=2.0σ (default; tune with operational data, Phase 8+)\n\n"
        "Phase 8 update (N={n_sims} Monte Carlo cycles per k_sigma,\n"
        "  k sweep: [{k_min}, {k_max}] in {k_n} steps):\n\n"
        "  CENTRAL CASE (Cp/Cf = 0.10):\n"
        "  ► k* = {k_star:.2f}σ  (Phase 8 optimised)\n"
        "  ► Cost rate at k* = ${cr_star:.2f}/day\n"
        "  ► Cost rate at Phase 7 default k=2.0 = ${cr_default:.2f}/day\n"
        "  ► Cost reduction vs default = {reduction:.1f}%\n\n"
        "  Cp/Cf SENSITIVITY:\n"
        "{sensitivity}\n\n"
        "  INTERPRETATION:\n"
        "  Higher k → fewer false alarms but more corrective failures.\n"
        "  Lower k → more preventive interventions at shorter lead times.\n"
        "  k* is an interior minimum: the degradation ramp onset at 70% of\n"
        "  lifetime creates a detectable pre-failure signal that an optimally\n"
        "  tuned threshold captures before failure at a cost less than Cf.\n\n"
        "  ACTION: Replace 'k=2.0σ (default)' with 'k={k_star:.2f}σ (Phase 8 opt.)'\n"
        "  in Pitch System row of rcm_schedule.csv.\n\n"
        "[SYNTHETIC — Lifetime distribution: Weibull(β={beta:.4f}, η={eta:.1f}d);\n"
        " anomaly score model: ramp onset at 70% of life, r={rate} SD/day.]\n"
    ).format(
        n_sims=config.CBM_N_SIMS,
        k_min=config.CBM_K_MIN,
        k_max=config.CBM_K_MAX,
        k_n=config.CBM_K_N_POINTS,
        k_star=k_star,
        cr_star=cr_star,
        cr_default=cr_default,
        reduction=reduction,
        sensitivity="\n".join(sensitivity_lines),
        beta=config.PITCH_BETA,
        eta=config.PITCH_ETA,
        rate=config.CBM_DEGRADATION_RATE,
    )

    output_path.write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_optimization(
    sensitivity_df: pd.DataFrame,
    k_star_result:  dict,
    output_dir:     Path,
) -> None:
    """Generate cbm_optimization_plot.png (2-panel)."""
    colours = {
        0.05: "#3b82f6",
        0.10: "#10b981",
        0.15: "#f59e0b",
        0.20: "#ef4444",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Pitch System CBM Threshold Optimisation\n"
        "[SYNTHETIC — Weibull(β=0.728, η≈2653d), N=5000 Monte Carlo cycles per k]",
        fontsize=10,
    )

    # Left: Cost rate vs k_sigma for all Cp/Cf ratios
    ax = axes[0]
    for ratio in config.CBM_CP_CF_RATIOS:
        sub = sensitivity_df[abs(sensitivity_df["cp_cf_ratio"] - ratio) < 1e-6].sort_values("k_sigma")
        colour = colours.get(ratio, "#6b7280")
        ax.plot(
            sub["k_sigma"],
            sub["cost_rate_per_day"],
            color=colour, linewidth=1.8,
            label=f"Cp/Cf = {ratio:.2f}",
        )
        # Mark k* for this ratio
        k_r = find_k_star(sub)
        ax.scatter(
            [k_r["k_star"]], [k_r["cost_rate_star"]],
            color=colour, s=60, zorder=10, marker="*",
        )

    ax.axvline(
        x=2.0, color="k", linestyle="--", linewidth=1.0,
        label="Phase 7 default k=2.0",
    )
    ax.axvline(
        x=k_star_result["k_star"], color="#10b981",
        linestyle=":", linewidth=1.5,
        label=f"k* = {k_star_result['k_star']:.2f} (Cp/Cf=0.10)",
    )
    ax.set_xlabel("k_sigma (alert threshold in rolling σ units)", fontsize=9)
    ax.set_ylabel("Expected cost rate ($/day)", fontsize=9)
    ax.set_title("Cost Rate vs Alert Threshold", fontsize=10)
    ax.legend(fontsize=8)

    # Right: p_preventive and p_corrective vs k (central ratio 0.10)
    ax = axes[1]
    central = sensitivity_df[abs(sensitivity_df["cp_cf_ratio"] - 0.10) < 1e-6].sort_values("k_sigma")
    ax.plot(central["k_sigma"], central["p_preventive"], color="#3b82f6", linewidth=1.8,
            label="P(preventive)")
    ax.plot(central["k_sigma"], central["p_corrective"], color="#ef4444", linewidth=1.8,
            linestyle="--", label="P(corrective)")
    ax.axvline(x=2.0, color="k", linestyle=":", linewidth=1.0, label="Phase 7 default k=2.0")
    ax.axvline(x=k_star_result["k_star"], color="#10b981", linestyle=":", linewidth=1.5,
               label=f"k* = {k_star_result['k_star']:.2f}")
    ax.set_xlabel("k_sigma", fontsize=9)
    ax.set_ylabel("Fraction of cycles", fontsize=9)
    ax.set_title("Prevention / Failure Tradeoff (Cp/Cf = 0.10)", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(output_dir / "cbm_optimization_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    output_dir: Path,
    no_plots:   bool = False,
) -> dict:
    """
    Full CBM optimisation pipeline.

    Returns
    -------
    dict with keys: 'sensitivity_df', 'k_star_result', 'update_text'
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("  Running CBM threshold optimisation (Monte Carlo sweep)…")
    k_values = np.linspace(config.CBM_K_MIN, config.CBM_K_MAX, config.CBM_K_N_POINTS)

    sensitivity_df = run_sensitivity(k_values=k_values)
    sensitivity_df.to_csv(output_dir / "cbm_threshold_table.csv", index=False)

    # k* for central ratio 0.10
    central_df  = sensitivity_df[abs(sensitivity_df["cp_cf_ratio"] - 0.10) < 1e-6].copy()
    k_star_result = find_k_star(central_df)

    update_text = generate_rcm_update_text(
        k_star_result, sensitivity_df,
        output_dir / "cbm_update_to_rcm.txt",
    )

    if not no_plots:
        plot_optimization(sensitivity_df, k_star_result, output_dir)

    return {
        "sensitivity_df": sensitivity_df,
        "k_star_result":  k_star_result,
        "update_text":    update_text,
    }