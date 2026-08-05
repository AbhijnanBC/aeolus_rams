"""
aeolus_rams_phase7.plots
==========================
All four Phase 7 figures.

Figure 1 — Cost-rate curves: C(T) for each age-replacement component
    Subplot grid (2×2): one panel per component (bearing, gearbox, generator)
    plus the Pitch System β<1 panel showing WHY no minimum exists.
    Each panel annotates T*, C*, C_no_PM, and savings fraction.

Figure 2 — PFDavg vs. proof-test interval for G3b
    X-axis: τ in months. Y-axis: PFDavg (log scale).
    Horizontal lines at SIL-1/2/3 targets.
    Vertical line at τ_SIL2 ≈ 26d. Current operating point annotated.

Figure 3 — RCM summary chart: all 13 components on one visual
    Horizontal bar chart: maintenance interval in years (or "CBM/RTF").
    Colour-coded by RCM category.

Figure 4 — Cost-ratio sensitivity: T* vs Cp/Cf for bearing and gearbox
    Two overlaid curves showing how T* responds to cost uncertainty.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from .config import (
    LITERATURE_WEIBULL, REPLACEMENT_COSTS, DESIGN_LIFE_DAYS,
    LAMBDA_G3b, SIL2_PFDavg_TARGET, SIL_TARGETS,
    PITCH_BETA, PITCH_ETA, PITCH_MTBF,
    TAU_IMPLIED_G3b_DAYS,
)
from .age_replacement import cost_rate, optimal_replacement_age
from .failure_finding import PFDavg_exact, required_proof_test_interval

FONT_SIZE = 10
matplotlib.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 1,
    "axes.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE - 1,
    "figure.dpi": 150,
})

CATEGORY_COLORS = {
    "AGE_REPLACEMENT": "#1976D2",   # blue
    "CONDITION_BASED": "#388E3C",   # green
    "TIME_DIRECTED":   "#F57C00",   # orange
    "FAILURE_FINDING": "#7B1FA2",   # purple
    "RUN_TO_FAILURE":  "#757575",   # grey
}


# ---------------------------------------------------------------------------
# Figure 1: Cost-rate curves
# ---------------------------------------------------------------------------

def plot_cost_rate_curves(output_path: Path) -> Path:
    """Figure 1: C(T) vs T for age-replacement components + Pitch β<1 illustration."""
    from .config import WeibullParams, ReplacementCosts

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "AEOLUS-RAMS Phase 7 — Age-Replacement Cost-Rate Curves\n"
        "C(T) = [Cp·R(T) + Cf·(1−R(T))] / ∫₀ᵀ R(t)dt   (Barlow-Proschan)",
        fontsize=11, y=0.98,
    )

    components_to_plot = [
        ("Main/Rotor Bearing", "Panel A", "#1976D2"),
        ("Gearbox",            "Panel B", "#388E3C"),
        ("Generator",          "Panel C", "#E64A19"),
    ]

    for ax, (comp_name, panel_label, color) in zip(axes.flat[:3], components_to_plot):
        params = LITERATURE_WEIBULL[comp_name]
        costs  = REPLACEMENT_COSTS[comp_name]
        res    = optimal_replacement_age(params, costs)

        T_yrs = res.T_grid / 365.25
        C = res.C_grid

        ax.plot(T_yrs, C, color=color, linewidth=2.0, label="C(T) — cost rate")
        ax.axhline(res.C_no_PM, color="#B71C1C", linewidth=1.2, linestyle="--",
                   label=f"Run-to-failure = {res.C_no_PM:.2f} $/d")

        if res.T_star_days is not None:
            T_star_yr = res.T_star_years
            ax.axvline(T_star_yr, color="#1A237E", linewidth=1.5, linestyle=":",
                       label=f"T* = {T_star_yr:.1f} yr")
            ax.scatter([T_star_yr], [res.C_star], color="#1A237E", zorder=5, s=60)
            # Design life line
            ax.axvline(25.0, color="#78909C", linewidth=1.0, linestyle="-.",
                       alpha=0.6, label="Design life (25 yr)")
            savings_pct = res.savings_fraction * 100
            ax.annotate(
                f"T* = {T_star_yr:.1f} yr\nC* = {res.C_star:.2f} $/d\nSavings {savings_pct:.1f}%",
                xy=(T_star_yr, res.C_star),
                xytext=(T_star_yr + T_yrs[-1] * 0.08, res.C_star + (res.C_no_PM - res.C_star) * 0.3),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="#1A237E"),
            )

        ax.set_title(
            f"{panel_label}: {comp_name}\n"
            f"β={params.beta}, η={params.eta/365.25:.0f}yr, MTTF={res.mttf_days/365.25:.0f}yr"
        )
        ax.set_xlabel("Replacement age T (years)")
        ax.set_ylabel("Cost rate ($/day)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, min(res.T_grid[-1] / 365.25, 80))

    # Panel D: Pitch System β<1 — the canonical "no minimum" illustration
    ax4 = axes.flat[3]
    from .config import WeibullParams, ReplacementCosts
    pitch_params = WeibullParams(
        component="Pitch System",
        beta=PITCH_BETA, eta=PITCH_ETA,
        source="fitted_tier_a",
        citation="Phase 2 mtbf_table.csv: β=0.7285, Tier A Weibull MLE (10 TBF intervals).",
    )
    pitch_costs = ReplacementCosts("Pitch System", Cp=50_000, Cf=200_000)
    res_pitch = optimal_replacement_age(pitch_params, pitch_costs)

    T_yrs_p = res_pitch.T_grid / 365.25
    ax4.plot(T_yrs_p, res_pitch.C_grid, color="#D32F2F", linewidth=2.0)
    ax4.axhline(res_pitch.C_no_PM, color="#B71C1C", linewidth=1.2, linestyle="--",
                label=f"Run-to-failure = {res_pitch.C_no_PM:.2f} $/d")
    ax4.set_title(
        f"Panel D: Pitch System (ILLUSTRATION — no T* exists)\n"
        f"β={PITCH_BETA:.4f} < 1: C(T) monotonically decreasing"
    )
    ax4.set_xlabel("Replacement age T (years)")
    ax4.set_ylabel("Cost rate ($/day)")
    ax4.text(
        0.55, 0.65,
        f"β = {PITCH_BETA:.4f} < 1\n"
        "C(T) decreasing → no minimum\n"
        "PM is always suboptimal\n"
        "→ Use CBM (ML anomaly score)",
        transform=ax4.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFEBEE", alpha=0.9),
    )
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)
    ax4.set_xlim(0, min(T_yrs_p[-1], 40))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Figure 2: PFDavg vs. proof-test interval
# ---------------------------------------------------------------------------

def plot_pfd_curve(output_path: Path) -> Path:
    """Figure 2: PFDavg vs. τ for G3b safety relay with SIL target lines."""
    tau_range = np.linspace(1.0, 730.0, 500)
    pfd_values = np.array([PFDavg_exact(LAMBDA_G3b, tau) for tau in tau_range])
    tau_months = tau_range / 30.44

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogy(tau_months, pfd_values, color="#1976D2", linewidth=2.5, label="PFDavg (exact)")
    ax.semilogy(
        tau_months,
        LAMBDA_G3b * tau_range / 2,
        color="#1976D2", linewidth=1.0, linestyle="--", alpha=0.6,
        label="PFDavg ≈ λτ/2 (approx)",
    )

    # SIL target horizontal lines
    sil_colors = {"SIL-1": "#FFA000", "SIL-2": "#388E3C", "SIL-3": "#7B1FA2"}
    for sil_label, threshold in SIL_TARGETS.items():
        if sil_label in sil_colors:
            ax.axhline(threshold, color=sil_colors[sil_label], linewidth=1.2,
                       linestyle=":", label=f"{sil_label} target: PFDavg ≤ {threshold:.0e}")

    # Required τ for SIL-2
    tau_sil2 = required_proof_test_interval(LAMBDA_G3b, SIL2_PFDavg_TARGET)
    tau_sil2_months = tau_sil2 / 30.44
    ax.axvline(tau_sil2_months, color="#388E3C", linewidth=1.5, linestyle="-",
               label=f"τ_SIL2 = {tau_sil2:.0f}d ({tau_sil2_months:.1f} months)")

    # Current operating point (no proof-testing → τ_implied)
    pfd_current = PFDavg_exact(LAMBDA_G3b, TAU_IMPLIED_G3b_DAYS)
    ax.scatter(
        [TAU_IMPLIED_G3b_DAYS / 30.44], [pfd_current],
        color="#D32F2F", s=80, zorder=5, label=f"Current state (τ={TAU_IMPLIED_G3b_DAYS:.0f}d, PFD={pfd_current:.4f})"
    )
    ax.annotate(
        f"Current:\nPFDavg = {pfd_current:.4f}\n(no proof-test plan)",
        xy=(TAU_IMPLIED_G3b_DAYS / 30.44, pfd_current),
        xytext=(TAU_IMPLIED_G3b_DAYS / 30.44 - 12, pfd_current * 5),
        fontsize=8, color="#D32F2F",
        arrowprops=dict(arrowstyle="->", color="#D32F2F"),
    )

    # Phase 6 ALARP: after SIL-2 relay replacement, new operating point
    from .config import PHASE6_Q_G3b_SIL2
    ax.axhline(PHASE6_Q_G3b_SIL2, color="#1A237E", linewidth=1.0, linestyle="-.",
               alpha=0.8, label=f"Phase 6 target Q_G3b = {PHASE6_Q_G3b_SIL2:.0e} (SIL-2 relay)")

    ax.set_xlabel("Proof-test interval τ (months)")
    ax.set_ylabel("PFDavg (log scale)")
    ax.set_title(
        "AEOLUS-RAMS Phase 7 — Failure-Finding Proof-Test Interval\n"
        f"G3b Safety Relay  |  λ_G3b = {LAMBDA_G3b:.3e}/day  |  "
        f"SIL-2 requires τ ≤ {tau_sil2:.0f}d ({tau_sil2_months:.1f} months)"
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0, tau_range.max() / 30.44)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path



# ---------------------------------------------------------------------------
# Figure 3: RCM summary chart
# ---------------------------------------------------------------------------

def plot_rcm_summary_chart(
    rcm_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Figure 3: All 13 components on one RCM overview visual."""
    fig, ax = plt.subplots(figsize=(11, 7))

    # Assign x-value for each row
    def _x_value(row: pd.Series) -> float:
        if row["maintenance_interval_days"] is not None and not pd.isna(row["maintenance_interval_days"]):
            return float(row["maintenance_interval_days"]) / 365.25
        if row["rcm_category"] in ("CONDITION_BASED", "RUN_TO_FAILURE"):
            return 0.0   # plotted as text only
        return 0.0

    components = rcm_df["component"].tolist()
    y_pos = np.arange(len(components))
    colors = [CATEGORY_COLORS.get(cat, "#757575") for cat in rcm_df["rcm_category"]]
    x_vals = [_x_value(row) for _, row in rcm_df.iterrows()]

    # Horizontal bars
    for i, (x, color, row) in enumerate(zip(x_vals, colors, rcm_df.itertuples())):
        cat = row.rcm_category
        if cat in ("CONDITION_BASED", "RUN_TO_FAILURE") or x == 0:
            label_text = cat.replace("_", " ")
            ax.text(0.3, i, label_text, va="center", ha="left", fontsize=8,
                    color=color, fontweight="bold")
        else:
            ax.barh(i, x, color=color, alpha=0.85, height=0.6, edgecolor="white")
            interval_label = getattr(row, "maintenance_interval_label", f"{x:.1f}yr")
            ax.text(x + 0.1, i, interval_label, va="center", fontsize=8)

    ax.set_yticks(y_pos)
    comp_labels = [
        c[:45] + "…" if len(c) > 45 else c
        for c in components
    ]
    ax.set_yticklabels(comp_labels, fontsize=8)
    ax.set_xlabel("Maintenance interval (years)")
    ax.set_title(
        "AEOLUS-RAMS Phase 7 — RCM Maintenance Schedule Summary\n"
        "All 13 components | Bar length = interval | Category by colour"
    )
    ax.set_xlim(0, 30)
    ax.axvline(25.0, color="#78909C", linewidth=1.0, linestyle="-.", alpha=0.6,
               label="25-yr design life")

    # Legend
    patches = [
        mpatches.Patch(color=c, label=k.replace("_", " "))
        for k, c in CATEGORY_COLORS.items()
    ]
    patches.append(
        mpatches.Patch(color="#78909C", alpha=0.6, label="25yr design life")
    )
    ax.legend(handles=patches, loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Figure 4: T* sensitivity to Cp/Cf ratio
# ---------------------------------------------------------------------------

def plot_cost_ratio_sensitivity(
    sensitivity_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Figure 4: T* (years) vs Cp/Cf ratio for bearing and gearbox."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    comp_colors = {
        "Main/Rotor Bearing": "#1976D2",
        "Gearbox":            "#388E3C",
        "Generator":          "#E64A19",
    }

    for comp_name, color in comp_colors.items():
        sub = sensitivity_df[sensitivity_df["component"] == comp_name]
        if sub.empty:
            continue
        ax1.plot(sub["cp_cf_ratio"], sub["T_star_years"], color=color,
                 linewidth=2.0, label=comp_name)
        ax2.plot(sub["cp_cf_ratio"], sub["savings_fraction"] * 100,
                 color=color, linewidth=2.0, label=comp_name)

    # Design life reference
    ax1.axhline(25.0, color="#78909C", linewidth=1.0, linestyle="-.",
                alpha=0.7, label="25yr design life")

    ax1.set_xlabel("Cp / Cf ratio")
    ax1.set_ylabel("Optimal replacement age T* (years)")
    ax1.set_title("T* vs. Cost Ratio")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Cp / Cf ratio")
    ax2.set_ylabel("Savings vs. run-to-failure (%)")
    ax2.set_title("PM Savings vs. Cost Ratio")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle(
        "AEOLUS-RAMS Phase 7 — Age-Replacement Cost-Ratio Sensitivity\n"
        "Key insight: T* and savings are robust to cost uncertainty for β > 1.5",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path