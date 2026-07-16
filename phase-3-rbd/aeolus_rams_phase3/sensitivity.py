"""
aeolus_rams_phase3.sensitivity
================================
Section 3.7 — Sensitivity analysis for the six `not_yet_sourced`
components whose MTBFs are carried as Option A placeholders.

Purpose
--------
Option A placeholders are the best available estimates, but they carry
meaningful epistemic uncertainty. This module quantifies how much
R_turbine(5yr) moves if the six placeholder MTBFs turn out to be
anywhere in the range 500–20,000 days — a range that brackets every
published wind-turbine subassembly figure in the literature.

The aggregate treatment (all 6 unknowns share the same swept MTBF)
intentionally overestimates the uncertainty — real sub-assembly failure
rates span the range and are not all equal — but it gives a rigorous
outer bound: "regardless of where in this range the true values land,
R_turbine(5yr) stays between X and Y."

Separately, a per-component sweep is also provided, varying one
component at a time while holding the other five at their Option A
values. This identifies which of the six unknowns matters most —
expected to be Yaw System and Mechanical Brake, because their
Option A MTBFs are the shortest (lowest reliability).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from . import config
from .component_rt import ComponentRT
from .turbine_rbd import lambda_system, R_turbine_exponential


# ---------------------------------------------------------------------------
# Aggregate sweep: all 6 unknowns assumed equal MTBF
# ---------------------------------------------------------------------------

def aggregate_sensitivity(
    components: dict[str, ComponentRT],
    t: float = config.T_5YR,
    mtbf_range: tuple[float, float] = config.SENSITIVITY_MTBF_RANGE,
    n_points: int = config.SENSITIVITY_N_POINTS,
) -> pd.DataFrame:
    """R_turbine(t) as ALL six placeholder MTBFs are simultaneously varied.

    Columns: assumed_mtbf_days, R_turbine_aggregate_sweep,
             lambda_placeholder_total, lambda_known_total.
    """
    placeholder_names = {
        name for name, c in components.items() if c.is_placeholder
    }
    lam_known = sum(
        c.lambda_per_day for name, c in components.items()
        if name not in placeholder_names
    )
    n_unknowns = len(placeholder_names)

    rows = []
    for mtbf_unknown in np.linspace(mtbf_range[0], mtbf_range[1], n_points):
        lam_unknown_total = n_unknowns / mtbf_unknown
        lam_total = lam_known + lam_unknown_total
        R_sys = float(np.exp(-lam_total * t))
        rows.append({
            "assumed_placeholder_mtbf_days": round(mtbf_unknown, 1),
            "lambda_known_per_day": lam_known,
            "lambda_placeholder_total_per_day": lam_unknown_total,
            "lambda_system_per_day": lam_total,
            "R_turbine": R_sys,
            "Q_turbine": 1.0 - R_sys,
            "t_days": t,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-component sweep: vary one at a time, hold others at Option A
# ---------------------------------------------------------------------------

def per_component_sensitivity(
    components: dict[str, ComponentRT],
    t: float = config.T_5YR,
    mtbf_range: tuple[float, float] = config.SENSITIVITY_MTBF_RANGE,
    n_points: int = config.SENSITIVITY_N_POINTS,
) -> pd.DataFrame:
    """Vary one placeholder component at a time; hold others at Option A.

    Columns: component, assumed_mtbf_days, R_turbine, delta_R_from_option_a.
    """
    placeholder_names = [name for name, c in components.items() if c.is_placeholder]
    lam_base = lambda_system(components)
    R_base = float(np.exp(-lam_base * t))

    rows = []
    for comp_name in placeholder_names:
        c_option_a = components[comp_name]
        for mtbf_varied in np.linspace(mtbf_range[0], mtbf_range[1], n_points):
            # Replace this component's lambda, keep all others at Option A
            lam_adjusted = (
                lam_base
                - c_option_a.lambda_per_day
                + (1.0 / mtbf_varied)
            )
            R_adj = float(np.exp(-lam_adjusted * t))
            rows.append({
                "component": comp_name,
                "assumed_mtbf_days": round(mtbf_varied, 1),
                "R_turbine": R_adj,
                "delta_R_from_option_a": R_adj - R_base,
                "t_days": t,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reference points: Option A values and their R_turbine
# ---------------------------------------------------------------------------

def option_a_reference(
    components: dict[str, ComponentRT],
    t: float = config.T_5YR,
) -> pd.DataFrame:
    """The actual Option A placeholder values and corresponding R_turbine.

    Used to mark the 'current estimate' on the sensitivity band plot."""
    lam_base = lambda_system(components)
    R_base = float(np.exp(-lam_base * t))
    rows = []
    for name, c in components.items():
        if c.is_placeholder:
            rows.append({
                "component": name,
                "option_a_mtbf_days": c.mtbf_days,
                "R_turbine_at_option_a": R_base,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot: sensitivity band
# ---------------------------------------------------------------------------

def render_sensitivity_band(
    components: dict[str, ComponentRT],
    output_path: str | Path,
    t: float = config.T_5YR,
) -> Path:
    """Sensitivity band plot: R_turbine(5yr) vs assumed placeholder MTBF.

    Two panels:
    Left — Aggregate sweep: all 6 unknowns vary together.
    Right — Per-component sweep: each unknown varied independently.

    The shaded band on the left panel represents the range of outcomes
    purely from uncertainty in the 6 not_yet_sourced components, with
    known components' λ values held constant.
    """
    output_path = Path(output_path)
    agg = aggregate_sensitivity(components, t=t)
    per_comp = per_component_sensitivity(components, t=t)
    ref = option_a_reference(components, t=t)

    R_option_a = float(agg.loc[
        np.abs(agg["assumed_placeholder_mtbf_days"] - agg["assumed_placeholder_mtbf_days"].mean()).idxmin(),
        "R_turbine"
    ])
    # Use the actual Option A R_turbine (computed from lambda_system)
    from .turbine_rbd import lambda_system as ls
    lam_base = ls(components)
    R_base = float(np.exp(-lam_base * t))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: aggregate band ---
    ax1.plot(
        agg["assumed_placeholder_mtbf_days"],
        agg["R_turbine"],
        color="#2980b9", lw=2.5, label="R_turbine (all 6 unknowns = swept value)",
    )
    ax1.axhline(R_base, color="#e74c3c", ls="--", lw=1.5,
                label=f"Option A point estimate (R={R_base:.4f})")
    ax1.fill_between(
        agg["assumed_placeholder_mtbf_days"], agg["R_turbine"], R_base,
        alpha=0.15, color="#2980b9", label="Uncertainty band",
    )
    ax1.set_xlabel("Assumed MTBF for all 6 placeholder components (days)")
    ax1.set_ylabel(f"R_turbine({int(t/365.25)}yr)")
    ax1.set_title("Aggregate sensitivity\n(all 6 unknowns swept together)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(config.SENSITIVITY_MTBF_RANGE)

    R_min = float(agg["R_turbine"].min())
    R_max = float(agg["R_turbine"].max())
    ax1.annotate(
        f"Range: [{R_min:.4f}, {R_max:.4f}]\n"
        f"Δ from best to worst assumption: {R_max-R_min:.4f}",
        xy=(0.97, 0.05), xycoords="axes fraction",
        ha="right", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    # --- Right: per-component ---
    cmap = plt.get_cmap("tab10")
    placeholder_names = per_comp["component"].unique()
    for i, comp in enumerate(placeholder_names):
        sub = per_comp[per_comp["component"] == comp].sort_values("assumed_mtbf_days")
        ax2.plot(
            sub["assumed_mtbf_days"], sub["R_turbine"],
            color=cmap(i), lw=1.8, label=comp[:28],
        )

    ax2.axhline(R_base, color="#e74c3c", ls="--", lw=1.5,
                label=f"Option A baseline (R={R_base:.4f})")
    ax2.set_xlabel("Assumed MTBF for this component only (days)")
    ax2.set_ylabel(f"R_turbine({int(t/365.25)}yr)")
    ax2.set_title("Per-component sensitivity\n(one unknown varied, others at Option A)")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(config.SENSITIVITY_MTBF_RANGE)

    fig.suptitle(
        "Sensitivity of R_turbine to Option A MTBF assumptions\n"
        "for six not-yet-sourced components (Section 3.7)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
