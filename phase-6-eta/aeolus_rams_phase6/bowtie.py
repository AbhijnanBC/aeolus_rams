"""
aeolus_rams_phase6.bowtie
===========================
Renders the bow-tie diagram combining FTA (left/threats) with ETA
(right/consequences) around the central turbine overspeed hazard.

The bow-tie is the synthesis artifact of AEOLUS Phases 5–6:
  LEFT:    Why protection fails — Phase 5 fault tree, basic events coloured
           by FV importance rank, CCF pathway annotated.
  CENTRE:  The hazard event — Q_top (independent and CCF-adjusted).
  RIGHT:   What happens when protection fails — Phase 6 event tree, four
           consequence branches with absolute frequencies and risk placement.

This matplotlib rendering produces a publication-quality diagram without
requiring draw.io or graphviz as external dependencies.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .event_tree import EventTreeBranch
from . import config as _cfg


# ── Colour scheme ────────────────────────────────────────────────────────────
# Left side: basic events coloured by confidence tier (matching Phase 5 diagrams)
_CONF_COLORS: dict[str, str] = {
    "fitted_tier_a":       "#1565C0",   # blue — directly fitted from CARE data
    "fitted_tier_b":       "#0288D1",   # lighter blue — fitted with assumptions
    "posterior_informed":  "#E65100",   # orange — literature-informed posterior
    "assumed_placeholder": "#757575",   # grey — assumed, not yet data-backed
}

# Right side: consequence branch colours
_BRANCH_COLORS: dict[str, str] = {
    "Branch_1_SAFE_controlled":  "#2E7D32",   # green
    "Branch_2_SAFE_brake":       "#388E3C",   # green
    "Branch_3_NEAR_MISS_SCADA":  "#F57C00",   # amber
    "Branch_4_CATASTROPHIC":     "#C62828",   # red
}

# FTA threat groups: (gate_label, y_position, [(event_name, confidence, Q), ...])
# Q values from Phase 5 gate_Q_table.csv @ 365d — displayed only, not recomputed.
_THREAT_GROUPS: list[tuple[str, float, list[tuple[str, str, float]]]] = [
    ("G1: Pitch System\nFails to Feather", 8.0, [
        ("G1a: Control/Encoder\nfault",       "fitted_tier_a",       0.0820),
        ("G1b: Battery/Elec.\nsupply fault",  "fitted_tier_a",       0.0638),
        ("G1c: Actuator\nmechanical failure", "fitted_tier_a",       0.0365),
    ]),
    ("G2: Mechanical Brake\nFails to Engage", 5.0, [
        ("G2a: Hydraulic\nsupply loss",        "fitted_tier_b",       0.1078),
        ("G2b: Brake hardware\nfault",          "assumed_placeholder", 0.0319),
    ]),
    ("G3: SCADA Overspeed\nTrip Fails", 2.0, [
        ("G3a: Comms/fieldbus\nloss",           "posterior_informed",  0.0059),
        ("G3b: Safety relay/\nelec. fault",     "assumed_placeholder", 0.0271),
    ]),
]

# ETA consequence layout: (branch_label, y_position)
_BRANCH_YS: dict[str, float] = {
    "Branch_1_SAFE_controlled":  8.5,
    "Branch_2_SAFE_brake":       6.5,
    "Branch_3_NEAR_MISS_SCADA":  4.5,
    "Branch_4_CATASTROPHIC":     2.0,
}

_BRANCH_SHORT: dict[str, str] = {
    "Branch_1_SAFE_controlled":  "SAFE\n(Controlled shutdown)",
    "Branch_2_SAFE_brake":       "SAFE\n(Brake engages)",
    "Branch_3_NEAR_MISS_SCADA":  "NEAR-MISS\n(SCADA trip)",
    "Branch_4_CATASTROPHIC":     "CATASTROPHIC\n(Blade ejection risk)",
}


def plot_bowtie(
    gate_Q: dict[str, float],
    consequence_branches: list[EventTreeBranch],
    output_path: Path,
    ccf_uplift: float = _cfg.CCF_UPLIFT_CENTRAL,
    figsize: tuple[float, float] = (20, 11),
) -> Path:
    """
    Render the bow-tie diagram to a PNG file.

    Parameters
    ----------
    gate_Q : dict[str, float]
        Gate Q values from load_gate_Q_values() — used for Q_top label.
    consequence_branches : list[EventTreeBranch]
        Four ETA branches at λ_IE central estimate (independent, not CCF).
    output_path : Path
        Destination PNG file path.
    ccf_uplift : float
        CCF uplift factor for annotating the hazard centre box.

    Returns
    -------
    Path
        The output_path argument.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 11)
    ax.axis("off")

    # ── Centre: Hazard box ────────────────────────────────────────────────
    Q_top_indep = gate_Q.get(_cfg.GATE_NAMES["TE"], float("nan"))
    Q_top_ccf   = Q_top_indep * ccf_uplift

    hazard_box = mpatches.FancyBboxPatch(
        (8.0, 3.8), 4.0, 3.4,
        boxstyle="round,pad=0.15",
        facecolor="#FFCDD2", edgecolor="#B71C1C", linewidth=2.5,
    )
    ax.add_patch(hazard_box)
    ax.text(
        10.0, 6.3,
        "TURBINE OVERSPEED\nHAZARD",
        ha="center", va="center", fontsize=10, fontweight="bold", color="#B71C1C",
    )
    ax.text(
        10.0, 5.5,
        "Rotor exceeds ≥120%\nrated speed",
        ha="center", va="center", fontsize=8, color="#C62828",
    )
    ax.text(
        10.0, 4.65,
        f"Q_top = {Q_top_indep:.2e} (independent)\n"
        f"Q_top = {Q_top_ccf:.2e} (CCF β=0.10)",
        ha="center", va="center", fontsize=7.5, color="#555555",
    )

    # ── Left: FTA threats ─────────────────────────────────────────────────
    for gate_label, gate_y, events in _THREAT_GROUPS:
        # Gate diamond
        gate_x = 6.5
        diamond = plt.Polygon(
            [[gate_x, gate_y + 0.35],
             [gate_x + 0.45, gate_y],
             [gate_x, gate_y - 0.35],
             [gate_x - 0.45, gate_y]],
            closed=True,
            facecolor="#FFF9C4", edgecolor="#F9A825", linewidth=1.5,
        )
        ax.add_patch(diamond)
        ax.text(gate_x, gate_y, gate_label,
                ha="center", va="center", fontsize=7, fontweight="bold",
                multialignment="center")

        # Arrow from gate to hazard
        ax.annotate(
            "", xy=(8.0, 5.5), xytext=(gate_x + 0.46, gate_y),
            arrowprops=dict(arrowstyle="->", color="#888888", lw=1.3),
        )

        # Basic events feeding the gate
        n = len(events)
        for j, (ev_name, conf, Q_ev) in enumerate(events):
            ev_y = gate_y + (j - (n - 1) / 2) * 1.1
            ev_x = 3.8
            color = _CONF_COLORS.get(conf, "#777777")
            ev_box = mpatches.FancyBboxPatch(
                (ev_x - 0.85, ev_y - 0.35), 1.7, 0.7,
                boxstyle="round,pad=0.1",
                facecolor="#E3F2FD", edgecolor=color, linewidth=1.5,
            )
            ax.add_patch(ev_box)
            ax.text(ev_x, ev_y,
                    f"{ev_name}\nQ={Q_ev:.4f}",
                    ha="center", va="center", fontsize=7,
                    color="#1A237E")
            ax.annotate(
                "", xy=(gate_x - 0.46, gate_y), xytext=(ev_x + 0.85, ev_y),
                arrowprops=dict(arrowstyle="-", color=color, lw=1.0),
            )

    # AND gate symbol (all three G1/G2/G3 must fail → top event)
    and_x, and_y = 7.3, 5.5
    ax.text(and_x, and_y, "AND",
            ha="center", va="center", fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#F3E5F5",
                      edgecolor="#7B1FA2", linewidth=1.5))

    # CCF annotation: dashed red arc between G2 (y=5.0) and G1 (y=8.0)
    ax.annotate(
        "", xy=(6.0, 7.5), xytext=(6.0, 5.4),
        arrowprops=dict(
            arrowstyle="<->", color="#CC0000", lw=1.5,
            linestyle="dashed", connectionstyle="arc3,rad=0.3"
        ),
    )
    ax.text(5.3, 6.5,
            "CCF\nhydraulic\nβ=0.10",
            ha="center", va="center", fontsize=7,
            color="#CC0000", style="italic")

    # ── Right: ETA consequence branches ──────────────────────────────────
    branch_map = {b.label: b for b in consequence_branches}
    for label, branch_y in _BRANCH_YS.items():
        branch = branch_map.get(label)
        color  = _BRANCH_COLORS.get(label, "#1A237E")
        short  = _BRANCH_SHORT.get(label, label)

        # Arrow from hazard to consequence
        ax.annotate(
            "", xy=(14.0, branch_y), xytext=(12.0, 5.5),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        )

        # Consequence box
        cons_box = mpatches.FancyBboxPatch(
            (14.0, branch_y - 0.55), 5.5, 1.1,
            boxstyle="round,pad=0.1",
            facecolor=color, alpha=0.18,
            edgecolor=color, linewidth=1.8,
        )
        ax.add_patch(cons_box)

        if branch:
            freq_str = f"λ = {branch.lambda_outcome_per_turbine:.2e}/turbine/yr"
            risk_str = (f"P(branch|IE) = {branch.P_branch_given_IE:.4f}   "
                        f"[{branch.risk_acceptability()}]")
        else:
            freq_str = ""
            risk_str = ""

        ax.text(16.75, branch_y + 0.2,
                short,
                ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=color)
        ax.text(16.75, branch_y - 0.2,
                freq_str,
                ha="center", va="center", fontsize=7.5, color="#333333")
        ax.text(16.75, branch_y - 0.42,
                risk_str,
                ha="center", va="center", fontsize=7, color="#555555")

    # ── Title and footer ─────────────────────────────────────────────────
    ax.set_title(
        "AEOLUS-RAMS — Bow-Tie Diagram: Turbine Overspeed Hazard\n"
        "◄  FTA: Why protection fails (Phase 5)   |   HAZARD   |   "
        "ETA: What happens when it fails (Phase 6)  ►",
        fontsize=11, pad=14,
    )

    ax.text(
        10.0, 0.4,
        f"Q_top (independent) = {Q_top_indep:.3e}   |   "
        f"CCF uplift (β=0.10) = {ccf_uplift:.4f}×   |   "
        f"λ_IE central = {_cfg.LAMBDA_IE_CENTRAL}/turbine/yr   |   "
        f"N_turbines = {_cfg.N_TURBINES}",
        ha="center", va="bottom", fontsize=8, color="#555555",
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor="#FAFAFA", edgecolor="#CCCCCC"),
    )

    # Confidence legend (left-side colours)
    conf_patches = [
        mpatches.Patch(color=_CONF_COLORS["fitted_tier_a"],       label="Fitted (Tier A — CARE data)"),
        mpatches.Patch(color=_CONF_COLORS["fitted_tier_b"],       label="Fitted (Tier B — with assumptions)"),
        mpatches.Patch(color=_CONF_COLORS["posterior_informed"],  label="Posterior-informed (literature)"),
        mpatches.Patch(color=_CONF_COLORS["assumed_placeholder"], label="Assumed placeholder"),
    ]
    ax.legend(handles=conf_patches, loc="lower left",
              fontsize=8, framealpha=0.92, title="Data confidence (FTA left side)")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path