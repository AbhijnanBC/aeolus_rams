"""
aeolus_rams_phase6.risk_matrix
=================================
Generates the IEC 61400-1 Ed.4 Annex K risk matrix visualization.

Places:
  (1) All four ETA branches from the turbine overspeed event tree.
  (2) The BoP export cable availability risk from Phase 4 (different category).
  (3) Optional: ALARP-improved positions for barrier upgrade scenarios.

The matrix is the central deliverable of Phase 6 — it makes the risk
comparison between safety risk (overspeed) and economic risk (BoP cable)
explicit on a single plot, which is the closing argument of the AEOLUS project.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from . import config as _cfg


CELL_COLORS: dict[str, str] = {
    "UNACCEPTABLE": "#EF5350",
    "ALARP":        "#FFA726",
    "ACCEPTABLE":   "#66BB6A",
}

# Row 0 = Cat A (Extreme), Row 4 = Cat E (Negligible)
# Col 0 = F1 (Frequent),   Col 4 = F5 (Extremely Unlikely)
_MATRIX_CELLS: list[list[str]] = [
    ["UNACCEPTABLE", "UNACCEPTABLE", "UNACCEPTABLE", "ALARP",        "ACCEPTABLE"],
    ["UNACCEPTABLE", "UNACCEPTABLE", "ALARP",        "ACCEPTABLE",   "ACCEPTABLE"],
    ["UNACCEPTABLE", "ALARP",        "ACCEPTABLE",   "ACCEPTABLE",   "ACCEPTABLE"],
    ["ALARP",        "ACCEPTABLE",   "ACCEPTABLE",   "ACCEPTABLE",   "ACCEPTABLE"],
    ["ACCEPTABLE",   "ACCEPTABLE",   "ACCEPTABLE",   "ACCEPTABLE",   "ACCEPTABLE"],
]

_SEV_DISPLAY: list[str] = [
    "A\n(Extreme)", "B\n(Major)", "C\n(Significant)", "D\n(Minor)", "E\n(Negligible)"
]
_FREQ_DISPLAY: list[str] = [
    "F1\n≥10⁻¹/yr", "F2\n10⁻²–10⁻¹", "F3\n10⁻³–10⁻²", "F4\n10⁻⁴–10⁻³", "F5\n<10⁻⁴/yr"
]

_SEV_IDX:  dict[str, int] = {s: i for i, s in enumerate("ABCDE")}
_FREQ_IDX: dict[str, int] = {"F1": 0, "F2": 1, "F3": 2, "F4": 3, "F5": 4}

# Marker shapes per point — cycle if more than 7 points
_MARKERS = ["o", "s", "D", "^", "v", "P", "*"]


def _jitter(col: int, row: int, index: int, n_at_cell: int = 1) -> tuple[float, float]:
    """Slight positional jitter so overlapping points in the same cell separate."""
    offsets_x = [-0.15, 0.0, 0.15, -0.1, 0.1]
    offsets_y = [0.0, 0.15, -0.15, -0.1, 0.1]
    i = index % len(offsets_x)
    return col + 0.5 + offsets_x[i], (4 - row) + 0.5 + offsets_y[i]


def plot_risk_matrix(
    event_points: list[dict],
    output_path: Path,
    title_suffix: str = "",
    figsize: tuple[float, float] = (11, 8.5),
) -> Path:
    """
    Plot a 5×5 IEC 61400-1 risk matrix with event points placed on it.

    Parameters
    ----------
    event_points : list of dict
        Each dict requires: label (str), severity (str, A-E),
        freq_cat (str, F1-F5). Optional: color (str), marker (str),
        annotation (str, extra label text).
    output_path : Path
        Where to save the PNG.
    title_suffix : str
        Appended to the figure title (e.g. "— ALARP Option 1 Applied").

    Returns
    -------
    Path
        The output_path argument (for chaining).
    """
    fig, ax = plt.subplots(figsize=figsize)

    # ── Draw coloured cells ────────────────────────────────────────────────
    for row in range(5):
        for col in range(5):
            accept = _MATRIX_CELLS[row][col]
            color  = CELL_COLORS[accept]
            rect = plt.Rectangle(
                (col, 4 - row), 1, 1,
                facecolor=color, alpha=0.30,
                edgecolor="white", linewidth=2,
            )
            ax.add_patch(rect)
            ax.text(
                col + 0.5, (4 - row) + 0.5, accept,
                ha="center", va="center",
                fontsize=7.5, color="#2E2E2E", fontweight="bold",
            )

    # ── Group points by cell for jitter ───────────────────────────────────
    from collections import defaultdict
    cell_counts: dict[tuple[int, int], int] = defaultdict(int)

    for i, pt in enumerate(event_points):
        sev_i  = _SEV_IDX[pt["severity"]]
        freq_i = _FREQ_IDX[pt["freq_cat"]]
        jitter_idx = cell_counts[(sev_i, freq_i)]
        cell_counts[(sev_i, freq_i)] += 1

        x, y = _jitter(freq_i, sev_i, jitter_idx)
        m     = pt.get("marker", _MARKERS[i % len(_MARKERS)])
        color = pt.get("color", "#1A237E")

        ax.scatter(x, y, s=200, marker=m, color=color,
                   zorder=5, linewidths=1.5, edgecolors="white")
        label_text = pt["label"]
        if pt.get("annotation"):
            label_text += f"\n{pt['annotation']}"
        ax.text(x + 0.04, y + 0.14, label_text,
                fontsize=7.5, color=color, fontweight="bold",
                ha="left", va="bottom")

    # ── Axes ──────────────────────────────────────────────────────────────
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xticks(np.arange(0.5, 5.5))
    ax.set_xticklabels(_FREQ_DISPLAY, fontsize=9)
    ax.set_yticks(np.arange(0.5, 5.5))
    ax.set_yticklabels(_SEV_DISPLAY[::-1], fontsize=9)
    ax.set_xlabel("Event Frequency (per turbine per year)", fontsize=10, labelpad=8)
    ax.set_ylabel("Consequence Severity — IEC 61400-1 Ed.4 Annex K", fontsize=10, labelpad=8)
    title = "AEOLUS-RAMS Phase 6 — Risk Matrix (IEC 61400-1 Ed.4)"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title, fontsize=11, pad=12)

    legend_patches = [
        mpatches.Patch(color=CELL_COLORS["UNACCEPTABLE"], alpha=0.7,
                       label="Unacceptable (design change required)"),
        mpatches.Patch(color=CELL_COLORS["ALARP"],        alpha=0.7,
                       label="ALARP (formal risk reduction case required)"),
        mpatches.Patch(color=CELL_COLORS["ACCEPTABLE"],   alpha=0.7,
                       label="Acceptable"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9, framealpha=0.92)
    ax.grid(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_risk_points(
    cft: "pd.DataFrame",
    lambda_IE: float = _cfg.LAMBDA_IE_CENTRAL,
    ccf_adjusted: bool = False,
    include_bop: bool = True,
) -> list[dict]:
    """
    Build the list of event_points dicts for plot_risk_matrix().

    Parameters
    ----------
    cft : pd.DataFrame
        Output of event_tree.consequence_frequency_table().
    lambda_IE : float
        Initiating event frequency to extract from the table.
    ccf_adjusted : bool
        Whether to use CCF-adjusted or independent rows.
    include_bop : bool
        Whether to include the BoP cable failure point from Phase 4.

    Returns
    -------
    list of dict suitable for plot_risk_matrix(event_points=...).
    """
    import pandas as pd
    branch_colors = {
        "Branch_1_SAFE_controlled": "#2E7D32",
        "Branch_2_SAFE_brake":      "#388E3C",
        "Branch_3_NEAR_MISS_SCADA": "#F57C00",
        "Branch_4_CATASTROPHIC":    "#C62828",
    }
    branch_short = {
        "Branch_1_SAFE_controlled": "B1: SAFE\n(controlled)",
        "Branch_2_SAFE_brake":      "B2: SAFE\n(brake)",
        "Branch_3_NEAR_MISS_SCADA": "B3: Near-miss\n(SCADA)",
        "Branch_4_CATASTROPHIC":    "B4: CATASTROPHIC",
    }

    subset = cft[
        (np.isclose(cft["lambda_IE"], lambda_IE, rtol=1e-6)) &
        (cft["ccf_adjusted"] == ccf_adjusted)
    ]

    points: list[dict] = []
    for _, row in subset.iterrows():
        bl = row["branch_label"]
        points.append({
            "label":      branch_short.get(bl, bl),
            "severity":   row["severity_category"],
            "freq_cat":   row["freq_category"],
            "color":      branch_colors.get(bl, "#1A237E"),
            "annotation": f"λ={row['lambda_per_turbine']:.1e}/yr",
        })

    if include_bop:
        bop_freq_cat = _cfg.freq_category(_cfg.BOP_CABLE_LAMBDA_PER_TURBINE)
        points.append({
            "label":      "BoP Cable\n(Phase 4)",
            "severity":   _cfg.BOP_CABLE_SEVERITY,
            "freq_cat":   bop_freq_cat,
            "color":      "#6A1B9A",
            "marker":     "P",
            "annotation": f"λ={_cfg.BOP_CABLE_LAMBDA_PER_TURBINE:.2f}/yr\n(economic, not safety)",
        })

    return points