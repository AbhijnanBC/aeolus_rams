"""
aeolus_rams_phase4.plots
==========================
Generates the four required output figures.

Figure 1 — Farm availability distribution (Baseline)
    Histogram of A_farm across 10,000 simulations with vertical lines
    at mean and 5th/95th percentiles. Annotated with numeric values.

Figure 2 — Three-scenario comparison
    Overlapping filled histograms (Baseline, Optimised, Degraded) using
    config.SCENARIOS colour codes. Highlights the value of SOV investment
    and the operational risk exposure.

Figure 3 — Bottleneck sensitivity tornado chart
    Horizontal bar chart of (A_farm_best - A_farm_worst) for each sweep.
    Sorted by impact. The longest bar shows which single parameter
    improvement yields the largest gain in mean farm availability.

Figure 4 — BoP vs. turbine contribution to unavailability
    Stacked bar: 1 - A_farm decomposed into (1 - A_kofN_exact) and
    (1 - A_bop), per scenario. Shows operator where to invest.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for pipeline runs)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from . import config


# ── Plotting defaults ──────────────────────────────────────────────────────
FONT_SIZE = 11
matplotlib.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 1,
    "axes.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE - 1,
    "figure.dpi": 150,
})


def plot_availability_distribution(
    mc_df: pd.DataFrame,
    output_path: Path,
    scenario_label: str = "Baseline",
) -> Path:
    """Figure 1 — Farm availability distribution for one scenario.

    Parameters
    ----------
    mc_df : pd.DataFrame
        Output of run_monte_carlo() for the baseline scenario.
    output_path : Path
        Where to write the PNG.
    scenario_label : str
        Scenario name for the title.
    """
    A = mc_df["A_farm"].values
    mean_A = A.mean()
    p5 = np.percentile(A, 5)
    p95 = np.percentile(A, 95)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Histogram
    n_bins = min(60, int(np.sqrt(len(A))) * 2)
    counts, bin_edges, patches = ax.hist(
        A, bins=n_bins, color="#2196F3", alpha=0.7, edgecolor="white",
        linewidth=0.4, label="Simulated A_farm"
    )

    # Shade 5th–95th percentile band
    for patch, left, right in zip(patches, bin_edges[:-1], bin_edges[1:]):
        if p5 <= (left + right) / 2 <= p95:
            patch.set_facecolor("#90CAF9")
            patch.set_alpha(0.85)

    # Vertical lines
    ax.axvline(mean_A, color="#1565C0", linewidth=2, linestyle="-", label=f"Mean = {mean_A:.4f}")
    ax.axvline(p5, color="#F44336", linewidth=1.5, linestyle="--", label=f"P5 = {p5:.4f}")
    ax.axvline(p95, color="#F44336", linewidth=1.5, linestyle="--", label=f"P95 = {p95:.4f}")

    # Annotation
    ax.text(
        0.97, 0.95,
        f"Mean A_farm = {mean_A:.4f}\n"
        f"90% of years: [{p5:.4f}, {p95:.4f}]\n"
        f"n = {len(A):,} simulations",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=FONT_SIZE - 1,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    ax.set_xlabel("Annual Farm Availability (A_farm)")
    ax.set_ylabel("Count of simulated years")
    ax.set_title(f"AEOLUS-RAMS Phase 4 — Farm Availability Distribution ({scenario_label})")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_scenario_comparison(
    mc_dfs: dict[str, pd.DataFrame],
    output_path: Path,
) -> Path:
    """Figure 2 — Three-scenario A_farm distribution overlay.

    Parameters
    ----------
    mc_dfs : dict[str, pd.DataFrame]
        Keys: 'baseline', 'optimised', 'degraded'. Values: MC DataFrames.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    handles = []
    for key in ["degraded", "baseline", "optimised"]:  # plot in order: worst→best
        if key not in mc_dfs:
            continue
        A = mc_dfs[key]["A_farm"].values
        sc = config.SCENARIOS[key]
        mean_A = A.mean()
        n_bins = min(60, int(np.sqrt(len(A))) * 2)
        ax.hist(
            A, bins=n_bins, color=sc["color"], alpha=0.50,
            edgecolor="white", linewidth=0.3,
            label=f"{sc['label']} (mean={mean_A:.4f})",
        )
        ax.axvline(mean_A, color=sc["color"], linewidth=2.0, linestyle="-")

    ax.set_xlabel("Annual Farm Availability (A_farm)")
    ax.set_ylabel("Count of simulated years")
    ax.set_title("AEOLUS-RAMS Phase 4 — Three-Scenario Comparison")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_tornado_chart(
    tornado_df: pd.DataFrame,
    output_path: Path,
    baseline_A_farm: float,
) -> Path:
    """Figure 3 — Bottleneck sensitivity tornado chart.

    Parameters
    ----------
    tornado_df : pd.DataFrame
        Output of sensitivity.build_tornado_data().
        Required columns: sweep_name, A_farm_at_worst, A_farm_at_best,
        impact_total, baseline_A_farm.
    output_path : Path
        Where to write the PNG.
    baseline_A_farm : float
        Mean baseline A_farm — used to centre the bars.
    """
    display_names = {
        "pitch": "Pitch System MTBF",
        "hydraulic": "Hydraulic System MTBF",
        "cable": "Export Cable MTBF",
        "mttr": "MTTR Multiplier (all)",
    }

    df = tornado_df.sort_values("impact_total", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.8 + 1.5)))

    for i, row in df.iterrows():
        # Bar from A_farm_at_worst to A_farm_at_best
        left = row["A_farm_at_worst"] - baseline_A_farm
        right = row["A_farm_at_best"] - baseline_A_farm
        width = right - left

        # Left portion (below baseline): red
        if left < 0:
            ax.barh(
                i, min(right, 0) - left, left=left + baseline_A_farm,
                color="#F44336", alpha=0.8, height=0.6,
            )
        # Right portion (above baseline): green
        if right > 0:
            ax.barh(
                i, right - max(left, 0), left=max(left, 0) + baseline_A_farm,
                color="#4CAF50", alpha=0.8, height=0.6,
            )

        # Total impact label
        ax.text(
            max(row["A_farm_at_best"], baseline_A_farm) + 0.001,
            i, f"Δ{row['impact_total']:.4f}",
            va="center", fontsize=FONT_SIZE - 2,
        )

    ax.axvline(baseline_A_farm, color="black", linewidth=1.2, linestyle="-", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [display_names.get(row["sweep_name"], row["sweep_name"])
         for _, row in df.iterrows()],
        fontsize=FONT_SIZE - 1,
    )
    ax.set_xlabel("Mean A_farm")
    ax.set_title(
        "AEOLUS-RAMS Phase 4 — Bottleneck Sensitivity Tornado Chart\n"
        f"(baseline A_farm = {baseline_A_farm:.4f})"
    )

    # Legend
    red_patch = mpatches.Patch(color="#F44336", alpha=0.8, label="Worse than baseline")
    green_patch = mpatches.Patch(color="#4CAF50", alpha=0.8, label="Better than baseline")
    ax.legend(handles=[green_patch, red_patch], loc="lower right", fontsize=FONT_SIZE - 2)

    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_bop_vs_turbine(
    summary_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Figure 4 — BoP vs. turbine contribution to unavailability.

    Decomposes 1 - A_farm into:
      (a) turbine contribution: 1 - A_kofN_exact  (k-of-N threshold failures)
      (b) BoP contribution:    1 - A_bop          (cable or substation failures)

    Note: these two contributions are NOT strictly additive (the farm can be
    down for turbine AND BoP reasons simultaneously), but for A_farm close
    to 1 (as expected here), 1 - A_farm ≈ (1-A_kofN) + (1-A_bop) is a
    good approximation. For the exact decomposition, use:
        P(farm down) = P(kofN fails OR BoP fails)
                     = P(kofN fails) + P(BoP fails) - P(both fail)
    The plot shows the approximate decomposition with a footnote.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Stacked summarise_mc_results() output for all three scenarios.
    """
    scenarios_order = ["baseline", "optimised", "degraded"]
    scenario_labels = {k: config.SCENARIOS[k]["label"] for k in config.SCENARIOS}
    scenario_colors = {k: config.SCENARIOS[k]["color"] for k in config.SCENARIOS}

    # Extract mean values per scenario from summary_df
    def _get_mean(scenario: str, metric: str) -> float:
        row = summary_df[
            (summary_df["scenario"] == scenario) & (summary_df["metric"] == metric)
        ]
        return float(row["mean"].iloc[0]) if not row.empty else float("nan")

    x = np.arange(len(scenarios_order))
    width = 0.5

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, key in enumerate(scenarios_order):
        A_farm = _get_mean(key, "A_farm")
        A_kofN = _get_mean(key, "A_kofN_exact")
        A_bop = _get_mean(key, "A_bop")

        unavail_turbine = max(0.0, 1.0 - A_kofN)
        unavail_bop = max(0.0, 1.0 - A_bop)

        # Stacked: BoP on bottom, turbine on top
        ax.bar(
            i, unavail_bop, width, color="#FF8F00", alpha=0.85,
            label="BoP unavailability" if i == 0 else "",
        )
        ax.bar(
            i, unavail_turbine, width, bottom=unavail_bop,
            color="#1976D2", alpha=0.85,
            label="Turbine k-of-N unavailability" if i == 0 else "",
        )
        # Total farm unavailability marker
        ax.hlines(
            1.0 - A_farm, i - width / 2, i + width / 2,
            color="black", linewidth=2.0, linestyle="--",
            label="Total 1-A_farm" if i == 0 else "",
        )
        ax.text(
            i, 1.0 - A_farm + 0.001, f"{1-A_farm:.4f}",
            ha="center", va="bottom", fontsize=FONT_SIZE - 2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([scenario_labels.get(k, k) for k in scenarios_order])
    ax.set_ylabel("Fraction of time unavailable (1 - A)")
    ax.set_title(
        "AEOLUS-RAMS Phase 4 — BoP vs. Turbine Contribution to Unavailability\n"
        "(dashed line = total farm unavailability; bars = approximate decomposition)"
    )
    ax.legend(loc="upper right", fontsize=FONT_SIZE - 2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path