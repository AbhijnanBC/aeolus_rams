"""
aeolus_rams_phase2.hazard
=============================
Section 2.6 — Hazard Rate and the System Bathtub Curve.

For Tier A/B components, a real fitted hazard curve is plotted. For Tier
C, Section 2.6 is explicit: "you cannot plot a fitted hazard curve — plot
nothing fabricated." This module enforces that literally — Tier C
components never reach `matplotlib`'s plotting calls; they only ever
appear as a text annotation reading "shape unknown — insufficient data".

The system-level composite (`render_illustrative_system_bathtub`) is
built and captioned as explicitly illustrative, with the disclaimer baked
into the image itself (not just the surrounding report text), since an
image can be copied out of its original context far more easily than a
paragraph can.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe: never touch a display backend
import matplotlib.pyplot as plt
import numpy as np

from . import config
from .distributions import WeibullFitResult, ExponentialFitResult
from .literature_priors import LiteraturePrior

SHAPE_UNKNOWN_LABEL = "shape unknown — insufficient data"


def weibull_hazard(t: np.ndarray, beta: float, eta: float) -> np.ndarray:
    """h(t) = (beta/eta) * (t/eta)^(beta-1)"""
    t = np.asarray(t, dtype=float)
    return (beta / eta) * (t / eta) ** (beta - 1)


def exponential_hazard(lam: float) -> float:
    """Constant, by definition — matches Section 2.6's exact signature."""
    return lam


def exponential_hazard_curve(t: np.ndarray, lam: float) -> np.ndarray:
    """Broadcast form of `exponential_hazard`, for plotting alongside a
    Weibull curve over the same time axis."""
    t = np.asarray(t, dtype=float)
    return np.full_like(t, lam)


def _default_time_range(mtbf_days: float, n_points: int = 500) -> np.ndarray:
    return np.linspace(1e-6, 3 * mtbf_days, n_points)


def render_component_hazard(
    tier_a_fits: dict[str, WeibullFitResult],
    tier_b_fits: dict[str, ExponentialFitResult],
    tier_c_components: list[str],
    output_path: str | Path,
) -> Path:
    """One figure, one curve per Tier A/B component (real fits only),
    plus a text panel listing every Tier C component as explicitly
    unplotted. Returns the path written."""
    output_path = Path(output_path)
    fig, (ax_curve, ax_text) = plt.subplots(
        1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [2.2, 1]},
    )

    for component, fit in tier_a_fits.items():
        t = _default_time_range(fit.mtbf_days)
        ax_curve.plot(t, weibull_hazard(t, fit.beta, fit.eta),
                      label=f"{component} (Weibull, β={fit.beta:.2f})")

    for component, fit in tier_b_fits.items():
        t = _default_time_range(fit.mtbf_days)
        ax_curve.plot(t, exponential_hazard_curve(t, fit.lambda_rate),
                      linestyle="--",
                      label=f"{component} (exponential, constant hazard)")

    ax_curve.set_xlabel("Time since last failure (days)")
    ax_curve.set_ylabel("Hazard rate h(t)  [failures / day]")
    ax_curve.set_title("Fitted hazard curves — Tier A/B components only")
    if tier_a_fits or tier_b_fits:
        ax_curve.legend(fontsize=8, loc="best")
    else:
        ax_curve.text(0.5, 0.5, "No Tier A/B components with a fit yet",
                       ha="center", va="center", transform=ax_curve.transAxes)

    ax_text.axis("off")
    lines = [SHAPE_UNKNOWN_LABEL.upper() + ":"] + [f"  • {c}" for c in tier_c_components]
    ax_text.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9,
                 transform=ax_text.transAxes, family="monospace")
    ax_text.set_title("Tier C components (not plotted)", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def render_illustrative_system_bathtub(
    tier_a_fits: dict[str, WeibullFitResult],
    tier_b_fits: dict[str, ExponentialFitResult],
    tier_c_priors: dict[str, LiteraturePrior],
    output_path: str | Path,
    t_max_days: float | None = None,
) -> Path:
    """A composite, overlaid hazard picture across ALL components —
    Tier A/B from their real fits, Tier C from their literature-prior
    MTBF under a SIMPLIFYING assumed-exponential model (Section 2.10's
    bridge-to-Phase-3 assumption, borrowed one phase early purely so this
    composite can exist at all).

    Section 2.6 is explicit that a composite built on 11/13 unfit
    components is "only as honest as its weakest input" — the disclaimer
    below is rendered INTO the image itself, not left to the surrounding
    report text, so the picture can't be lifted out of context and shown
    as a real system hazard function.
    """
    output_path = Path(output_path)

    all_mtbfs = (
        [f.mtbf_days for f in tier_a_fits.values()]
        + [f.mtbf_days for f in tier_b_fits.values()]
        + [p.mtbf_days for p in tier_c_priors.values() if p.mtbf_days is not None]
    )
    if t_max_days is None:
        t_max_days = 2 * max(all_mtbfs) if all_mtbfs else 3650.0
    t = np.linspace(1e-6, t_max_days, 500)

    fig, ax = plt.subplots(figsize=(11, 6))

    for component, fit in tier_a_fits.items():
        ax.plot(t, weibull_hazard(t, fit.beta, fit.eta), lw=2,
                label=f"{component} — Tier A (fitted Weibull)")

    for component, fit in tier_b_fits.items():
        ax.plot(t, exponential_hazard_curve(t, fit.lambda_rate), lw=2, linestyle="--",
                label=f"{component} — Tier B (fitted exponential)")

    for component, prior in tier_c_priors.items():
        if prior.mtbf_days is None:
            continue
        lam = 1.0 / prior.mtbf_days
        ax.plot(t, exponential_hazard_curve(t, lam), lw=1, linestyle=":",
                alpha=0.7,
                label=f"{component} — Tier C (literature prior, assumed exponential)")

    n_plotted = len(tier_a_fits) + len(tier_b_fits) + sum(
        1 for p in tier_c_priors.values() if p.mtbf_days is not None
    )

    ax.set_xlabel("Time since last failure (days)")
    ax.set_ylabel("Hazard rate h(t)  [failures / day]")
    ax.set_title("System-level hazard overlay — ILLUSTRATIVE ONLY")
    if n_plotted > 0:
        ax.legend(fontsize=7, loc="upper right", ncol=1)
    else:
        ax.text(0.5, 0.5, "No components with a usable fit or literature prior yet",
                 ha="center", va="center", transform=ax.transAxes)

    disclaimer = (
        "NOT a validated system hazard function.\n"
        f"{len(tier_c_priors)} of {len(tier_a_fits) + len(tier_b_fits) + len(tier_c_priors)} "
        "components shown here are literature-informed priors under a\n"
        "simplifying assumed-exponential model — this composite is only as reliable\n"
        "as its weakest input. See mtbf_table.csv 'confidence' column before citing any curve."
    )
    fig.text(0.5, -0.02, disclaimer, ha="center", va="top", fontsize=9,
              color="firebrick", wrap=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
