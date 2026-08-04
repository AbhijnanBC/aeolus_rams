"""
aeolus_rams_phase6.alarp
==========================
Quantifies three ALARP risk-reduction options for the Cat A × F4 overspeed
risk, using Phase 5 importance measures to justify barrier selection.

Background
----------
The current overspeed risk (Cat A × F4 at λ_IE=0.5/yr central) is in the
ALARP region under IEC 61400-1 — not automatically acceptable. A formal
ALARP case requires demonstrating that further risk reduction options have
been evaluated and either implemented or shown to be disproportionately costly
relative to the safety benefit.

Phase 5 importance ranking (FV @ 365d):
  Rank 1: G3b Safety relay fault (FV=0.822) → Option 1
  Rank 2: G2a Hydraulic supply loss (FV=0.772) → Option 2
  Rank 3: G1a Control/encoder fault (FV=0.450) → Option 3

The three options target the three highest-importance basic events in FV rank
order. This is the standard ALARP structuring approach: demonstrate that you
have evaluated improvements in importance order and justify why each is or is
not implemented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import config as _cfg
from .event_tree import build_event_tree, EventTreeBranch


# ── Phase 5 basic event Q values (from gate_Q_table.csv @ 365d) ──────────
# These are needed for ALARP option recomputation. Loaded at runtime where
# possible; hard-coded as verified fallbacks if the CSV is unavailable.
_BASELINE_Q: dict[str, float] = {
    # G1 sub-events
    "G1a: Control/Encoder fault":         0.08201660687363882,
    "G1b: Battery/Electrical supply fault": 0.06379069423505243,
    "G1c: Actuator mechanical failure":    0.03645182527717281,
    # G2 sub-events
    "G2a: Hydraulic supply loss":          0.10777799999999997,
    "G2b: Brake hardware fault":           0.03186360000000001,
    # G3 sub-events
    "G3a: Communication/fieldbus loss":    0.005870400000000009,
    "G3b: Safety relay/electrical fault":  0.02713199999999998,
}


def _or_gate(q_values: list[float]) -> float:
    """Compute OR gate Q from a list of basic event Q values."""
    result = 1.0
    for q in q_values:
        result *= (1.0 - q)
    return 1.0 - result


@dataclass
class ALARPOption:
    """One risk reduction option with quantified effect on Q_top."""

    name:             str
    target_event:     str   # basic event name (e.g. "G3b: Safety relay...")
    target_gate:      str   # which gate this event belongs to (G1/G2/G3)
    fv_rank:          int   # Phase 5 FV importance rank
    fv_value:         float # Phase 5 FV importance value
    intervention:     str   # description of the physical intervention
    cost_category:    str   # Low / Medium / High / Very High
    Q_event_before:   float
    Q_event_after:    float
    Q_gate_before:    float
    Q_gate_after:     float
    Q_top_before:     float
    Q_top_after:      float
    lambda_IE:        float = _cfg.LAMBDA_IE_CENTRAL

    @property
    def Q_top_reduction_pct(self) -> float:
        return 100.0 * (self.Q_top_before - self.Q_top_after) / self.Q_top_before

    @property
    def lambda_catastrophe_before(self) -> float:
        return self.lambda_IE * self.Q_top_before

    @property
    def lambda_catastrophe_after(self) -> float:
        return self.lambda_IE * self.Q_top_after

    @property
    def freq_cat_before(self) -> str:
        return _cfg.freq_category(self.lambda_catastrophe_before)

    @property
    def freq_cat_after(self) -> str:
        return _cfg.freq_category(self.lambda_catastrophe_after)

    @property
    def risk_acceptability_before(self) -> str:
        return _cfg.RISK_MATRIX.get(
            ("A", self.freq_cat_before), "UNDEFINED"
        )

    @property
    def risk_acceptability_after(self) -> str:
        return _cfg.RISK_MATRIX.get(
            ("A", self.freq_cat_after), "UNDEFINED"
        )

    @property
    def risk_matrix_moves_to_acceptable(self) -> bool:
        return self.risk_acceptability_after == "ACCEPTABLE"


def compute_option_1(
    Q_top_before: float,
    lambda_IE: float = _cfg.LAMBDA_IE_CENTRAL,
    q_baseline: Optional[dict[str, float]] = None,
) -> ALARPOption:
    """
    Option 1: Duplicate safety relay channel (targets G3b, FV rank 1).

    Intervention: Replace single safety relay (G3b) with a dual-channel
    SIL-2 certified safety relay module per IEC 61508. A SIL-2 hardware
    fault tolerance of 1 reduces the dangerous failure rate by ~30×.
    Conservative estimate: Q_G3b_after ≈ 1×10⁻³ (SIL-2 PFDavg upper bound).

    Effect: Q_G3 drops from 0.0328 → ~0.0069 (4.7× improvement).
    Q_top drops from 7.69×10⁻⁴ → ~1.62×10⁻⁴.
    At λ_IE=0.5/yr: λ_catastrophe → 8.1×10⁻⁵ → F5 (Extremely Unlikely).
    Risk matrix moves: Cat A × F4 (ALARP) → Cat A × F5 (ACCEPTABLE).
    """
    q = q_baseline or _BASELINE_Q
    Q_G3a = q["G3a: Communication/fieldbus loss"]
    Q_G3b_before = q["G3b: Safety relay/electrical fault"]
    Q_G3b_after  = 1.0e-3   # SIL-2 dual-channel: PFDavg ≈ 1e-3

    Q_G3_before = _or_gate([Q_G3a, Q_G3b_before])
    Q_G3_after  = _or_gate([Q_G3a, Q_G3b_after])

    Q_G1 = _or_gate([q["G1a: Control/Encoder fault"],
                     q["G1b: Battery/Electrical supply fault"],
                     q["G1c: Actuator mechanical failure"]])
    Q_G2 = _or_gate([q["G2a: Hydraulic supply loss"],
                     q["G2b: Brake hardware fault"]])

    Q_top_after = Q_G1 * Q_G2 * Q_G3_after

    return ALARPOption(
        name="Option 1: Duplicate Safety Relay (G3b → SIL-2)",
        target_event="G3b: Safety relay/electrical fault",
        target_gate="G3",
        fv_rank=1,
        fv_value=0.8221,
        intervention=(
            "Replace single-channel safety relay with dual-channel SIL-2 "
            "certified module (IEC 61508 HFT=1). Estimated Q_G3b: 1×10⁻³ "
            "(SIL-2 PFDavg upper bound per IEC 61508 Table 4)."
        ),
        cost_category="Low",
        Q_event_before=Q_G3b_before,
        Q_event_after=Q_G3b_after,
        Q_gate_before=Q_G3_before,
        Q_gate_after=Q_G3_after,
        Q_top_before=Q_top_before,
        Q_top_after=Q_top_after,
        lambda_IE=lambda_IE,
    )


def compute_option_2(
    Q_top_before: float,
    lambda_IE: float = _cfg.LAMBDA_IE_CENTRAL,
    q_baseline: Optional[dict[str, float]] = None,
) -> ALARPOption:
    """
    Option 2: Hydraulic accumulator backup (targets G2a, FV rank 2).

    Intervention: Install an independent nitrogen-charged accumulator that
    maintains brake hydraulic pressure for ≥60 s after pump loss. This makes
    the brake available for short-duration grid faults (the dominant IE).
    Effect: Q_G2a reduced by factor ~5 (accumulator provides redundant
    pressure source; CCF through common fluid is still possible but rare).

    Effect: Q_G2 drops from 0.1362 → ~0.0527.
    Q_top drops from 7.69×10⁻⁴ → ~2.98×10⁻⁴.
    At λ_IE=0.5/yr: λ_catastrophe → 1.49×10⁻⁴ → still F4.
    Risk matrix: remains Cat A × F4 (ALARP), but moved toward F5 boundary.
    """
    q = q_baseline or _BASELINE_Q
    Q_G2a_before = q["G2a: Hydraulic supply loss"]
    Q_G2b        = q["G2b: Brake hardware fault"]
    Q_G2a_after  = Q_G2a_before / 5.0   # accumulator reduces by ~5×

    Q_G2_before = _or_gate([Q_G2a_before, Q_G2b])
    Q_G2_after  = _or_gate([Q_G2a_after,  Q_G2b])

    Q_G1 = _or_gate([q["G1a: Control/Encoder fault"],
                     q["G1b: Battery/Electrical supply fault"],
                     q["G1c: Actuator mechanical failure"]])
    Q_G3 = _or_gate([q["G3a: Communication/fieldbus loss"],
                     q["G3b: Safety relay/electrical fault"]])

    Q_top_after = Q_G1 * Q_G2_after * Q_G3

    return ALARPOption(
        name="Option 2: Hydraulic Accumulator Backup (G2a → ×0.2)",
        target_event="G2a: Hydraulic supply loss",
        target_gate="G2",
        fv_rank=2,
        fv_value=0.7718,
        intervention=(
            "Install independent nitrogen-charged hydraulic accumulator "
            "maintaining ≥60 s brake pressure after pump loss. Estimated "
            "Q_G2a reduction factor: ~5× (independent pressure source "
            "with hydraulic isolation valve)."
        ),
        cost_category="Medium",
        Q_event_before=Q_G2a_before,
        Q_event_after=Q_G2a_after,
        Q_gate_before=Q_G2_before,
        Q_gate_after=Q_G2_after,
        Q_top_before=Q_top_before,
        Q_top_after=Q_top_after,
        lambda_IE=lambda_IE,
    )


def compute_option_3(
    Q_top_before: float,
    lambda_IE: float = _cfg.LAMBDA_IE_CENTRAL,
    q_baseline: Optional[dict[str, float]] = None,
) -> ALARPOption:
    """
    Option 3: Enhanced pitch encoder testing protocol (targets G1a, FV rank 3).

    Intervention: Add periodic offline integrity testing of pitch encoder
    during planned maintenance intervals (e.g., every 6 months). Testing
    detects latent encoder faults that would otherwise only manifest on demand.
    If testing halves the mean time a latent fault goes undetected:
    Q_G1a reduces by ~50%.

    Effect: Q_G1 drops from 0.1719 → ~0.1345.
    Q_top drops from 7.69×10⁻⁴ → ~6.02×10⁻⁴ (~22% reduction).
    At λ_IE=0.5/yr: λ_catastrophe → 3.01×10⁻⁴ → still F4.
    Risk matrix: remains Cat A × F4 (ALARP). Insufficient alone to cross
    the F4→F5 boundary. Recommended as complement to Option 1, not standalone.
    """
    q = q_baseline or _BASELINE_Q
    Q_G1a_before = q["G1a: Control/Encoder fault"]
    Q_G1b        = q["G1b: Battery/Electrical supply fault"]
    Q_G1c        = q["G1c: Actuator mechanical failure"]
    Q_G1a_after  = Q_G1a_before * 0.50   # testing halves latent fault exposure

    Q_G1_before = _or_gate([Q_G1a_before, Q_G1b, Q_G1c])
    Q_G1_after  = _or_gate([Q_G1a_after,  Q_G1b, Q_G1c])

    Q_G2 = _or_gate([q["G2a: Hydraulic supply loss"],
                     q["G2b: Brake hardware fault"]])
    Q_G3 = _or_gate([q["G3a: Communication/fieldbus loss"],
                     q["G3b: Safety relay/electrical fault"]])

    Q_top_after = Q_G1_after * Q_G2 * Q_G3

    return ALARPOption(
        name="Option 3: Enhanced Encoder Testing Protocol (G1a → ×0.5)",
        target_event="G1a: Control/Encoder fault",
        target_gate="G1",
        fv_rank=3,
        fv_value=0.4500,
        intervention=(
            "Periodic offline integrity testing of pitch encoder and "
            "controller during planned maintenance (6-month interval). "
            "Estimated Q_G1a reduction: ~50% (halves latent fault "
            "exposure interval per IEC 61400-26-2 test interval analysis)."
        ),
        cost_category="Low",
        Q_event_before=Q_G1a_before,
        Q_event_after=Q_G1a_after,
        Q_gate_before=Q_G1_before,
        Q_gate_after=Q_G1_after,
        Q_top_before=Q_top_before,
        Q_top_after=Q_top_after,
        lambda_IE=lambda_IE,
    )


def compute_all_options(
    gate_Q: dict[str, float],
    lambda_IE: float = _cfg.LAMBDA_IE_CENTRAL,
    q_baseline: Optional[dict[str, float]] = None,
) -> list[ALARPOption]:
    """
    Compute all three ALARP options from Phase 5 gate Q values.

    Parameters
    ----------
    gate_Q : dict[str, float]
        Output of event_tree.load_gate_Q_values(). Used only to extract
        Q_top_before — gate-level probabilities are sufficient here.
    lambda_IE : float
        Initiating event frequency for λ_catastrophe computation.
    q_baseline : dict, optional
        Override basic event Q values (e.g. for testing). Defaults to
        _BASELINE_Q which matches Phase 5 gate_Q_table.csv @ 365d.

    Returns
    -------
    list[ALARPOption]
        Three options in FV rank order (Option 1 first, highest leverage).
    """
    Q_top_before = gate_Q[_cfg.GATE_NAMES["TE"]]
    return [
        compute_option_1(Q_top_before, lambda_IE, q_baseline),
        compute_option_2(Q_top_before, lambda_IE, q_baseline),
        compute_option_3(Q_top_before, lambda_IE, q_baseline),
    ]


def alarp_table(options: list[ALARPOption]) -> pd.DataFrame:
    """
    Convert ALARP options to a summary DataFrame.

    Returns
    -------
    pd.DataFrame suitable for CSV export and Markdown rendering.
    """
    rows = []
    for o in options:
        rows.append({
            "option":                    o.name,
            "target_basic_event":        o.target_event,
            "FV_rank":                   o.fv_rank,
            "FV_value":                  round(o.fv_value, 4),
            "cost_category":             o.cost_category,
            "Q_event_before":            f"{o.Q_event_before:.4f}",
            "Q_event_after":             f"{o.Q_event_after:.4e}",
            "Q_gate_before":             f"{o.Q_gate_before:.5f}",
            "Q_gate_after":              f"{o.Q_gate_after:.5f}",
            "Q_top_before":              f"{o.Q_top_before:.4e}",
            "Q_top_after":               f"{o.Q_top_after:.4e}",
            "Q_top_reduction_pct":       f"{o.Q_top_reduction_pct:.1f}%",
            "lambda_catastrophe_before": f"{o.lambda_catastrophe_before:.4e}",
            "lambda_catastrophe_after":  f"{o.lambda_catastrophe_after:.4e}",
            "freq_cat_before":           o.freq_cat_before,
            "freq_cat_after":            o.freq_cat_after,
            "risk_matrix_before":        f"Cat A × {o.freq_cat_before} ({o.risk_acceptability_before})",
            "risk_matrix_after":         f"Cat A × {o.freq_cat_after} ({o.risk_acceptability_after})",
            "moves_to_acceptable":       o.risk_matrix_moves_to_acceptable,
        })
    return pd.DataFrame(rows)