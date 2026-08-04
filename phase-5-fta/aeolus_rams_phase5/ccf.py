"""
aeolus_rams_phase5.ccf
========================
Common-Cause Failure (CCF) analysis using the β-factor method.

The β-factor model (IEC 61508-6 Annex D, IEC 61511-3 Appendix F):

    Q_i_independent = Q_i × (1 − β)          — failures unique to system i
    Q_CCF           = β × Q_shared_cause      — failures causing both i and j to fail

Top event with G1/G2 CCF:
    Q_top_ccf = Q_G1_ind × Q_G2_ind × Q_G3    (all three fail independently)
              + Q_CCF_G1G2 × Q_G3             (G1 and G2 fail together; G3 also fails)

    where: Q_CCF_G1G2 = β × Q_hydraulic
           (hydraulic system failure simultaneously disables both pitch and brake)

Physical justification:
    Phase 1 FMECA event: "Rotorbrake and Hydraulic problemes — Hydraulic pump A disabled"
    This single root cause disabled both G1 (pitch actuators use hydraulic pressure to
    maintain blade pitch against aerodynamic loads) and G2 (brake caliper is hydraulically
    actuated). This is a confirmed CCF pathway, not a hypothetical one.

Expected result:
    CCF uplift factor (Q_top_ccf / Q_top_independent) typically 1.2–2.0 for β=0.10.
    This finding should be highlighted in the Phase 5 report as a genuine engineering
    insight beyond standard FTA.

Reference:
    IEC 61508-6:2010, Annex D — β-factor common cause failure method.
    Smith, D.J. (2011) "Reliability, Maintainability and Risk." 8th ed. Elsevier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fault_tree import Gate, BasicEvent


def apply_beta_factor_ccf(
    top_gate: Gate,
    Q_1yr: dict[str, float],
    beta_G1G2: float,
) -> dict[str, float]:
    """Compute Q_top with β-factor CCF adjustment for the G1/G2 shared hydraulic pathway.

    Parameters
    ----------
    top_gate : Gate
        The fully-built overspeed fault tree (from build_overspeed_fault_tree()).
    Q_1yr : dict[str, float]
        Component Q values (includes 'Hydraulic System' for CCF source).
    beta_G1G2 : float
        β-factor for G1/G2 CCF via shared hydraulic system. Range [0, 0.20].
        β=0.0 → no CCF (pure independence assumed).
        β=0.10 → 10% of G1+G2 combined failures are common-cause events.
        β=0.20 → conservative upper bound per IEC 61508 Annex D Table D.4.

    Returns
    -------
    dict with keys:
        Q_top_independent    : float — Q_top under independence (= top_gate.Q)
        Q_top_ccf_adjusted   : float — Q_top with CCF adjustment
        Q_CCF_G1G2           : float — common-cause failure probability
        ccf_risk_uplift      : float — Q_top_ccf / Q_top_independent
        beta_G1G2            : float — the β used
        Q_hydraulic          : float — source component Q (CCF driver)
        return_period_independent : float — 1 / Q_top_independent
        return_period_ccf    : float — 1 / Q_top_ccf_adjusted
    """
    Q_top_independent = top_gate.Q
    Q_hydraulic = Q_1yr["Hydraulic System"]

    # Retrieve G1 and G2 gate Q values from the tree
    gates = top_gate.collect_gates()
    Q_G1 = gates["G1: Pitch System fails to feather"].Q
    Q_G2 = gates["G2: Mechanical brake fails to engage"].Q
    Q_G3 = gates["G3: SCADA overspeed trip fails to activate"].Q

    # Independent portions (β-factor reduces each gate's independent contribution)
    Q_G1_ind = Q_G1 * (1.0 - beta_G1G2)
    Q_G2_ind = Q_G2 * (1.0 - beta_G1G2)

    # CCF contribution: hydraulic system failure simultaneously disables G1 and G2
    Q_CCF_G1G2 = beta_G1G2 * Q_hydraulic

    # Top event with CCF:
    # P(top) = P(G1_ind ∩ G2_ind ∩ G3) + P(CCF_G1G2 ∩ G3)
    #        = Q_G1_ind × Q_G2_ind × Q_G3 + Q_CCF_G1G2 × Q_G3
    Q_top_ccf = Q_G1_ind * Q_G2_ind * Q_G3 + Q_CCF_G1G2 * Q_G3

    uplift = Q_top_ccf / Q_top_independent if Q_top_independent > 0 else float("inf")

    return {
        "Q_top_independent":           Q_top_independent,
        "Q_top_ccf_adjusted":          Q_top_ccf,
        "Q_CCF_G1G2":                  Q_CCF_G1G2,
        "ccf_risk_uplift":             uplift,
        "beta_G1G2":                   beta_G1G2,
        "Q_G1_independent_portion":    Q_G1_ind,
        "Q_G2_independent_portion":    Q_G2_ind,
        "Q_G3":                        Q_G3,
        "Q_hydraulic":                 Q_hydraulic,
        "return_period_independent":   1.0 / Q_top_independent if Q_top_independent > 0 else float("inf"),
        "return_period_ccf":           1.0 / Q_top_ccf if Q_top_ccf > 0 else float("inf"),
    }


def ccf_sensitivity_sweep(
    top_gate: Gate,
    Q_1yr: dict[str, float],
    beta_range: np.ndarray | None = None,
) -> pd.DataFrame:
    """Sweep β_G1G2 over a range and record Q_top_ccf at each point.

    Parameters
    ----------
    top_gate : Gate
        Fully-built fault tree.
    Q_1yr : dict[str, float]
        Component Q values.
    beta_range : np.ndarray, optional
        β values to evaluate. Defaults to config.CCF_BETA_RANGE (0–0.20, 21 points).

    Returns
    -------
    pd.DataFrame
        Columns: beta_G1G2, Q_top_independent, Q_top_ccf_adjusted,
                 Q_CCF_G1G2, ccf_risk_uplift, return_period_ccf.
    """
    from . import config as _cfg

    if beta_range is None:
        beta_range = _cfg.CCF_BETA_RANGE

    rows = []
    for beta in beta_range:
        result = apply_beta_factor_ccf(top_gate, Q_1yr, float(beta))
        rows.append({
            "beta_G1G2":            result["beta_G1G2"],
            "Q_top_independent":    result["Q_top_independent"],
            "Q_top_ccf_adjusted":   result["Q_top_ccf_adjusted"],
            "Q_CCF_G1G2":           result["Q_CCF_G1G2"],
            "ccf_risk_uplift":      result["ccf_risk_uplift"],
            "return_period_ccf":    result["return_period_ccf"],
        })

    return pd.DataFrame(rows)


def subcause_fraction_sensitivity(
    Q_1yr: dict[str, float],
    g1_fraction_variants: list[tuple[float, float, float]],
    g2_fraction_variants: list[tuple[float, float]],
    g3_fraction_variants: list[tuple[float, float]],
) -> pd.DataFrame:
    """Sweep sub-cause fractions to confirm Q_top is insensitive to them.

    As Section 5.4 notes, sub-cause fractions are engineering estimates.
    This function runs the fault tree across ±50% variation on each set
    of fractions and reports the resulting Q_top range.

    Parameters
    ----------
    Q_1yr : dict[str, float]
        Component Q values.
    g1_fraction_variants : list of (f_G1a, f_G1b, f_G1c) tuples (summing to ~1.0)
    g2_fraction_variants : list of (f_G2a, f_G2b) tuples
    g3_fraction_variants : list of (f_G3a, f_G3b) tuples

    Returns
    -------
    pd.DataFrame
        One row per (G1, G2, G3) fraction combination tested.
        Columns: g1_fracs, g2_fracs, g3_fracs, Q_top, delta_from_baseline_pct.
    """
    from .fault_tree import build_overspeed_fault_tree

    # Baseline Q_top
    baseline_tree = build_overspeed_fault_tree(Q_1yr)
    Q_top_baseline = baseline_tree.Q

    rows = []
    for g1f in g1_fraction_variants:
        for g2f in g2_fraction_variants:
            for g3f in g3_fraction_variants:
                tree = build_overspeed_fault_tree(
                    Q_1yr,
                    sub_cause_fractions={
                        "G1": list(g1f),
                        "G2": list(g2f),
                        "G3": list(g3f),
                    },
                )
                Q_top = tree.Q
                rows.append({
                    "g1_fracs": str(g1f),
                    "g2_fracs": str(g2f),
                    "g3_fracs": str(g3f),
                    "Q_top": Q_top,
                    "delta_from_baseline_pct": 100.0 * (Q_top - Q_top_baseline) / Q_top_baseline,
                })

    return pd.DataFrame(rows)