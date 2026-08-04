"""
aeolus_rams_phase6.event_tree
================================
Encodes the turbine overspeed event tree and computes branch probabilities,
conditional frequencies, and farm-level aggregate risks.

Conceptual role
---------------
Phase 5 FTA answered: "Given a demand event occurs, what is P(protection fails)?"
This is a conditional probability — independent of how often demands occur.

Phase 6 ETA answers: "How often does an actual catastrophe occur?"
This requires an absolute frequency: λ_catastrophe = λ_IE × Q_top.

Neither tool alone produces this. The ETA uses the FTA's gate Q values as
branch probabilities, so the two tools are genuinely complementary — the
bow-tie joins them around the central hazard.

Branch structure
----------------
  IE occurs  →  G1 activates?  →  G2 activates?  →  G3 activates?  →  Outcome
  P(IE/yr)      P(G1 ok)=1-Q_G1  P(G2 ok)=1-Q_G2  P(G3 ok)=1-Q_G3

Four mutually exclusive, exhaustive branches:
  Branch 1: P_G1_ok                        = 1-Q_G1              (SAFE controlled)
  Branch 2: Q_G1 × P_G2_ok                 = Q_G1×(1-Q_G2)       (SAFE brake)
  Branch 3: Q_G1 × Q_G2 × P_G3_ok          = Q_G1×Q_G2×(1-Q_G3) (NEAR-MISS)
  Branch 4: Q_G1 × Q_G2 × Q_G3             = Q_top               (CATASTROPHIC)

Sum = 1 is verified by assertion in build_event_tree().
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import config as _cfg


# ── Data loading ─────────────────────────────────────────────────────────────

def load_gate_Q_values(
    gate_Q_path: Path,
    t_filter: str = "365d",
) -> dict[str, float]:
    """
    Load gate Q values from Phase 5 gate_Q_table.csv.

    Parameters
    ----------
    gate_Q_path : Path
        Path to gate_Q_table.csv produced by Phase 5 pipeline.
    t_filter : str
        Mission time label to select (e.g. "365d" or "1825d").

    Returns
    -------
    dict mapping stripped node name → Q(t) for Gate nodes only.

    Notes
    -----
    The Phase 5 CSV uses indented node names (leading whitespace encodes tree
    depth). This function strips all leading/trailing whitespace before keying.
    Only 'Gate' node_type rows are returned — BasicEvents are excluded.
    """
    df = pd.read_csv(gate_Q_path)
    df = df[df["t"] == t_filter].copy()
    df = df[df["node_type"] == "Gate"].copy()
    df["node_clean"] = df["node"].str.strip()
    return dict(zip(df["node_clean"], df["Q"]))


def load_basic_event_Q_values(
    gate_Q_path: Path,
    t_filter: str = "365d",
) -> dict[str, dict]:
    """
    Load BasicEvent rows from Phase 5 gate_Q_table.csv.

    Returns dict mapping stripped node name → {Q, confidence, parent_component}
    for use in the bow-tie diagram's left-side threat labels.
    """
    df = pd.read_csv(gate_Q_path)
    df = df[df["t"] == t_filter].copy()
    df = df[df["node_type"] == "BasicEvent"].copy()
    df["node_clean"] = df["node"].str.strip()
    result = {}
    for _, row in df.iterrows():
        result[row["node_clean"]] = {
            "Q":                 float(row["Q"]),
            "confidence":        str(row["confidence"]),
            "parent_component":  str(row["parent_component"]),
        }
    return result


def load_ccf_uplift(
    ccf_path: Path,
    beta: float = _cfg.CCF_BETA_CENTRAL,
) -> float:
    """
    Load CCF risk uplift factor from Phase 5 ccf_sensitivity.csv at given beta.

    Falls back to config.CCF_UPLIFT_CENTRAL if the file is unavailable or the
    beta value is not found.
    """
    try:
        df = pd.read_csv(ccf_path)
        mask = np.isclose(df["beta_G1G2"], beta, atol=1e-6)
        if mask.sum() == 1:
            return float(df.loc[mask, "ccf_risk_uplift"].iloc[0])
    except Exception:
        pass
    return _cfg.CCF_UPLIFT_CENTRAL


# ── Branch dataclass ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EventTreeBranch:
    """One outcome branch in the four-branch event tree."""

    label:                 str     # e.g. "Branch_4_CATASTROPHIC"
    description:           str
    severity_category:     str     # IEC 61400-1: A / B / C / D / E
    P_branch_given_IE:     float   # conditional probability of this outcome
    lambda_IE:             float   # initiating event frequency /turbine/yr
    ccf_adjusted:          bool    # whether Q_top carries CCF uplift

    # ── Derived quantities ──────────────────────────────────────────────────

    @property
    def lambda_outcome_per_turbine(self) -> float:
        """Absolute outcome frequency per turbine per year."""
        return self.lambda_IE * self.P_branch_given_IE

    @property
    def lambda_outcome_per_farm(self) -> float:
        """Absolute outcome frequency per farm per year (N turbines, no CCF
        correlation between turbines assumed — conservative for independence)."""
        return self.lambda_outcome_per_turbine * _cfg.N_TURBINES

    @property
    def return_period_turbine_years(self) -> float:
        """Mean return period per turbine in years."""
        lam = self.lambda_outcome_per_turbine
        return float("inf") if lam == 0.0 else 1.0 / lam

    @property
    def P_one_or_more_in_design_life(self) -> float:
        """P(≥1 event anywhere on the farm in design life) — Poisson model."""
        lam_farm_life = self.lambda_outcome_per_farm * _cfg.DESIGN_LIFE
        return 1.0 - np.exp(-lam_farm_life)

    def frequency_category(self) -> str:
        """IEC 61400-1 Ed.4 frequency category label (F1–F5)."""
        return _cfg.freq_category(self.lambda_outcome_per_turbine)

    def risk_acceptability(self) -> str:
        """IEC 61400-1 Ed.4 risk matrix cell for this branch."""
        key = (self.severity_category, self.frequency_category())
        return _cfg.RISK_MATRIX.get(key, "UNDEFINED")


# ── Tree builder ─────────────────────────────────────────────────────────────

_DESCRIPTIONS: dict[str, str] = {
    "Branch_1_SAFE_controlled":
        "G1 (pitch feathering) successfully activates. Controlled normal "
        "shutdown. No structural loading. Production interruption only.",
    "Branch_2_SAFE_brake":
        "Pitch feathering fails; mechanical brake engages and arrests rotor "
        "before structural limits are reached. Minor brake wear; inspection "
        "recommended before return to service.",
    "Branch_3_NEAR_MISS_SCADA":
        "Pitch and brake both fail; SCADA overspeed trip activates. Rotor "
        "briefly exceeds rated speed (~105–115%). Structural fatigue "
        "inspection required; possible blade/bearing fatigue damage.",
    "Branch_4_CATASTROPHIC":
        "All three protection layers fail simultaneously. Rotor exceeds "
        "≥120% rated speed. Centrifugal forces exceed blade root structural "
        "limits — blade ejection and/or tower collapse. Total asset loss. "
        "Safety-critical event with potential personnel/public fatality.",
}


def build_event_tree(
    lambda_IE: float,
    Q_G1: float,
    Q_G2: float,
    Q_G3: float,
    ccf_adjusted: bool = False,
    ccf_uplift: float = _cfg.CCF_UPLIFT_CENTRAL,
) -> list[EventTreeBranch]:
    """
    Build the four-branch turbine overspeed event tree.

    Parameters
    ----------
    lambda_IE : float
        Initiating event frequency (events per turbine per year).
    Q_G1, Q_G2, Q_G3 : float
        Protection layer failure probabilities from Phase 5 gate_Q_table.csv.
    ccf_adjusted : bool
        If True, Branch 4 probability is multiplied by ccf_uplift, and the
        excess probability is subtracted from Branch 3 (conservative
        redistribution: the near-miss branch is most credibly affected).
    ccf_uplift : float
        CCF risk uplift factor loaded from Phase 5 ccf_sensitivity.csv.

    Returns
    -------
    list[EventTreeBranch]
        Four branches in order: Branch 1 (most likely) → Branch 4 (least).

    Raises
    ------
    AssertionError
        If branch probabilities do not sum to 1.0 within 1e-9 tolerance.
    """
    P_G1_ok = 1.0 - Q_G1
    P_G2_ok = 1.0 - Q_G2
    P_G3_ok = 1.0 - Q_G3

    P_branch: dict[str, float] = {
        "Branch_1_SAFE_controlled":  P_G1_ok,
        "Branch_2_SAFE_brake":       Q_G1 * P_G2_ok,
        "Branch_3_NEAR_MISS_SCADA":  Q_G1 * Q_G2 * P_G3_ok,
        "Branch_4_CATASTROPHIC":     Q_G1 * Q_G2 * Q_G3,
    }

    if ccf_adjusted:
        Q_top_base = Q_G1 * Q_G2 * Q_G3
        Q_top_ccf  = Q_top_base * ccf_uplift
        delta = Q_top_ccf - Q_top_base
        # Redistribute excess from Branch 3 (nearest plausible source)
        P_branch["Branch_4_CATASTROPHIC"]    = Q_top_ccf
        P_branch["Branch_3_NEAR_MISS_SCADA"] = max(
            0.0, P_branch["Branch_3_NEAR_MISS_SCADA"] - delta
        )

    total = sum(P_branch.values())
    assert abs(total - 1.0) < 1e-9, (
        f"Branch probabilities must sum to 1.0, got {total:.12f}. "
        f"Check Q values: Q_G1={Q_G1}, Q_G2={Q_G2}, Q_G3={Q_G3}, "
        f"ccf_adjusted={ccf_adjusted}, ccf_uplift={ccf_uplift}."
    )

    return [
        EventTreeBranch(
            label=label,
            description=_DESCRIPTIONS[label],
            severity_category=_cfg.BRANCH_SEVERITY[label],
            P_branch_given_IE=P_branch[label],
            lambda_IE=lambda_IE,
            ccf_adjusted=ccf_adjusted,
        )
        for label in P_branch
    ]


# ── Consequence frequency table ──────────────────────────────────────────────

def consequence_frequency_table(
    gate_Q: dict[str, float],
    lambda_IE_values: Optional[list[float]] = None,
    include_ccf: bool = True,
    ccf_uplift: float = _cfg.CCF_UPLIFT_CENTRAL,
) -> pd.DataFrame:
    """
    Build a comprehensive consequence frequency table sweeping over λ_IE values
    and (optionally) CCF-adjusted versus independent scenarios.

    Parameters
    ----------
    gate_Q : dict[str, float]
        Gate Q values from load_gate_Q_values(). Must contain keys for G1, G2, G3.
    lambda_IE_values : list[float], optional
        Initiating event frequencies to sweep. Defaults to config.LAMBDA_IE_RANGE.
    include_ccf : bool
        If True, include both independent and CCF-adjusted rows.
    ccf_uplift : float
        CCF uplift factor from Phase 5 ccf_sensitivity.csv.

    Returns
    -------
    pd.DataFrame
        Columns: lambda_IE, branch_label, severity_category, P_branch,
                 lambda_per_turbine, lambda_per_farm, return_period_yr,
                 P_one_plus_25yr_farm, freq_category, risk_acceptability,
                 ccf_adjusted.
    """
    if lambda_IE_values is None:
        lambda_IE_values = list(_cfg.LAMBDA_IE_RANGE)

    Q_G1 = gate_Q[_cfg.GATE_NAMES["G1"]]
    Q_G2 = gate_Q[_cfg.GATE_NAMES["G2"]]
    Q_G3 = gate_Q[_cfg.GATE_NAMES["G3"]]

    rows: list[dict] = []
    for lam_IE in lambda_IE_values:
        for ccf in ([False, True] if include_ccf else [False]):
            branches = build_event_tree(
                lam_IE, Q_G1, Q_G2, Q_G3,
                ccf_adjusted=ccf, ccf_uplift=ccf_uplift,
            )
            for b in branches:
                rows.append({
                    "lambda_IE":              lam_IE,
                    "branch_label":           b.label,
                    "severity_category":      b.severity_category,
                    "P_branch":               b.P_branch_given_IE,
                    "lambda_per_turbine":     b.lambda_outcome_per_turbine,
                    "lambda_per_farm":        b.lambda_outcome_per_farm,
                    "return_period_yr":       b.return_period_turbine_years,
                    "P_one_plus_25yr_farm":   b.P_one_or_more_in_design_life,
                    "freq_category":          b.frequency_category(),
                    "risk_acceptability":     b.risk_acceptability(),
                    "ccf_adjusted":           ccf,
                })

    return pd.DataFrame(rows)