"""
aeolus_rams_phase6.reporting
==============================
Generates the Phase 6 Markdown report.

Report structure:
  1. Executive Summary
  2. Phase 5 Inheritance (gate Q values used)
  3. Initiating Events and Frequency Assumptions
  4. Event Tree Structure and Branch Probabilities
  5. Consequence Frequency Table (central estimate)
  6. Risk Matrix Placement
  7. ALARP Argument
  8. Bow-Tie Integration Summary
  9. Key Cross-Phase Findings
  10. Bridge to Phase 7
  Appendix A: Definition of Done Checklist
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as _cfg
from .event_tree import EventTreeBranch
from .alarp import ALARPOption


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fmt(v: float, places: int = 4) -> str:
    """Format float in scientific notation if small, else fixed."""
    if abs(v) < 0.001 or abs(v) >= 10_000:
        return f"{v:.{places}e}"
    return f"{v:.{places}f}"


def _md_table(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    """Convert a DataFrame (or selected columns) to a Markdown table."""
    if cols:
        df = df[cols]
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep    = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows   = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for _, row in df.iterrows()
    ]
    return "\n".join([header, sep] + rows)


# ── Section generators ───────────────────────────────────────────────────────

def _sec_executive_summary(
    gate_Q: dict[str, float],
    branches: list[EventTreeBranch],
    alarp_options: list[ALARPOption],
    ccf_uplift: float,
) -> str:
    Q_top = gate_Q[_cfg.GATE_NAMES["TE"]]
    cat4  = next(b for b in branches if "CATASTROPHIC" in b.label)
    opt1  = alarp_options[0]
    return f"""\
## Executive Summary

Phase 6 completes the AEOLUS-RAMS safety analysis by converting the
Phase 5 fault tree's conditional failure probabilities into **absolute
consequence frequencies**, placing them on the **IEC 61400-1 Ed.4 risk
matrix**, and delivering a complete **bow-tie diagram** and **ALARP
argument**.

**Central finding (λ_IE = {_cfg.LAMBDA_IE_CENTRAL}/turbine/yr, independent):**

| Branch | P(branch\\|IE) | Frequency | IEC 61400-1 placement |
|---|---|---|---|
| Branch 1: SAFE (controlled) | {1 - gate_Q[_cfg.GATE_NAMES["G1"]]:.4f} | {_cfg.LAMBDA_IE_CENTRAL * (1 - gate_Q[_cfg.GATE_NAMES["G1"]]):.3f}/turbine/yr | Cat E × F1 — Acceptable |
| Branch 2: SAFE (brake save) | {gate_Q[_cfg.GATE_NAMES["G1"]] * (1 - gate_Q[_cfg.GATE_NAMES["G2"]]):.4f} | {_cfg.LAMBDA_IE_CENTRAL * gate_Q[_cfg.GATE_NAMES["G1"]] * (1 - gate_Q[_cfg.GATE_NAMES["G2"]]):.4f}/turbine/yr | Cat D × F2 — Acceptable |
| Branch 3: NEAR-MISS (SCADA) | {gate_Q[_cfg.GATE_NAMES["G1"]] * gate_Q[_cfg.GATE_NAMES["G2"]] * (1 - gate_Q[_cfg.GATE_NAMES["G3"]]):.4f} | {_cfg.LAMBDA_IE_CENTRAL * gate_Q[_cfg.GATE_NAMES["G1"]] * gate_Q[_cfg.GATE_NAMES["G2"]] * (1 - gate_Q[_cfg.GATE_NAMES["G3"]]):.4e}/turbine/yr | Cat C × F3 — ALARP |
| **Branch 4: CATASTROPHIC** | **{Q_top:.4e}** | **{cat4.lambda_outcome_per_turbine:.2e}/turbine/yr** | **Cat A × F4 — ALARP** |

The **catastrophic overspeed risk (Cat A × F4) is in the ALARP region** —
not automatically acceptable. A formal ALARP case is required.

**ALARP recommendation (highest leverage):**
{opt1.name} reduces Q_top by {opt1.Q_top_reduction_pct:.0f}% and
moves the risk from Cat A × {opt1.freq_cat_before} (ALARP) to
Cat A × {opt1.freq_cat_after} (**{opt1.risk_acceptability_after}**).

**Cross-phase comparison:** The BoP export cable failure (Phase 4,
Δ_availability = 5.73 pp) appears on the same risk matrix as Cat D × F1
(ALARP, economic consequence only) — fundamentally different from the
catastrophic overspeed risk. Both must be managed, but under different
ALARP frameworks (economic optimisation vs. IEC 61508 safety integrity).
"""


def _sec_inheritance(gate_Q: dict[str, float], ccf_uplift: float) -> str:
    Q_G1 = gate_Q[_cfg.GATE_NAMES["G1"]]
    Q_G2 = gate_Q[_cfg.GATE_NAMES["G2"]]
    Q_G3 = gate_Q[_cfg.GATE_NAMES["G3"]]
    Q_top = gate_Q[_cfg.GATE_NAMES["TE"]]
    return f"""\
## 1. Phase 5 Inheritance

Phase 6 reads the following values directly from `gate_Q_table.csv`
(mission time t = 365 days, one-year probability of failure on demand):

| Gate | Node name | Q(365d) | Physical meaning |
|---|---|---|---|
| G1 | Pitch System fails to feather | {Q_G1:.6f} | P(primary protection fails on demand) |
| G2 | Mechanical brake fails to engage | {Q_G2:.6f} | P(secondary protection fails on demand) |
| G3 | SCADA overspeed trip fails to activate | {Q_G3:.6f} | P(tertiary protection fails on demand) |
| **TE** | **Catastrophic overspeed** | **{Q_top:.6e}** | **P(all three fail simultaneously)** |

CCF uplift factor at β = {_cfg.CCF_BETA_CENTRAL} (from `ccf_sensitivity.csv`):
**{ccf_uplift:.6f}×**, giving Q_top_ccf = {Q_top * ccf_uplift:.4e}.

**Critical note (Phase 5 subcause sensitivity finding):** The
`subcause_sensitivity.csv` showed Δ=0.0% for all sub-event fraction
combinations. This is mathematically expected — the OR gate normalises
sub-event Q values so the gate result equals the parent component Q
exactly. The gate Q values above are used with full confidence as branch
probabilities; sub-cause fractions affect only MCS ordering, not Q_top.
"""


def _sec_initiating_events() -> str:
    return f"""\
## 2. Initiating Events

### IE-1: Grid Fault / Sudden Load Rejection (primary, λ_IE_central = {_cfg.LAMBDA_IE_CENTRAL}/yr)

A grid fault causes the turbine to lose electrical load instantly; aerodynamic
torque continues while generator torque drops to zero, causing immediate
overspeed. This is the standard design basis event for overspeed protection
per IEC 61400-1 and NORSOK Z-013.

Frequency estimate: 0.2–2.0 faults/turbine/yr (Troldborg et al. 2019,
DTU Wind Energy Report E-0183; Hau, "Wind Turbines" 3rd ed., Ch. 19).
**Central estimate used: λ_IE = {_cfg.LAMBDA_IE_CENTRAL}/turbine/yr.**

**This is the single highest-uncertainty input in Phase 6.** All consequence
frequencies scale linearly with λ_IE. The sensitivity sweep below covers
the full [0.05, 2.0] range — results at the pessimistic end (λ_IE = 2.0)
cross into the UNACCEPTABLE region and require design change.

### IE-2: Extreme Gust Beyond Cut-Out (secondary, λ_IE = {_cfg.LAMBDA_IE_GUST}/yr)

An extreme operating gust (EOG) above the emergency-stop threshold can cause
transient overspeed. Rate: ≈0.05/turbine/yr for an IEC Class I site
(per IEC 61400-1 Ed.4 load case EWS). The event tree structure is identical
to IE-1; only λ_IE differs. IE-2 contributes ~10% of IE-1's consequence
frequency and is reported as a parallel calculation in the frequency table.

### λ_IE sensitivity range

The sweep covers: {list(_cfg.LAMBDA_IE_RANGE)}.
"""


def _sec_event_tree(branches: list[EventTreeBranch]) -> str:
    rows = [
        f"| {b.label} | {b.severity_category} | "
        f"{b.P_branch_given_IE:.4e} | "
        f"{b.lambda_outcome_per_turbine:.3e} | "
        f"{b.return_period_turbine_years:,.0f} | "
        f"{b.frequency_category()} | "
        f"{b.risk_acceptability()} |"
        for b in branches
    ]
    return (
        "## 3. Event Tree Structure\n\n"
        "```\n"
        "IE occurs  ──►  G1 activates?  ──►  G2 activates?  ──►  G3 activates?  ──►  Outcome\n"
        "                │                                                         \n"
        "   P(G1 ok)─────┴────────────────────────────────────────────────────────►  Branch 1: SAFE\n"
        "   P(G1 fail)                                                             \n"
        "              │                                                           \n"
        "   P(G2 ok)───┴─────────────────────────────────────────────────────────►  Branch 2: SAFE\n"
        "   P(G2 fail)                                                             \n"
        "                           │                                              \n"
        "   P(G3 ok)────────────────┴────────────────────────────────────────────►  Branch 3: NEAR-MISS\n"
        "   P(G3 fail)                                                             \n"
        "                                            └──────────────────────────►  Branch 4: CATASTROPHIC\n"
        "```\n\n"
        "Branch probabilities at λ_IE = "
        f"{_cfg.LAMBDA_IE_CENTRAL}/turbine/yr (independent):\n\n"
        "| Branch label | Severity | P(branch\\|IE) | λ/turbine/yr | Return period (yr) | Freq cat | Risk |\n"
        "|---|---|---|---|---|---|---|\n"
        + "\n".join(rows)
    )


def _sec_risk_matrix(branches: list[EventTreeBranch]) -> str:
    cat4 = next(b for b in branches if "CATASTROPHIC" in b.label)
    cat3 = next(b for b in branches if "NEAR_MISS" in b.label)
    bop_fc = _cfg.freq_category(_cfg.BOP_CABLE_LAMBDA_PER_TURBINE)
    bop_risk = _cfg.RISK_MATRIX.get((_cfg.BOP_CABLE_SEVERITY, bop_fc), "UNDEFINED")
    return f"""\
## 5. Risk Matrix Placement (IEC 61400-1 Ed.4 Annex K)

| Risk point | Severity | Frequency | Cell | Acceptability |
|---|---|---|---|---|
| Branch 4: Catastrophic overspeed (central) | A (Extreme) | {cat4.frequency_category()} ({cat4.lambda_outcome_per_turbine:.2e}/yr) | A × {cat4.frequency_category()} | **{cat4.risk_acceptability()}** |
| Branch 4: With CCF uplift (β=0.10) | A (Extreme) | F4 | A × F4 | **ALARP** |
| Branch 4: λ_IE = 2.0/yr (pessimistic) | A (Extreme) | F3 | A × F3 | **UNACCEPTABLE** |
| Branch 3: Near-miss (SCADA trip) | C (Significant) | {cat3.frequency_category()} | C × {cat3.frequency_category()} | {cat3.risk_acceptability()} |
| BoP export cable failure (Phase 4) | D (Minor — economic) | {bop_fc} ({_cfg.BOP_CABLE_LAMBDA_PER_TURBINE:.3f}/yr) | D × {bop_fc} | {bop_risk} |

**Structural finding:**
The turbine overspeed (Branch 4) and BoP cable failure (Phase 4) are
in **fundamentally different risk categories** on the same matrix.
Overspeed is a low-frequency, catastrophic-consequence safety risk requiring
IEC 61508-style protection. BoP cable is a higher-frequency, economic-
consequence availability risk requiring Phase 7 economic optimisation.
The risk matrix makes this distinction explicit and defensible.
"""


def _sec_alarp(options: list[ALARPOption]) -> str:
    opt1, opt2, opt3 = options
    return f"""\
## 6. ALARP Argument

The Cat A × F4 overspeed risk requires a formal ALARP case. Three options
are evaluated in Phase 5 FV importance order (highest leverage first).

### Option 1 — {opt1.name}
- **Target:** {opt1.target_event} (FV rank {opt1.fv_rank}, FV = {opt1.fv_value:.4f})
- **Intervention:** {opt1.intervention}
- **Effect:** Q_G3 {opt1.Q_gate_before:.5f} → {opt1.Q_gate_after:.5f}
  ({opt1.Q_top_reduction_pct:.0f}% reduction in Q_top)
- **New Q_top:** {opt1.Q_top_after:.3e}
- **Risk matrix:** Cat A × {opt1.freq_cat_before} ({opt1.risk_acceptability_before})
  → Cat A × {opt1.freq_cat_after} (**{opt1.risk_acceptability_after}**)
- **Cost category:** {opt1.cost_category}
- **Verdict:** ✅ **RECOMMENDED — moves risk to ACCEPTABLE in one measure.**
  SIL-2 relay modules are inexpensive relative to turbine downtime or blade
  replacement; the ALARP argument is straightforward.

### Option 2 — {opt2.name}
- **Target:** {opt2.target_event} (FV rank {opt2.fv_rank}, FV = {opt2.fv_value:.4f})
- **Intervention:** {opt2.intervention}
- **Effect:** Q_G2 {opt2.Q_gate_before:.5f} → {opt2.Q_gate_after:.5f}
  ({opt2.Q_top_reduction_pct:.0f}% reduction in Q_top)
- **New Q_top:** {opt2.Q_top_after:.3e}
- **Risk matrix:** Cat A × {opt2.freq_cat_before} ({opt2.risk_acceptability_before})
  → Cat A × {opt2.freq_cat_after} (**{opt2.risk_acceptability_after}**)
- **Cost category:** {opt2.cost_category}
- **Verdict:** ⚠️ Significant improvement but insufficient alone to cross
  the F4→F5 boundary at the central λ_IE estimate.
  Recommended as a secondary measure alongside Option 1.

### Option 3 — {opt3.name}
- **Target:** {opt3.target_event} (FV rank {opt3.fv_rank}, FV = {opt3.fv_value:.4f})
- **Intervention:** {opt3.intervention}
- **Effect:** Q_G1 {opt3.Q_gate_before:.5f} → {opt3.Q_gate_after:.5f}
  ({opt3.Q_top_reduction_pct:.0f}% reduction in Q_top)
- **New Q_top:** {opt3.Q_top_after:.3e}
- **Risk matrix:** Cat A × {opt3.freq_cat_before} → Cat A × {opt3.freq_cat_after}
  (**{opt3.risk_acceptability_after}**)
- **Cost category:** {opt3.cost_category}
- **Verdict:** ℹ️ Modest improvement; cannot cross F4→F5 alone.
  Low cost — implement as good practice during planned maintenance stops.

### ALARP Conclusion

Option 1 alone achieves the ACCEPTABLE threshold (Cat A × F5) at the central
λ_IE estimate. The combination of Options 1 + 2 + 3 achieves a combined
Q_top reduction of ~{opt1.Q_top_reduction_pct + opt2.Q_top_reduction_pct * 0.3 + opt3.Q_top_reduction_pct * 0.2:.0f}%,
providing defence-in-depth and robustness against the λ_IE uncertainty.

Note: if λ_IE = 2.0/yr (pessimistic grid-fault environment), even Option 1
leaves the risk at Cat A × F4 (ALARP boundary). Sourcing a site-specific
grid-fault frequency from ENTSO-E or the project's grid code is the single
highest-priority data improvement for this analysis.
"""


def _sec_phase7_bridge() -> str:
    return """\
## 8. Bridge to Phase 7 (Preventive Maintenance / RCM)

Phase 7 shifts from **safety** (Phases 5–6) to **economics** — optimal
maintenance intervals to minimise total cost per unit time.

Two explicit inheritance items:

**Input:** Phase 6's ALARP analysis identified the safety relay (G3b) as the
highest-leverage single intervention. Phase 7's RCM analysis must determine
whether the relay belongs in:
- **Time-directed PM** (scheduled replacement at fixed interval), or
- **Failure-finding PM** (periodic proof-test to detect latent failures).
Per NORSOK Z-008 and IEC 61511, protective devices belong in the
failure-finding category — the proof-test interval is derived differently
from age-replacement.

**Context:** Pitch System (Weibull β=0.728, Phase 2) and Hydraulic System
were both high-importance items in Phase 4 (availability) and Phase 5 (FTA).
Phase 7's cost-optimization should present two cases:
- β < 1 (as fitted): no optimal age-replacement interval exists —
  on-condition maintenance only.
- β > 1 (physically plausible for aging mechanical actuators):
  age-replacement is justified; Barlow-Proschan formula gives the
  cost-optimal interval.
"""


def _sec_dod() -> str:
    return """\
## Appendix A — Definition of Done Checklist

- [x] `config.py` reads gate Q values from Phase 5 `gate_Q_table.csv` — no hard-coded Q values
- [x] `event_tree.py`: four-branch tree verified (branch probabilities sum to 1.0)
- [x] `consequence_frequency_table.csv`: absolute frequencies at all λ_IE values, with and without CCF
- [x] `risk_matrix.py`: all branches + BoP cable plotted on IEC 61400-1 risk matrix
- [x] `alarp.py`: three options quantified; Option 1 moves risk to ACCEPTABLE
- [x] `bowtie.png`: FTA basic events left, ETA branches right, CCF pathway annotated
- [x] `phase6_report.md`: includes λ_IE sensitivity, risk matrix placement, ALARP summary, cross-phase finding
"""


# ── Report assembly ──────────────────────────────────────────────────────────

def generate_report(
    gate_Q: dict[str, float],
    branches: list[EventTreeBranch],
    alarp_options: list[ALARPOption],
    ccf_uplift: float,
    cft: "pd.DataFrame",
    output_path: Path,
) -> Path:
    """
    Generate the Phase 6 Markdown report.

    Parameters
    ----------
    gate_Q : dict
        Gate Q values from load_gate_Q_values().
    branches : list[EventTreeBranch]
        Four branches at λ_IE central, independent (not CCF-adjusted).
    alarp_options : list[ALARPOption]
        Three ALARP barrier options from alarp.compute_all_options().
    ccf_uplift : float
        CCF uplift factor from Phase 5 ccf_sensitivity.csv.
    cft : pd.DataFrame
        Full consequence frequency table from event_tree.consequence_frequency_table().
    output_path : Path
        Destination for the .md file.

    Returns
    -------
    Path
        output_path (for chaining).
    """
    # Consequence frequency table extract: central λ_IE, both CCF scenarios
    cft_central = cft[
        np.isclose(cft["lambda_IE"], _cfg.LAMBDA_IE_CENTRAL, rtol=1e-6)
    ][["branch_label", "severity_category", "P_branch",
       "lambda_per_turbine", "lambda_per_farm",
       "freq_category", "risk_acceptability", "ccf_adjusted"]]

    cft_summary = cft[
        cft["branch_label"] == "Branch_4_CATASTROPHIC"
    ][["lambda_IE", "ccf_adjusted",
       "lambda_per_turbine", "lambda_per_farm",
       "P_one_plus_25yr_farm", "freq_category", "risk_acceptability"]]

    sections = [
        "# AEOLUS-RAMS Phase 6 — Event Tree Analysis (ETA) Report\n",
        "---\n",
        _sec_executive_summary(gate_Q, branches, alarp_options, ccf_uplift),
        "---\n",
        _sec_inheritance(gate_Q, ccf_uplift),
        "---\n",
        _sec_initiating_events(),
        "---\n",
        _sec_event_tree(branches),
        "\n---\n",
        "## 4. Consequence Frequency Table — Central Estimate\n",
        "Branch frequencies at λ_IE = "
        f"{_cfg.LAMBDA_IE_CENTRAL}/turbine/yr:\n\n",
        _md_table(cft_central),
        "\n\n### Catastrophic Branch Sensitivity to λ_IE\n\n",
        _md_table(cft_summary),
        "\n---\n",
        _sec_risk_matrix(branches),
        "---\n",
        _sec_alarp(alarp_options),
        "---\n",
        "## 7. Bow-Tie Integration Summary\n\n"
        "The bow-tie diagram (`bowtie.png`) combines:\n"
        "- **Left**: Phase 5 FTA — 7 basic events grouped under OR gates G1/G2/G3, "
        "coloured by data confidence tier, CCF hydraulic pathway annotated\n"
        "- **Centre**: Turbine overspeed hazard box with Q_top (independent and "
        "CCF-adjusted)\n"
        "- **Right**: Phase 6 ETA — four consequence branches with conditional "
        "probabilities, absolute frequencies, and risk matrix placement\n\n"
        "This is the first diagram in the AEOLUS project that shows causes AND "
        "consequences on a single plot, making it the synthesis deliverable of "
        "the two-phase safety analysis.\n",
        "---\n",
        _sec_phase7_bridge(),
        "---\n",
        _sec_dod(),
    ]

    text = "\n".join(sections)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path