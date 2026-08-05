"""
aeolus_rams_phase7.rcm_schedule
==================================
Assembles the master RCM maintenance schedule for all 13 AEOLUS components.

This is the primary Phase 7 deliverable — the table a maintenance planner
would use operationally.

RCM decision logic per NORSOK Z-008:2017 Annex C and IEC 60300-3-14:2004:

Q1: Does failure affect safety or environment?
    YES → Q2   NO → Q3

Q2: Is there a reliable condition indicator before functional failure?
    YES → CONDITION_BASED (CBM)
    NO  → Is there a PM task effective in reducing risk?
         YES → depends on mode (Q4 for dormant devices; AGE_REPLACEMENT or
               TIME_DIRECTED for active components)
         NO  → Redesign (unacceptable — cannot run to failure for safety-critical)

Q3: Does PM reduce failure probability? (i.e., β > 1)
    YES → Is PM cost-effective (Cp << Cf × savings)?
         YES → AGE_REPLACEMENT at T*
         NO  → RUN_TO_FAILURE with spare-part buffer
    NO (β ≤ 1) → Is CBM feasible?
         YES → CONDITION_BASED
         NO  → RUN_TO_FAILURE

Q4 (dormant protective devices):
    Is the device proof-tested at regular intervals?
    YES → FAILURE_FINDING (verify τ achieves SIL target via PFDavg)
    NO  → FAILURE_FINDING (derive required τ from SIL target)

Each component entry below cites: the RCM logic path taken, the data used
to make the decision, and the interval or trigger rule that results.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    LITERATURE_WEIBULL, REPLACEMENT_COSTS, DESIGN_LIFE_DAYS,
    PHASE5_Q_G3b, TAU_IMPLIED_G3b_DAYS, LAMBDA_G3b,
    SIL2_PFDavg_TARGET, SIL_TARGETS,
    PHASE2_MTBF_TABLE, PHASE5_GATE_Q, PHASE6_ALARP,
)
from .age_replacement import optimal_replacement_age, cost_ratio_sensitivity
from .failure_finding import (
    proof_test_interval_table, required_proof_test_interval, PFDavg_exact,
)


# ---------------------------------------------------------------------------
# RCM category constants
# ---------------------------------------------------------------------------

AGE_REPLACEMENT = "AGE_REPLACEMENT"
TIME_DIRECTED   = "TIME_DIRECTED"
CONDITION_BASED = "CONDITION_BASED"
FAILURE_FINDING = "FAILURE_FINDING"
RUN_TO_FAILURE  = "RUN_TO_FAILURE"

CATEGORY_DESCRIPTIONS = {
    AGE_REPLACEMENT: (
        "Replace preventively at optimal age T* (Barlow-Proschan). "
        "Applicable when β > 1 (wear-out). T* minimises expected cost rate."
    ),
    TIME_DIRECTED:   (
        "Scheduled PM at fixed calendar interval. Applicable for consumables "
        "(fluids, greases) or when condition monitoring is impractical and "
        "a vendor PM interval is well-established."
    ),
    CONDITION_BASED: (
        "Replace or intervene when a degradation indicator crosses a threshold. "
        "Appropriate when β ≤ 1 (no PM benefit) or when a reliable condition "
        "signal exists (vibration, oil analysis, ML anomaly score)."
    ),
    FAILURE_FINDING: (
        "Proof-test at interval τ to reveal latent faults in dormant protective "
        "devices. Interval τ derived from IEC 61511 PFDavg formula to maintain "
        "the required SIL level."
    ),
    RUN_TO_FAILURE:  (
        "No preventive action. Replace on failure; maintain spare-part buffer. "
        "Appropriate when β ≤ 1 (no PM benefit), MTTR is short, and "
        "condition monitoring is impractical or cost-ineffective."
    ),
}


# ---------------------------------------------------------------------------
# Schedule builder
# ---------------------------------------------------------------------------

def build_rcm_schedule(
    Q_G3b: float | None = None,
    tau_implied_G3b_days: float | None = None,
) -> pd.DataFrame:
    """Build the master RCM maintenance schedule for all 13 components.

    Parameters
    ----------
    Q_G3b : float, optional
        Override Phase 5 Q_G3b (for testing). Default: config.PHASE5_Q_G3b.
    tau_implied_G3b_days : float, optional
        Override implied proof-test interval for G3b. Default: 730d.

    Returns
    -------
    pd.DataFrame
        One row per component. Columns:
        component, rcm_category, maintenance_interval_days,
        maintenance_interval_label, trigger_rule,
        T_star_days, T_star_years, C_star_per_day, C_no_PM_per_day,
        savings_fraction, PFDavg_current, tau_required_days, SIL_achieved,
        mttf_days, beta, eta, data_source, notes.
    """
    if Q_G3b is None:
        Q_G3b = PHASE5_Q_G3b
    if tau_implied_G3b_days is None:
        tau_implied_G3b_days = TAU_IMPLIED_G3b_DAYS

    rows: list[dict] = []

    # ── A1: Main/Rotor Bearing — AGE_REPLACEMENT ──────────────────────────
    for comp_name in ["Main/Rotor Bearing", "Gearbox", "Generator"]:
        params = LITERATURE_WEIBULL[comp_name]
        costs  = REPLACEMENT_COSTS[comp_name]
        res    = optimal_replacement_age(params, costs)

        if res.T_star_days is not None and res.T_star_days <= DESIGN_LIFE_DAYS:
            category = AGE_REPLACEMENT
            interval_days  = res.T_star_days
            interval_label = f"{res.T_star_years:.1f} yr"
            trigger = (
                f"Replace at T* = {res.T_star_years:.1f} yr (Barlow-Proschan optimal, "
                f"β={params.beta}, Cp/Cf={costs.ratio:.2f}). "
                f"Savings vs run-to-failure: {res.savings_fraction*100:.1f}%."
            )
        else:
            # T* > design life → CBM preferred
            category = CONDITION_BASED
            interval_days  = None
            interval_label = "On-condition"
            trigger = (
                f"T* = {res.T_star_years:.1f} yr > design life (25 yr). "
                f"Condition-based monitoring preferred: vibration analysis (accelerometer), "
                f"oil debris monitoring (magnetic plug / spectrometry)."
            )

        rows.append({
            "component":                comp_name,
            "rcm_category":             category,
            "maintenance_interval_days": interval_days,
            "maintenance_interval_label": interval_label,
            "trigger_rule":             trigger,
            "T_star_days":              res.T_star_days,
            "T_star_years":             res.T_star_years,
            "C_star_per_day":           res.C_star,
            "C_no_PM_per_day":          res.C_no_PM,
            "savings_fraction":         res.savings_fraction,
            "PFDavg_current":           None,
            "tau_required_days":        None,
            "SIL_achieved":             None,
            "mttf_days":                res.mttf_days,
            "beta":                     params.beta,
            "eta":                      params.eta,
            "data_source":              f"Literature (β={params.beta}, η={params.eta:.0f}d). {params.citation[:80]}…",
            "notes":                    res.T_star_warning,
        })

    # ── B: Electrical Safety System G3b — FAILURE_FINDING ─────────────────
    tau_sil2   = required_proof_test_interval(LAMBDA_G3b, SIL2_PFDavg_TARGET)
    pfd_current = PFDavg_exact(LAMBDA_G3b, tau_implied_G3b_days)  # = Q_G3b ≈ 0.027132

    rows.append({
        "component":                "Electrical Safety System (G3b Safety Relay)",
        "rcm_category":             FAILURE_FINDING,
        "maintenance_interval_days": tau_sil2,
        "maintenance_interval_label": f"{tau_sil2:.0f} d ({tau_sil2/30.44:.1f} months)",
        "trigger_rule":             (
            f"Proof-test at τ ≤ {tau_sil2:.0f} d (IEC 61511 SIL-2 requirement). "
            f"Derived: λ_G3b = {LAMBDA_G3b:.3e}/d from Q_G3b={Q_G3b:.4f} at τ_implied={tau_implied_G3b_days:.0f}d. "
            f"SIL-2 target: PFDavg ≤ {SIL2_PFDavg_TARGET:.0e}. "
            f"Current PFDavg = {pfd_current:.4f} (well above SIL-2 — proof-testing not yet implemented). "
            f"Phase 6 ALARP Option 1: replace with dual-channel SIL-2 relay module AND implement monthly proof-test."
        ),
        "T_star_days":              None,
        "T_star_years":             None,
        "C_star_per_day":           None,
        "C_no_PM_per_day":          None,
        "savings_fraction":         None,
        "PFDavg_current":           pfd_current,
        "tau_required_days":        tau_sil2,
        "SIL_achieved":             "SIL-2 (after τ ≤ τ_required AND Q_G3b reduced to 1e-3 per Phase 6 Option 1)",
        "mttf_days":                1.0 / LAMBDA_G3b,
        "beta":                     1.0,      # exponential assumed for dormant device
        "eta":                      None,
        "data_source":              "Phase 5 gate_Q_table.csv (G3b.Q=0.027132); Phase 6 alarp_table.csv (Option 1); IEC 61511",
        "notes":                    (
            f"PFDavg formula: PFDavg = 1 - [1-exp(-λτ)]/(λτ). "
            f"Without proof-testing, PFDavg = {pfd_current:.4f} (effectively <SIL-1). "
            f"Implementing monthly proof-tests AND replacing the relay with SIL-2 hardware "
            f"achieves PFDavg ≤ 1e-3 (SIL-2) as specified by Phase 6 ALARP Option 1."
        ),
    })

    # ── C: Pitch System — CONDITION_BASED (β=0.728 < 1) ──────────────────
    from .config import PITCH_BETA, PITCH_MTBF
    rows.append({
        "component":                "Pitch System",
        "rcm_category":             CONDITION_BASED,
        "maintenance_interval_days": None,
        "maintenance_interval_label": "On-condition",
        "trigger_rule":             (
            f"ML anomaly score ≥ μ_rolling + 2σ_rolling (30-day rolling baseline). "
            f"β = {PITCH_BETA:.4f} < 1 (Phase 2 Tier A Weibull MLE fit, 10 TBF intervals): "
            f"hazard rate is decreasing — no finite T* exists, PM at any fixed interval "
            f"increases expected cost. The SCADA ML anomaly score IS the Phase 7 maintenance "
            f"trigger. Starting threshold k=2σ; tune with operational data (Phase 8+)."
        ),
        "T_star_days":              None,
        "T_star_years":             None,
        "C_star_per_day":           None,
        "C_no_PM_per_day":          None,
        "savings_fraction":         0.0,
        "PFDavg_current":           None,
        "tau_required_days":        None,
        "SIL_achieved":             None,
        "mttf_days":                PITCH_MTBF,
        "beta":                     PITCH_BETA,
        "eta":                      None,
        "data_source":              "Phase 2 mtbf_table.csv (β=0.7285, Tier A Weibull, AIC preferred exponential). Phase 1 SCADA anomaly pipeline.",
        "notes":                    (
            "β < 1: Barlow-Proschan C(T) is monotonically decreasing — scheduling PM at "
            "any fixed interval T ≤ MTTF is always suboptimal. The Phase 2 AIC analysis "
            "preferred the exponential model (ΔAIC=0.63), consistent with random/infant-"
            "mortality failure behaviour. CBM is the only mathematically justified strategy."
        ),
    })

    # ── D: Time-directed and run-to-failure components ────────────────────
    td_and_rtf = [
        {
            "component":   "Hydraulic System",
            "rcm_category": TIME_DIRECTED,
            "maintenance_interval_days": 548,      # 18 months
            "maintenance_interval_label": "18 months",
            "trigger_rule": "Hydraulic fluid replacement + pump/filter inspection at 18-month intervals. Sample oil at 9 months to check viscosity/contamination.",
            "T_star_days": None, "T_star_years": None,
            "C_star_per_day": None, "C_no_PM_per_day": None, "savings_fraction": None,
            "PFDavg_current": None, "tau_required_days": None, "SIL_achieved": None,
            "mttf_days": 1844.7, "beta": 1.0, "eta": None,
            "data_source": "Phase 2 mtbf_table.csv (MTBF=1845d, Tier B exponential). Vendor PM schedule (offshore hydraulic systems).",
            "notes": "β≈1 assumed (no Weibull fit — insufficient TBF data). Fluid degradation is time-cumulative; 18-month interval matches OEM guidance for offshore pitch hydraulics. Phase 5: G2a FV=0.77 (rank 2) — Hydraulic failure is a high-leverage safety risk, justifying structured PM.",
        },
        {
            "component":   "Mechanical Brake",
            "rcm_category": TIME_DIRECTED,
            "maintenance_interval_days": 730,       # 24 months
            "maintenance_interval_label": "24 months",
            "trigger_rule": "Brake disc measurement + caliper overhaul + 24VAC supply verification at 24-month intervals. Replace disc if thickness < 80% of nominal.",
            "T_star_days": None, "T_star_years": None,
            "C_star_per_day": None, "C_no_PM_per_day": None, "savings_fraction": None,
            "PFDavg_current": None, "tau_required_days": None, "SIL_achieved": None,
            "mttf_days": 4400.0, "beta": 1.0, "eta": None,
            "data_source": "Phase 2/3 (MTBF=4400d assumed_placeholder). OEM scheduled maintenance.",
            "notes": "Brake wear is correlated with rotor stop cycles (usage-dependent) not calendar time. 24-month conservative bound; consider cycle-count trigger if SCADA logs stop events.",
        },
        {
            "component":   "Transformer",
            "rcm_category": CONDITION_BASED,
            "maintenance_interval_days": 365,       # Annual DGA
            "maintenance_interval_label": "12 months (annual DGA)",
            "trigger_rule": "Annual dissolved gas analysis (DGA) of insulating oil. Maintenance triggered if key gas thresholds per IEC 60599 are exceeded (H₂ > 100ppm, C₂H₂ > 1ppm).",
            "T_star_days": None, "T_star_years": None,
            "C_star_per_day": None, "C_no_PM_per_day": None, "savings_fraction": None,
            "PFDavg_current": None, "tau_required_days": None, "SIL_achieved": None,
            "mttf_days": 6000.0, "beta": None, "eta": None,
            "data_source": "Phase 2/3 (MTBF=6000d assumed_placeholder). IEC 60076-14; IEC 60599 DGA thresholds.",
            "notes": "DGA detects partial discharge, arc, thermal faults, and moisture ingress — all leading indicators. Phase 4 found Transformer to be the largest per-turbine downtime driver (crane-dependent repair). Placeholder MTBF=6000d; actual MTBF likely 10,000–20,000d for modern offshore units.",
        },
        {
            "component":   "Yaw System",
            "rcm_category": TIME_DIRECTED,
            "maintenance_interval_days": 365,
            "maintenance_interval_label": "12 months",
            "trigger_rule": "Yaw bearing grease replenishment + slew ring gap inspection + yaw motor current logging annually.",
            "T_star_days": None, "T_star_years": None,
            "C_star_per_day": None, "C_no_PM_per_day": None, "savings_fraction": None,
            "PFDavg_current": None, "tau_required_days": None, "SIL_achieved": None,
            "mttf_days": 4300.0, "beta": 1.0, "eta": None,
            "data_source": "Phase 2/3 (MTBF=4300d assumed_placeholder). OEM grease replenishment schedule.",
            "notes": "",
        },
        {
            "component":   "SCADA/Communication",
            "rcm_category": RUN_TO_FAILURE,
            "maintenance_interval_days": None,
            "maintenance_interval_label": "On failure",
            "trigger_rule": "Replace BK1120/NC300 module on failure. Maintain minimum 1 spare module on-site. Monitor communication health via SCADA watchdog alarms.",
            "T_star_days": None, "T_star_years": None,
            "C_star_per_day": None, "C_no_PM_per_day": None, "savings_fraction": None,
            "PFDavg_current": None, "tau_required_days": None, "SIL_achieved": None,
            "mttf_days": 37150.0, "beta": 1.0, "eta": None,
            "data_source": "Phase 2/3 (MTBF=37150d, eff. MTTR=1.2d, Tier C posterior).",
            "notes": "MTBF=37,150d ≈ 102yr. Very long MTBF + very short MTTR (1.2d) → PM incurs Cp with near-zero availability benefit. β≈1 confirmed by posterior. Phase 5: G3a FV=0.178 (rank 7) — FV reflects its appearance in 6/12 MCS but its individual Q is low (0.006).",
        },
        {
            "component":   "Converter",
            "rcm_category": RUN_TO_FAILURE,
            "maintenance_interval_days": None,
            "maintenance_interval_label": "On failure",
            "trigger_rule": "Replace power electronics module on failure. Monitor IGBT junction temperature via SCADA thermal alarms. Keep 1 replacement module in spare parts store.",
            "T_star_days": None, "T_star_years": None,
            "C_star_per_day": None, "C_no_PM_per_day": None, "savings_fraction": None,
            "PFDavg_current": None, "tau_required_days": None, "SIL_achieved": None,
            "mttf_days": 37290.0, "beta": 1.0, "eta": None,
            "data_source": "Phase 2/3 (MTBF=37290d, eff. MTTR=8.3d, Tier C posterior).",
            "notes": "MTBF=37,290d ≈ 102yr, β≈1 (exponential). PM has no β>1 justification. MTTR=8.3d — spare part pre-positioning reduces effective MTTR further.",
        },
        {
            "component":   "Cooling System",
            "rcm_category": TIME_DIRECTED,
            "maintenance_interval_days": 730,
            "maintenance_interval_label": "24 months",
            "trigger_rule": "Coolant flush + heat exchanger inspection + pump bearing check at 24-month intervals. Verify coolant inhibitor concentration and pH.",
            "T_star_days": None, "T_star_years": None,
            "C_star_per_day": None, "C_no_PM_per_day": None, "savings_fraction": None,
            "PFDavg_current": None, "tau_required_days": None, "SIL_achieved": None,
            "mttf_days": 11000.0, "beta": 1.0, "eta": None,
            "data_source": "Phase 2/3 (MTBF=11000d assumed_placeholder). Standard offshore cooling PM schedule.",
            "notes": "Coolant degradation (inhibitor depletion, corrosion) is time-dependent. OEM interval is typically 2 years for offshore generators/converters.",
        },
        {
            "component":   "Grounding/Lightning Protection",
            "rcm_category": TIME_DIRECTED,
            "maintenance_interval_days": 365,
            "maintenance_interval_label": "12 months",
            "trigger_rule": "Visual inspection of grounding brushes + earth continuity test (resistance < 1Ω) annually. Check lightning receptor tip wear per IEC 61400-24.",
            "T_star_days": None, "T_star_years": None,
            "C_star_per_day": None, "C_no_PM_per_day": None, "savings_fraction": None,
            "PFDavg_current": None, "tau_required_days": None, "SIL_achieved": None,
            "mttf_days": 14600.0, "beta": 1.0, "eta": None,
            "data_source": "Phase 2/3 (MTBF=14600d assumed_placeholder). IEC 61400-24 inspection interval.",
            "notes": "Annual lightning protection inspection is mandated by IEC 61400-24 Section 8. Low cost, low consequence if caught early; corrosion of grounding brushes is time-dependent.",
        },
    ]
    rows.extend(td_and_rtf)

    df = pd.DataFrame(rows)
    # Canonical column order
    ordered_cols = [
        "component", "rcm_category", "maintenance_interval_days",
        "maintenance_interval_label", "trigger_rule",
        "T_star_days", "T_star_years", "C_star_per_day", "C_no_PM_per_day",
        "savings_fraction", "PFDavg_current", "tau_required_days", "SIL_achieved",
        "mttf_days", "beta", "eta", "data_source", "notes",
    ]
    return df[[c for c in ordered_cols if c in df.columns]]


# ---------------------------------------------------------------------------
# Age-replacement summary table
# ---------------------------------------------------------------------------

def build_age_replacement_table() -> pd.DataFrame:
    """Build the age-replacement result table for bearing, gearbox, generator."""
    rows = []
    for comp_name, params in LITERATURE_WEIBULL.items():
        costs = REPLACEMENT_COSTS[comp_name]
        res = optimal_replacement_age(params, costs)
        rows.append({
            "component":          comp_name,
            "beta":               params.beta,
            "eta_days":           params.eta,
            "mttf_days":          res.mttf_days,
            "mttf_years":         res.mttf_days / 365.25,
            "Cp_usd":             costs.Cp,
            "Cf_usd":             costs.Cf,
            "cp_cf_ratio":        costs.ratio,
            "T_star_days":        res.T_star_days,
            "T_star_years":       res.T_star_years,
            "C_star_per_day":     res.C_star,
            "C_no_PM_per_day":    res.C_no_PM,
            "savings_fraction":   res.savings_fraction,
            "beta_supports_PM":   res.beta_supports_PM,
            "exceeds_design_life": (
                res.T_star_days is not None and res.T_star_days > DESIGN_LIFE_DAYS
            ),
            "data_source":        params.source,
            "notes":              res.T_star_warning,
        })
    return pd.DataFrame(rows)


def build_cost_ratio_sensitivity_table() -> pd.DataFrame:
    """Sweep Cp/Cf ratio for all three age-replacement components."""
    all_frames = []
    for comp_name, params in LITERATURE_WEIBULL.items():
        if params.beta <= 1.0:
            continue
        base_Cf = REPLACEMENT_COSTS[comp_name].Cf
        df = cost_ratio_sensitivity(params, base_Cf)
        all_frames.append(df)
    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()


def build_failure_finding_table() -> pd.DataFrame:
    """Build the PFDavg vs. τ table for G3b safety relay."""
    return proof_test_interval_table(
        lambda_per_day=LAMBDA_G3b,
        component="Electrical Safety System (G3b Safety Relay)",
    )