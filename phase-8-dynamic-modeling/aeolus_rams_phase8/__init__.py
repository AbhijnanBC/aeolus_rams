"""
AEOLUS-RAMS — Phase 8: Dynamic Reliability & Predictive Modeling
=================================================================

Closes the four Phase 7 open items documented in rcm_schedule.csv and
cbm_trigger.py, and synthesises the complete 8-phase RAMS pipeline into
a unified technical narrative.

Analytical pillars
------------------
1. Bayesian Weibull Updating  — Main/Rotor Bearing (β=2.0, η=25000d)
   Grid posterior (150×150) updates Phase 2 point estimates with 15 new
   synthetic failure events; MAP shifts replacement interval T*.

2. Cox Proportional Hazards   — Pitch System (β=0.728, η=2653d)
   lifelines.CoxPHFitter estimates γ_vibration that parameterises the
   Phase 7 CBM trigger h(t|S) = h₀(t)·exp(γ·S(t)).

3. CBM Threshold Optimisation — Pitch System
   Monte Carlo sweep over k_sigma ∈ [0.5, 4.0] at N=5000 cycles/k
   finds k* minimising expected cost rate; replaces Phase 7 default k=2.

Module map
----------
config          All constants — zero hard-coded numbers anywhere else
synthetic_data  Reproducible synthetic failure + SCADA covariate generator
bayesian_update Grid Bayesian posterior; sequential update; MAP extraction
cox_survival    Cox PH fitting; hazard ratios; survival curve stratification
cbm_optimizer   k_sigma cost-rate sweep; k* extraction; RCM update text
reporting       Phase 8 Markdown report + project-wide synthesis narrative
pipeline        CLI orchestrator: reads Phase 7 CSVs, writes all outputs

Quick start
-----------
    pip install -e ".[dev]"
    aeolus-rams-phase8 --phase7-dir ../phase-7-preventive-maintenance \
                       --phase2-dir ../phase-2-weibull-mtbf-hazard   \
                       --output-dir outputs/
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("aeolus-rams-phase8")
except PackageNotFoundError:          # pragma: no cover
    __version__ = "1.0.0-phase8"

__all__ = ["__version__"]