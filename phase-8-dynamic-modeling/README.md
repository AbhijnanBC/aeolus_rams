# AEOLUS-RAMS Phase 8 — Dynamic Reliability & Predictive Modeling

**Capstone phase** of the 8-phase AEOLUS offshore wind RAMS pipeline.

Closes all Phase 7 open items:

| Open item | This phase's answer |
|---|---|
| CBM k_sigma threshold (default k=2.0) | Monte Carlo optimisation → k* |
| Cox PH γ coefficient (stub) | Cox PH fitted from SCADA covariates → γ_estimated |
| Weibull parameters are static | Bayesian grid posterior → MAP update mechanism |
| No cost-uncertainty quantification | Posterior-propagated T* with 90% CI |

## Installation

```bash
# From the phase-8-dynamic-modeling/ directory:
pip install -e ".[dev]"

Running the pipeline
SH

Copy
# With upstream phases installed:
aeolus-rams-phase8 \
    --phase7-dir ../phase-7-preventive-maintenance \
    --phase2-dir ../phase-2-weibull-mtbf-hazard    \
    --output-dir outputs/

# Standalone (uses config.py fallback constants):
aeolus-rams-phase8 --output-dir outputs/
Running tests
SH

Copy
pytest tests/ -v
Expected: 46+ tests, 0 failures (Cox PH tests skipped if lifelines absent).

Outputs
File	Description
synthetic_failures.csv	23 bearing failure events (15 + 8 censored)
synthetic_covariates.csv	220 Pitch System operational periods
posterior_grid.csv	22,500-row 150×150 posterior grid
bayesian_update_summary.csv	4-stage MAP estimates and T*
bayesian_update_plot.png	2×2 posterior diagnostic figure
sequential_update_plot.png	4-stage contour overlay
cox_hazard_ratios.csv	HR, 95% CI, z-score, p-value per covariate
cox_survival_curves.png	KM + Cox predicted survival
cox_ph_diagnostics.png	Schoenfeld + martingale residuals
cbm_threshold_table.csv	Full k_sigma × Cp/Cf sweep table
cbm_optimization_plot.png	Cost rate curves + k*
cbm_update_to_rcm.txt	Phase 7 rcm_schedule.csv update text
phase8_report.md	Complete Phase 8 report with synthesis