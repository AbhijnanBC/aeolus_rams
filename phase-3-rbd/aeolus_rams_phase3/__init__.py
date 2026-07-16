"""
AEOLUS-RAMS — Phase 3: Reliability Block Diagram (RBD)
========================================================

Reads Phase 2's `mtbf_table.csv` directly — every R(t) value, topology
decision, and bottleneck call is grounded in what Phase 2 actually
produced.

Critical decisions grounded in the Phase 2 report and execution plan
----------------------------------------------------------------------
1. AIC-preferred exponential for Pitch System (Section 3.0.4):
   Phase 2's Weibull fit returned AIC_W=172.63, AIC_exp=172.00
   (ΔAIC=0.63, preferred=False for Weibull). Phase 3 uses the
   AIC-preferred exponential (MTBF=1,936 days, λ=1/1936).

2. All-exponential series system (Section 3.3.3):
   AIC comparison supports exponential for every component. The
   series product reduces to exp(-λ_total × t), enabling a clean
   closed-form analytical result at the turbine level.

3. Option A for six NaN-MTBF components (Section 3.0.3):
   Mechanical Brake, Transformer, Yaw System, Electrical Safety
   System, Grounding/Lightning Protection, Cooling System are
   assigned literature placeholders tagged `assumed_placeholder`.
   See `config.PLACEHOLDER_MTBF` for exact values and sources.

4. Bayesian posteriors treated as best available (Section 3.0.2):
   The weak Gamma prior means posteriors ≈ MLE rates. Use them as
   the best available MTBF — but Phase 4 Monte Carlo must apply
   wider variance to `posterior_informed` rows than to
   `fitted_tier_a` rows.

5. Linkage uncertainty (Section 3.0.1):
   88/95 events were sensor-resolved ambiguous matches. Failure
   counts carry ±1 implicit uncertainty. Phase 4's sensitivity
   sweeps should probe this.

Module map
----------
config              Mission times, farm topology (N, k), BoP, placeholders
component_rt        R(t) callables from mtbf_table.csv — no NaN propagation
topology            Series topology definition and component roles
turbine_rbd         Analytical turbine-level R_system(t) and λ_system
farm_rbd            Farm-level k-of-N + BoP (partial analytical)
importance          Birnbaum and Criticality importance measures
sensitivity         Sensitivity band for six placeholder components
diagrams            Matplotlib RBD diagrams (turbine + farm)
reporting           Phase 3 Markdown report with live DoD checklist
pipeline            CLI orchestrator — reads mtbf_table.csv, writes outputs

Quick start
-----------
    python -m aeolus_rams_phase3.pipeline \\
        --mtbf-table ../phase-2-weibull-mtbf-hazard/outputs/mtbf_table.csv \\
        --output-dir outputs/

or, programmatically:

    from aeolus_rams_phase3.pipeline import run_phase3
    result = run_phase3(mtbf_table_path="path/to/mtbf_table.csv")
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("aeolus-rams-phase3")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "1.0.0-phase3"

__all__ = ["__version__"]
