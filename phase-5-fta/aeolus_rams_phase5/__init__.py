"""
AEOLUS-RAMS — Phase 5: Fault Tree Analysis
============================================

Answers a fundamentally different question from Phase 4:

  Phase 4: What fraction of time is the farm delivering power?   (availability)
  Phase 5: Given a load-rejection demand, what is P(overspeed)? (safety)

These are independent analyses. A turbine can have excellent availability
(rarely fails, quickly repaired) while simultaneously having a non-trivial
probability of catastrophic overspeed if its protection systems fail on demand.

Top event: Turbine Overspeed → Catastrophic Structural Failure
Three-layer defence (IEC 61400-1 compliant):
  G1 — Pitch-to-feather system (primary)
  G2 — Mechanical rotor brake (secondary)
  G3 — SCADA/Controller overspeed trip (tertiary)

Top event requires ALL THREE layers to fail simultaneously (AND gate).
Each layer can fail via multiple sub-causes (OR gates).

Phase 3 cross-validation entry points:
  Q values read directly from phase-3-rbd/outputs/component_rt_table.csv.
  Q_top must satisfy: Q_top = G1.Q × G2.Q × G3.Q (AND gate under independence).
  Verified against hand-calculation before any module code is committed.

Module map
----------
config          Q-values from Phase 3; sub-cause fractions; CCF parameters
fault_tree      BasicEvent, Gate, build_overspeed_fault_tree()
mcs             extract_minimal_cut_sets(), mcs_table()
importance      compute_importance_measures(): IB, CIM, Fussell-Vesely
ccf             apply_beta_factor_ccf(), ccf_sensitivity_sweep()
probabilistic   Q_uncertainty_from_mtbf_ci(), probabilistic_fta()
diagrams        render_fault_tree() (graphviz / text fallback)
reporting       render_phase5_report()
pipeline        CLI orchestrator
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("aeolus-rams-phase5")
except PackageNotFoundError:
    __version__ = "1.0.0-phase5"

__all__ = ["__version__"]