"""
AEOLUS-RAMS — Phase 7: Preventive Maintenance & RCM
=====================================================

Phase 7 converts all prior analysis into concrete, actionable maintenance
decisions. Three mathematically distinct problems correspond to three
engineering realities in the data:

PROBLEM A — Age-Based Replacement (Barlow-Proschan):
  For components with confirmed β > 1 (wear-out) from Phase 2 literature
  parameters, find the preventive-replacement interval T* that minimises
  expected cost per unit time. T* only exists when β > 1; for β ≤ 1 the
  cost-rate function is monotonically decreasing and run-to-failure or CBM
  is always cheaper than scheduled replacement.
  Components: Main/Rotor Bearing (β=2.0), Gearbox (β=1.8).
  Generator (β=1.5) has T* exceeding 25yr design life → CBM preferred.

PROBLEM B — Failure-Finding Proof-Test Interval (IEC 61511):
  For the G3b Safety Relay — a dormant protective device whose failures
  accumulate invisibly until a demand or proof test — derive the maximum
  proof-test interval τ_max that maintains PFDavg ≤ 10⁻³ (SIL-2 target
  from Phase 6 ALARP Option 1). The Phase 7 finding: τ_max ≈ 26 days
  (monthly proof-testing is required). Without explicit proof-testing,
  the relay's SIL-2 rating is meaningless — latent faults accumulate.

PROBLEM C — Condition-Based Maintenance (β ≤ 1):
  For the Pitch System (β=0.728, confirmed by Phase 2 Tier A fit), no
  finite optimal replacement age exists. The correct strategy is to
  replace or repair triggered by a degradation signal, not by calendar
  time. The degradation signal is the ML anomaly score from the SCADA
  pipeline — Phase 7 formalises this as a proportional-hazards CBM trigger.

PFDavg formula note
-------------------
The IEC 61508/61511 exact PFDavg for a device with constant failure rate
λ tested at interval τ is:

    PFDavg = 1 - [1 - exp(-λτ)] / (λτ)    [exact]
           ≈ λτ / 2                          [small-λτ approximation]

This is the correct formula used throughout Phase 7. An earlier plan
document incorrectly omitted the leading "1 -"; all code here uses the
validated expression confirmed by: PFDavg(λ_G3b, 730d) = Q_G3b = 0.027132.

Phase 5/6 handoff values (read-only — do not recompute)
---------------------------------------------------------
  Q_G3b     = 0.027132   (Phase 5 gate_Q_table.csv)
  λ_G3b     = 7.571e-5/d (derived via PFDavg_correct, τ_implied=730d)
  τ_SIL2    ≈ 26.4 days  (for PFDavg ≤ 1e-3 at λ_G3b)
  Q_top_1yr = 7.690e-4   (Phase 5 committed)
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("aeolus-rams-phase7")
except PackageNotFoundError:
    __version__ = "1.0.0-phase7"

__all__ = ["__version__"]