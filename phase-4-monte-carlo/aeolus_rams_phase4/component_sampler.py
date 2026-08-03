"""
aeolus_rams_phase4.component_sampler
======================================
Layer 1 of the simulation architecture.

Generates per-component failure/repair event sequences via a discrete-event
renewal process: (time_to_failure, time_to_repair) pairs drawn from the
fitted distributions in mtbf_table.csv (Phase 2) plus MTTR parameters
from Carroll et al. (2016).

All TBF distributions are Exponential — consistent with the memoryless
property of a Poisson failure process, confirmed by Phase 2's AIC analysis
for Tier A and Tier B components, and assumed for posterior_informed and
assumed_placeholder components following the same rationale.

TTR distributions are Exponential with mean = eff_MTTR (weather-adjusted).
The Exponential TTR assumption is conservative for minor repairs (Gamma fits
better in practice) but appropriate here: it is the maximum-entropy
distribution for a given mean, so it overstates repair time variability,
which conservatively overstates availability uncertainty.

Source for MTTR values and access fractions:
    Carroll, J., McDonald, A., McMillan, D. (2016) "Failure rate, repair
    time and unscheduled O&M cost analysis of offshore wind turbines."
    Wind Energy 19(7):1107–1119. DOI: 10.1002/we.1887.

Source for access fraction model:
    Dinwoodie, I. et al. (2015) "Reference Cases for Verification of
    Offshore Wind Farm O&M Decision Support Tools." Wind Engineering
    39(1):1–14. DOI: 10.1260/0309-524X.39.1.1.
    Model: eff_MTTR ≈ raw_MTTR / access_fraction (Poisson window arrivals,
    repair starts at next window after failure event).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class ComponentParams:
    """Immutable per-component simulation parameters.

    Attributes
    ----------
    name : str
        Canonical component name — must match Phase 3's topology.py.
    mtbf_days : float
        Mean time between failures (days). Source: Phase 2 mtbf_table.csv
        for 9 components; config.PLACEHOLDER_MTBF for 6 placeholders;
        config.BOP_PARAMS for Offshore Substation and Export Cable.
    mttr_raw_days : float
        Repair duration *if* a vessel were always available (days).
        Source: Carroll et al. (2016) Table 2 unless otherwise noted.
    access_fraction : float
        Fraction of the year during which a repair vessel can safely
        operate for this work category. Three tiers from Dinwoodie (2015):
          0.85 — light electrical (no vessel mobilisation required)
          0.60 — minor mechanical (CTV-accessible weather windows)
          0.30 — major crane lift (crane vessel, significant wave height < 1.5m)
    confidence : str
        Provenance tag inherited from Phase 2/3:
          fitted_tier_a | fitted_tier_b | posterior_informed |
          assumed_placeholder | assumed_bop
    is_bop : bool
        True for Offshore Substation and Export Cable.
    """

    name: str
    mtbf_days: float
    mttr_raw_days: float
    access_fraction: float
    confidence: str
    is_bop: bool = False

    @property
    def lambda_failure(self) -> float:
        """Failure rate (1/day). λ = 1/MTBF."""
        return 1.0 / self.mtbf_days

    @property
    def mttr_effective_days(self) -> float:
        """Weather-adjusted effective MTTR (days).

        eff_MTTR = raw_MTTR / access_fraction

        Derivation (Dinwoodie 2015, Section 3): Repair vessel access
        windows arrive as a Poisson process with rate ρ = access_fraction
        / mean_window_duration. Under this model, a turbine failing at a
        random time must wait E[W] days until the next window, where
        E[W] ≈ (1 - access_fraction) / (access_fraction * window_rate).
        For small access fractions (< 0.4), eff_MTTR ≈ raw_MTTR / access_fraction
        is accurate to within 10%. For higher fractions (> 0.7, e.g. SCADA),
        the formula slightly overstates waiting time — conservative.
        """
        return self.mttr_raw_days / self.access_fraction


def failure_repair_sequence(
    params: ComponentParams,
    rng: np.random.Generator,
) -> Iterator[tuple[float, float]]:
    """Yield (time_to_failure, time_to_repair) pairs indefinitely.

    Both drawn from independent Exponential distributions:
      TBF  ~ Exp(mean = params.mtbf_days)
      TTR  ~ Exp(mean = params.mttr_effective_days)

    The Exponential TBF model (memoryless) means the component has no
    ageing — consistent with the HPP assumption from Phase 2's AIC
    selection. Phase 5+ may introduce Weibull TBF for components where
    β ≠ 1 was physically meaningful.

    The caller accumulates pairs to build the component's failure/repair
    timeline: [up until t_fail₁, down until t_repair₁, up until t_fail₂, ...].
    The sequence always starts in the UP state (component is new/just-repaired
    at t=0), matching Phase 3's R(0) = 1 boundary condition.

    Parameters
    ----------
    params : ComponentParams
        Component-specific MTBF and effective MTTR.
    rng : np.random.Generator
        Seeded random generator — all callers must share a single generator
        initialised with np.random.default_rng(seed) to ensure cross-run
        reproducibility and avoid correlated streams.

    Yields
    ------
    (tbf, ttr) : tuple[float, float]
        Time to next failure (days from current UP start),
        time to repair (days of downtime).
    """
    while True:
        tbf = rng.exponential(params.mtbf_days)
        ttr = rng.exponential(params.mttr_effective_days)
        yield tbf, ttr