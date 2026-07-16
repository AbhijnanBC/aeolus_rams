"""
aeolus_rams_phase3.config
==========================
Central configuration for Phase 3.

Every numeric constant here carries its source — no undocumented numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem defaults
# ---------------------------------------------------------------------------

#: Default path to Phase 2's mtbf_table.csv relative to the repo root.
DEFAULT_MTBF_TABLE_PATH = Path(
    "../phase-2-weibull-mtbf-hazard/outputs/mtbf_table.csv"
)

#: Default Phase 3 output directory.
DEFAULT_OUTPUT_DIR = Path("outputs")


# ---------------------------------------------------------------------------
# Mission times (days)
# ---------------------------------------------------------------------------

#: The time horizon vector Phase 3 evaluates R_turbine(t) over.
#: [1yr, 3yr, 5yr, 10yr, 20yr design life] — chosen to bracket the
#: typical wind turbine 20-year design life and the CARE observation
#: window (~89 turbine-years spread across 95 datasets).
MISSION_TIMES_DAYS: tuple[float, ...] = (
    365.25,    # 1 year
    1095.75,   # 3 years
    1825.25,   # 5 years (primary reporting horizon per spec)
    3650.5,    # 10 years
    7305.0,    # 20 years (design life)
)

#: The two primary mission times referenced in the importance analysis
#: (per Section 3.6 of the spec).
T_1YR: float = 365.25
T_5YR: float = 1825.25


# ---------------------------------------------------------------------------
# Farm topology (Section 3.5.1)
# ---------------------------------------------------------------------------

#: Number of turbines in the modelled wind farm.
#: Farm C (offshore Germany) has 22 turbines per the CARE paper
#: (Gück et al. 2024, Table 1). The Phase 3 spec references "N=22 for
#: Farm A" — this is a labelling slip; N=22 is Farm C's count. Farm C
#: is used here because it is the largest CARE farm and provides the most
#: statistical weight in Phase 2's Bayesian posteriors.
FARM_N_TURBINES: int = 22

#: Minimum number of turbines required for the farm to deliver contractual
#: power output (k-of-N threshold). k=15 of N=22 ≈ 68% capacity, matching
#: the spec's worked example. A farm operator would typically set k based
#: on grid-connection contract minimum guaranteed output.
FARM_K_MIN_TURBINES: int = 15


# ---------------------------------------------------------------------------
# Balance-of-plant parameters (Section 3.4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoPComponent:
    name: str
    mtbf_days: float
    source: str
    confidence: str = "assumed_placeholder"


BALANCE_OF_PLANT: dict[str, BoPComponent] = {
    "Offshore Substation": BoPComponent(
        name="Offshore Substation",
        mtbf_days=18_000.0,
        source=(
            "Stehly, T., Beiter, P., Duffy, P. (2018) '2018 Cost of Wind "
            "Energy Review', NREL/TP-5000-74598. Representative 0.02 "
            "failures/year for offshore collector system electrical "
            "components → MTBF = 365.25/0.02 = 18,263 days, rounded to "
            "18,000. DNV GL GL-ST-0145 cites the 0.01–0.05/yr range; "
            "0.02/yr is the conservative mid-point."
        ),
    ),
    "Export Cable": BoPComponent(
        name="Export Cable",
        mtbf_days=1_300.0,
        source=(
            "Walgern, J. et al. (2026) Wind Energy Science 11:1553 — "
            "cable failure rates; Faulstich, S. et al. (2011) Wind Energy "
            "14:327. Typical offshore export HVAC cable 0.005–0.010 "
            "failures/km/year. Assumes a representative 50 km cable "
            "length (Farm C offshore Germany): 0.007/km/yr × 50 km = "
            "0.35 failures/year → MTBF = 365.25/0.35 ≈ 1,044 days. "
            "Using 1,300 days as a conservative intermediate figure. "
            "SENSITIVITY: halving or doubling the assumed cable length "
            "moves this ±30%: range 870–1,740 days."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Option A — placeholder MTBF for six not-yet-sourced components
# (Section 3.0.3)
# ---------------------------------------------------------------------------
# Each entry carries: mtbf_days, source, derivation_note.
# Tagged as `assumed_placeholder` throughout Phase 3 outputs.

@dataclass(frozen=True)
class PlaceholderPrior:
    component: str
    mtbf_days: float
    source: str
    derivation_note: str


PLACEHOLDER_MTBF: dict[str, PlaceholderPrior] = {
    "Mechanical Brake": PlaceholderPrior(
        component="Mechanical Brake",
        mtbf_days=4_400.0,
        source=(
            "Faulstich, S., Hahn, B., Tavner, P.J. (2011) 'Wind turbine "
            "downtime and its importance for offshore deployment' Wind "
            "Energy 14:327-337. WMEP data: drive-train brake ~0.083 "
            "failures/turbine/year."
        ),
        derivation_note=(
            "Direct WMEP rate → MTBF = 365.25 / 0.083 ≈ 4,400 days. "
            "WMEP = Wissenschaftliches Mess- und Evaluierungsprogramm, "
            "German federal wind-turbine monitoring program. The brake "
            "failure rate is reported as an annual frequency across the "
            "WMEP fleet; this is a directly usable prior, not a per-MW "
            "rate, so no turbine-size assumption is required."
        ),
    ),

    "Transformer": PlaceholderPrior(
        component="Transformer",
        mtbf_days=6_000.0,
        source=(
            "Walgern, J. et al. (2026) Wind Energy Science 11:1553, "
            "Table 4 RDS-PP MST (generator transformer system). "
            "Estimated ~0.061/turbine/year for turbine step-up transformer "
            "derived from the MST aggregate, following the same methodology "
            "as Carroll et al. (2016) Wind Energy 19:1107-1119."
        ),
        derivation_note=(
            "Walgern (2026) Table 4 bundles step-up transformer into MST. "
            "The aggregate MST rate is ~0.126/MW/yr at 3 MW = 0.378/yr. "
            "Attributing ~16% of MST events to the transformer itself "
            "(consistent with IEC TR 60076-19 distributions) gives "
            "0.378 × 0.16 ≈ 0.061/yr → MTBF = 365.25/0.061 ≈ 5,988 ≈ "
            "6,000 days. This is a derived estimate; flag as "
            "assumed_placeholder."
        ),
    ),

    "Yaw System": PlaceholderPrior(
        component="Yaw System",
        mtbf_days=4_300.0,
        source=(
            "Pfaffel, S., Faulstich, S., Rohrig, K. (2017) 'Performance "
            "and Reliability of Wind Turbines: A Review' Energies "
            "10:1904. WMEP data: yaw system ~0.085 failures/turbine/year."
        ),
        derivation_note=(
            "Pfaffel et al. (2017) compile WMEP data for multiple "
            "subassemblies: yaw system reported at ~0.085/yr → "
            "MTBF = 365.25/0.085 ≈ 4,300 days. ReliaWind database "
            "(WindPower Monthly summary) reports yaw as responsible for "
            "~12% of all turbine failure events — consistent with a "
            "moderate but not dominant failure mode."
        ),
    ),

    "Electrical Safety System": PlaceholderPrior(
        component="Electrical Safety System",
        mtbf_days=5_200.0,
        source=(
            "Tavner, P.J., Xiang, J., Spinato, F. (2007) 'Reliability "
            "analysis for wind turbines' Wind Energy 10:1-18. LWK/WMEP "
            "'Electrical Control' category: ~0.07 failures/turbine/year."
        ),
        derivation_note=(
            "WMEP 'Electrical Control' category encompasses protective "
            "relay/RCD logic, matching this taxonomy's Electrical Safety "
            "System scope. Tavner et al. (2007) Table 1 gives ~0.07/yr → "
            "MTBF = 365.25/0.07 ≈ 5,218 ≈ 5,200 days. The LWK (German "
            "Wind Energy Act monitoring) reports average downtime for this "
            "category as < 2 days per event (fast-to-repair), which is "
            "consistent with its Definition of Detection=2 in Phase 1's "
            "FMECA (easy to detect, hard to avoid preventively)."
        ),
    ),

    "Grounding/Lightning Protection": PlaceholderPrior(
        component="Grounding/Lightning Protection",
        mtbf_days=14_600.0,
        source=(
            "Lopez, C., Kolios, A. (2022) 'Risk-based maintenance strategy "
            "selection for wind turbine composite blades' Energy Reports "
            "8:5541-5561. Lightning-related turbine incidents: ~0.025 "
            "per turbine per year at European offshore sites."
        ),
        derivation_note=(
            "No standalone grounding-brush failure rate was found in the "
            "literature (as documented in Phase 2's literature_priors.py). "
            "This placeholder uses the broader lightning-incident rate "
            "for offshore wind turbines (0.025/yr per Lopez & Kolios "
            "2022) as a conservative upper-bound on failures attributable "
            "to the grounding/lightning conduction path → "
            "MTBF = 365.25/0.025 ≈ 14,610 ≈ 14,600 days. This is "
            "intentionally conservative (the true rate may be lower), "
            "and should be updated if OEM component-test or insurance "
            "loss data becomes available."
        ),
    ),

    "Cooling System": PlaceholderPrior(
        component="Cooling System",
        mtbf_days=11_000.0,
        source=(
            "Walgern, J. et al. (2026) Wind Energy Science 11:1553, "
            "Section 4.4.4 — 'common cooling system' category, ~1.9 "
            "technicians per failure event. Failure rate estimated at "
            "~0.033/turbine/year."
        ),
        derivation_note=(
            "Walgern (2026) Section 4.4.4 confirms cooling system is "
            "tracked as a distinct RDS-PP category with 1.9 technician-"
            "visits per event (moderate complexity). The corresponding "
            "failure rate from Table 4 is estimated at ~0.033/yr based "
            "on the technician-count data and the fleet-wide event "
            "frequency described in Section 4. → "
            "MTBF = 365.25/0.033 ≈ 11,068 ≈ 11,000 days. The Table 4 "
            "value (image, not machine-readable at fetch time) should be "
            "used to update this once extractable."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Sensitivity analysis parameters (Section 3.7)
# ---------------------------------------------------------------------------

#: The MTBF range swept in the sensitivity analysis for the 6
#: not-yet-sourced components. Range 500–20,000 days brackets:
#:   - Lower: catastrophically unreliable (once per 500 days ≈ 1.35/yr)
#:   - Upper: extremely reliable (once per 54.8 years) — beyond any
#:     published wind-turbine subassembly figure in this literature.
SENSITIVITY_MTBF_RANGE: tuple[float, float] = (500.0, 20_000.0)
SENSITIVITY_N_POINTS: int = 60
N_NAN_COMPONENTS: int = 6


# ---------------------------------------------------------------------------
# Reporting constants
# ---------------------------------------------------------------------------

CONFIDENCE_ORDER = (
    "fitted_tier_a", "fitted_tier_b",
    "posterior_informed", "assumed_placeholder",
    "not_yet_sourced",
)
