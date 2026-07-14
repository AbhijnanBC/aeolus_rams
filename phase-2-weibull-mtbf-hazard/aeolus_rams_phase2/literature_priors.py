"""
aeolus_rams_phase2.literature_priors
========================================
Section 2.4, Tier C — literature-informed MTBF placeholders.

The spec's own reference snippet ships this dict as all-`None`, with the
instruction "cite a source before filling this in." This module does
that filling-in for real, with each entry's derivation shown in full so
you can check it, disagree with it, or replace it with a better source
without archaeology.

Two honesty rules enforced throughout this file:

1. Every populated entry states EXACTLY where its number comes from and,
   where it isn't a direct citable annual-failure-rate, HOW it was
   derived (e.g. Main/Rotor Bearing's figure comes from a cumulative
   20-year failure-probability statistic converted via a constant-hazard
   / HPP assumption, not a direct annual rate — that conversion is shown,
   not hidden).
2. Where the wind-turbine reliability literature genuinely doesn't break
   a component out as its own tracked category (it's common for
   Mechanical Brake, Electrical Safety System, Cooling System, and
   Grounding/Lightning Protection to be bundled into broader "drive
   train", "auxiliary systems", or "electrical system" aggregates), this
   file says so explicitly via `confidence="not_yet_sourced"` and points
   at exactly which published table to go pull the number from — rather
   than fabricating a number to make the dict look complete.
"""

from __future__ import annotations

from dataclasses import dataclass


CITED_DIRECT = "cited_direct"                    # a published annual rate, used as-is
DERIVED = "derived_from_cumulative_probability"   # converted via an explicit assumption
DERIVED_ASSUMED_RATING = "derived_needs_rating_assumption"  # per-MW rate x assumed turbine size
NOT_YET_SOURCED = "not_yet_sourced"               # no defensible number found — do not fabricate


@dataclass(frozen=True)
class LiteraturePrior:
    component: str
    mtbf_days: float | None
    confidence: str
    source: str
    derivation_note: str

    @property
    def is_usable(self) -> bool:
        return self.mtbf_days is not None


# ---------------------------------------------------------------------------
# The priors
# ---------------------------------------------------------------------------

LITERATURE_PRIORS: dict[str, LiteraturePrior] = {

    "Main/Rotor Bearing": LiteraturePrior(
        component="Main/Rotor Bearing",
        mtbf_days=20481.0,
        confidence=DERIVED,
        source="Hart, E. et al. (2019), cited in Hart, E., Clarke, B., "
               "Nicholas, G., Kazemi Amiri, A., Stirling, J., Carroll, J., "
               "Dwyer-Joyce, R., McDonald, A., Long, H. (2020) 'A review of "
               "wind turbine main bearings: design, operation, modelling, "
               "damage mechanisms and fault detection', Wind Energy "
               "Science, 5, 105-124, https://doi.org/10.5194/wes-5-105-2020",
        derivation_note=(
            "Source reports MB failure rates 'as high as 30% over a "
            "20-year design lifetime' (a cumulative failure probability, "
            "not a direct annual rate). Converted via the constant-hazard "
            "/ homogeneous-Poisson-process assumption Dao et al. (2019) "
            "document as standard for WT reliability data: "
            "lambda = -ln(1 - 0.30) / 20 years = 0.01783 / year -> "
            "MTBF = 1/lambda = 56.07 years = 20481 days. This is a "
            "reported UPPER-END figure ('as high as') — treat this MTBF "
            "as a conservative (short) estimate, not a central tendency."
        ),
    ),

    "Gearbox": LiteraturePrior(
        component="Gearbox",
        mtbf_days=2372.0,
        confidence=CITED_DIRECT,
        source="Carroll, J., McDonald, A., McMillan, D. (2016) 'Failure "
               "rate, repair time and unscheduled O&M cost analysis of "
               "offshore wind turbines', Wind Energy, 19, 1107-1119, "
               "https://doi.org/10.1002/we.1887",
        derivation_note=(
            "Directly reported: 0.154 failures/turbine/year for offshore "
            "turbines rated 2-4 MW -> MTBF = 365.25 / 0.154 = 2372 days. "
            "Note this is an OFFSHORE figure; CARE's Farm A is onshore "
            "(Portugal, ex-EDP) while Farms B/C are offshore (Germany) "
            "per the CARE paper itself — if most of your Gearbox Tier-C "
            "incidents came from Farm A, an onshore-specific figure would "
            "be more appropriate than this one."
        ),
    ),

    "Generator": LiteraturePrior(
        component="Generator",
        mtbf_days=3845.0,
        confidence=CITED_DIRECT,
        source="Carroll, J., McDonald, A., McMillan, D. (2016) 'Failure "
               "rate, repair time and unscheduled O&M cost analysis of "
               "offshore wind turbines', Wind Energy, 19, 1107-1119, "
               "https://doi.org/10.1002/we.1887",
        derivation_note=(
            "Directly reported: 0.095 failures/turbine/year (offshore, "
            "2-4 MW) -> MTBF = 365.25 / 0.095 = 3845 days. Same onshore/"
            "offshore caveat as Gearbox above."
        ),
    ),

    "Converter": LiteraturePrior(
        component="Converter",
        mtbf_days=982.0,
        confidence=DERIVED_ASSUMED_RATING,
        source="Walgern, J., Stratmann, N., Horn, M., Then, N.W.Y., "
               "Menzel, M., Anderson, F., Kolios, A., Fischer, K. (2026) "
               "'Reliability and O&M key performance indicators of "
               "onshore and offshore wind turbines based on field-data "
               "analysis', Wind Energy Science, 11, 1553-1568, "
               "https://doi.org/10.5194/wes-11-1553-2026",
        derivation_note=(
            "Source reports the converter system at 0.124 failures/MW/"
            "year (offshore) and 0.223 failures/MW/year (onshore) — "
            "rates are normalised per MW of rated turbine capacity, not "
            "per turbine, because failure rate scales with turbine size. "
            "CARE's turbine rated power is anonymised and not recoverable "
            "from the published files, so this figure ASSUMES a "
            "representative 3 MW offshore turbine: "
            "0.124 x 3 = 0.372 failures/year -> MTBF = 365.25/0.372 = "
            "982 days. Recompute with your fleet's actual rated power if "
            "you can source it (e.g. from the wind farm operator or the "
            "CARE Zenodo record's metadata) — this is the single most "
            "assumption-dependent figure in this file."
        ),
    ),

    "SCADA/Communication": LiteraturePrior(
        component="SCADA/Communication",
        mtbf_days=959.0,
        confidence=DERIVED_ASSUMED_RATING,
        source="Walgern, J. et al. (2026), Wind Energy Science, 11, "
               "1553-1568, https://doi.org/10.5194/wes-11-1553-2026 — "
               "RDS-PP 'control system' category",
        derivation_note=(
            "No published source breaks out a narrow 'SCADA/fieldbus "
            "communication' failure rate on its own; the closest "
            "well-quantified proxy is the RDS-PP 'control system' "
            "category this source reports at 0.127 failures/MW/year "
            "(offshore) / 0.255 (onshore) — broader than pure SCADA/"
            "communication (it also covers general controller logic), so "
            "treat this as an UPPER-BOUND proxy, not a tight estimate. "
            "Same 3 MW assumption as Converter above: 0.127 x 3 = 0.381/"
            "year -> MTBF = 365.25/0.381 = 959 days."
        ),
    ),

    "Yaw System": LiteraturePrior(
        component="Yaw System",
        mtbf_days=None,
        confidence=NOT_YET_SOURCED,
        source="See Tavner, P.J. et al. (2007-2013) WMEP/LWK-based "
               "studies (e.g. summarised in "
               "https://www.windpowermonthly.com/article/1302791) and "
               "Table 4 of Walgern et al. (2026), WES 11:1553.",
        derivation_note=(
            "The ReliaWind taxonomy consistently tracks Yaw System as its "
            "own category (WindPower Monthly's summary of the ReliaWind "
            "database reports yaw responsible for ~12% of all turbine "
            "failure EVENTS, second only to pitch at ~16% — a relative "
            "share, not an annual rate) — but no source search turned up "
            "a directly citable failures/turbine/year figure specific to "
            "this taxonomy's Yaw System definition. Pull the number "
            "straight from Table 4 (RDS-PP code MDL) of Walgern et al. "
            "(2026) rather than trusting a back-calculation from the 12% "
            "share figure above, which depends on an assumed total "
            "turbine failure rate this file does not want to guess at."
        ),
    ),

    "Mechanical Brake": LiteraturePrior(
        component="Mechanical Brake",
        mtbf_days=None,
        confidence=NOT_YET_SOURCED,
        source="Tavner, P.J., Xiang, J., Spinato, F. (2007) 'Reliability "
               "analysis for wind turbines', Wind Energy, 10, 1-18 — LWK/"
               "WMEP subassembly chart names 'Mechanical Brake' as a "
               "distinct tracked category (reproduced in NREL/TP-5000-"
               "58774, https://docs.nrel.gov/docs/fy13osti/58774.pdf).",
        derivation_note=(
            "Confirmed this is tracked as its own named category in the "
            "classic German WMEP/LWK surveys, but the specific numeric "
            "failure-rate value wasn't extractable from the sources "
            "reachable here (only the category name, in a bar-chart "
            "figure rather than a machine-readable table). Read the "
            "value directly off that chart, or from Table 4 (RDS-PP "
            "drive-train-brake subcategory) of Walgern et al. (2026), "
            "before filling this in."
        ),
    ),

    "Transformer": LiteraturePrior(
        component="Transformer",
        mtbf_days=None,
        confidence=NOT_YET_SOURCED,
        source="Walgern et al. (2026), WES 11:1553, Table 4 — bundled "
               "under the 'generator transformer system' (RDS-PP MST) "
               "category.",
        derivation_note=(
            "This source's own taxonomy groups the transformer together "
            "with generator switchgear under 'generator transformer "
            "system', not as a standalone figure — Table 4 (an image, "
            "not machine-readable text at fetch time) is the right place "
            "to read the isolated number if the underlying spreadsheet "
            "distinguishes them; otherwise treat the bundled MST rate as "
            "an upper bound."
        ),
    ),

    "Electrical Safety System": LiteraturePrior(
        component="Electrical Safety System",
        mtbf_days=None,
        confidence=NOT_YET_SOURCED,
        source="WMEP/LWK 'Electrical Control' and 'Electrical System' "
               "categories (NREL/TP-5000-58774, "
               "https://docs.nrel.gov/docs/fy13osti/58774.pdf).",
        derivation_note=(
            "Classic surveys track 'Electrical Control' and 'Electrical "
            "System' as distinct named categories that plausibly bracket "
            "this taxonomy's safety-chain-relay/RCD scope, but again only "
            "the category names were extractable here, not the numeric "
            "rate. LWK reports this category's average DOWNTIME as under "
            "2 days per failure (fast to repair once found) — useful "
            "context for a Detection-score sanity check even without an "
            "MTBF figure."
        ),
    ),

    "Cooling System": LiteraturePrior(
        component="Cooling System",
        mtbf_days=None,
        confidence=NOT_YET_SOURCED,
        source="Walgern et al. (2026), WES 11:1553, Sect. 4.4.4 mentions "
               "a distinct 'common cooling system' category (technician-"
               "count data only, 1.9 technicians/event) and full failure "
               "rates in Table 4.",
        derivation_note=(
            "Confirmed as its own tracked RDS-PP category in this recent, "
            "high-quality source (>4200 turbine-years), but the failure-"
            "rate figure itself sits in Table 4, an image at fetch time. "
            "Pull it from there, or from the XLSX download linked "
            "alongside the article "
            "(https://wes.copernicus.org/articles/11/1553/2026/"
            "wes-11-1553-2026-t04.xlsx)."
        ),
    ),

    # Pitch System and Hydraulic System are included here only for
    # completeness (so `get_prior` never raises on a valid taxonomy
    # component). In THIS project's own data they landed in Tier A and
    # Tier B respectively and were fit directly — no literature review
    # was performed for them, and `pipeline.py` only ever consults this
    # dict for a component actually assigned Tier C on a given run.
    "Pitch System": LiteraturePrior(
        component="Pitch System",
        mtbf_days=None,
        confidence=NOT_YET_SOURCED,
        source="Not researched — Pitch System had >= 8 incidents in this "
               "project's own Phase 1 output and was fit directly "
               "(Tier A). See Walgern et al. (2026) or Walgern, Fischer, "
               "Hentschel, Kolios (2023) Energy Reports 9:3273-3281 if a "
               "future run needs this as a Tier C fallback.",
        derivation_note=(
            "Only present so `get_prior('Pitch System')` doesn't raise if "
            "a future re-run of this pipeline (different CARE download, "
            "different farms) happens to land Pitch System in Tier C."
        ),
    ),
    "Hydraulic System": LiteraturePrior(
        component="Hydraulic System",
        mtbf_days=None,
        confidence=NOT_YET_SOURCED,
        source="Not researched — Hydraulic System had 5-7 incidents in "
               "this project's own Phase 1 output and was fit directly "
               "as Tier B (exponential-only). See Carroll et al. (2016) "
               "for an offshore pitch/hydraulic combined figure if a "
               "future run needs this as a Tier C fallback.",
        derivation_note=(
            "Same completeness rationale as Pitch System above."
        ),
    ),

    "Grounding/Lightning Protection": LiteraturePrior(
        component="Grounding/Lightning Protection",
        mtbf_days=None,
        confidence=NOT_YET_SOURCED,
        source="No source search turned up a standalone grounding-brush/"
               "lightning-conduction-path failure rate distinct from "
               "general blade lightning-damage statistics.",
        derivation_note=(
            "Lightning-related damage is well documented at the BLADE "
            "level (e.g. Lopez & Kolios (2022), Energy Reports, 8, "
            "5541-5561, on composite blade risk-based maintenance "
            "including lightning strikes) but this taxonomy's specific "
            "scope — grounding brushes and the conduction path itself, "
            "as distinct from lightning-caused blade damage — is rarely "
            "broken out on its own in the published failure-rate "
            "literature. This is the least-precedented of the 13 "
            "components; a genuinely defensible prior here likely needs "
            "an OEM component-test spec or an insurance-industry loss "
            "database rather than an academic field-data study."
        ),
    ),
}


def get_prior(component: str) -> LiteraturePrior:
    if component not in LITERATURE_PRIORS:
        raise KeyError(
            f"No literature prior entry for '{component}'. Known: "
            f"{sorted(LITERATURE_PRIORS)}"
        )
    return LITERATURE_PRIORS[component]


def tier_c_r_of_t(prior: LiteraturePrior, t_days: float) -> float | None:
    """Section 2.10's bridge-to-Phase-3 formula: a Tier C component
    carries forward its literature MTBF as an ASSUMED-EXPONENTIAL
    reliability function R(t) = exp(-t / MTBF). This is a simplifying
    assumption to be stated explicitly wherever it's used (Phase 3's RBD
    in particular), not buried — returns None if no usable MTBF exists
    for this prior rather than silently defaulting to something."""
    if prior.mtbf_days is None:
        return None
    import math
    return math.exp(-t_days / prior.mtbf_days)
