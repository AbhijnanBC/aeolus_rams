"""
aeolus_rams_phase3.topology
============================
Section 3.3.1 — Series/parallel topology decisions for the AEOLUS
turbine model, grounded in the Phase 2 R(t) analysis and the
Phase 1 FMECA taxonomy.

Every component is in series: any single failure stops power production.
This is the conservative, defensible choice for an availability-focused
RBD and matches how turbine SCADA systems report faults (one component
fault = turbine trips to safe state).

Special cases documented here
------------------------------
Pitch System: three physical axes but all three must be operational
for a certified pitch-to-feather (primary overspeed protection).
Loss of any axis = trip. Models as a single series block at the
component level. Sub-axis redundancy is an Enhancement-layer addition.

Mechanical Brake / Electrical Safety System: fail-safe protective
components. Both cause downtime when they false-trip (fail-safe mode).
Modelled in series from an AVAILABILITY perspective. Their
fail-to-danger mode is the subject of Phase 5 (FTA) — not this phase.

Cooling System: a single primary circuit with no confirmed redundant
path in the CARE fault log evidence. Conservative series assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

from aeolus_rams.taxonomy import COMPONENT_NAMES


@dataclass(frozen=True)
class ComponentRole:
    name: str
    subsystem_group: str
    topology_type: str       # "series" for all 13 in this model
    series_position: int     # left-to-right ordering in the diagram
    role_note: str


# The canonical series order matches Section 3.3.2's diagram:
# drivetrain path → power conversion path → control/protection layer.
# Order matters for diagram layout; all are functionally in series.
COMPONENT_ROLES: dict[str, ComponentRole] = {
    "Main/Rotor Bearing": ComponentRole(
        "Main/Rotor Bearing", "Drivetrain", "series", 1,
        "Load-bearing critical path; failure cascades to entire drivetrain",
    ),
    "Gearbox": ComponentRole(
        "Gearbox", "Drivetrain", "series", 2,
        "Torque transfer; failure = immediate turbine stop",
    ),
    "Generator": ComponentRole(
        "Generator", "Drivetrain/Electrical", "series", 3,
        "Power generation; any winding or bearing fault = zero output",
    ),
    "Converter": ComponentRole(
        "Converter", "Power Conversion", "series", 4,
        "Grid-side power conditioning; fault = no export",
    ),
    "Transformer": ComponentRole(
        "Transformer", "Power Conversion", "series", 5,
        "Step-up to grid voltage; single point of failure for power export",
    ),
    "Pitch System": ComponentRole(
        "Pitch System", "Rotor", "series", 6,
        "All three axes required; single-axis loss = safety trip. "
        "AIC-preferred exponential (ΔAIC=0.63 vs Weibull, β CI spans 1.0).",
    ),
    "Hydraulic System": ComponentRole(
        "Hydraulic System", "Nacelle", "series", 7,
        "Shared actuator supply for pitch and brake; loss = multi-system trip",
    ),
    "Yaw System": ComponentRole(
        "Yaw System", "Nacelle", "series", 8,
        "Sustained misalignment ⇒ load exceedances ⇒ controlled shutdown",
    ),
    "SCADA/Communication": ComponentRole(
        "SCADA/Communication", "Control & Safety", "series", 9,
        "Loss of SCADA link ⇒ conservative remote trip per IEC 61400-25",
    ),
    "Mechanical Brake": ComponentRole(
        "Mechanical Brake", "Nacelle", "series", 10,
        "False-trip (fail-safe) mode causes downtime; modelled in series "
        "for availability. Fail-to-danger mode → Phase 5 FTA.",
    ),
    "Electrical Safety System": ComponentRole(
        "Electrical Safety System", "Control & Safety", "series", 11,
        "False-trip (fail-safe) mode causes downtime; modelled in series "
        "for availability. Fail-to-danger mode → Phase 5 FTA.",
    ),
    "Cooling System": ComponentRole(
        "Cooling System", "Nacelle", "series", 12,
        "Single primary circuit; over-temperature protection trips turbine",
    ),
    "Grounding/Lightning Protection": ComponentRole(
        "Grounding/Lightning Protection", "Nacelle/Rotor", "series", 13,
        "Degraded conduction path raises latent risk; conservative treatment "
        "as availability component following WMEP precedent",
    ),
}

# Sorted by series_position for diagram rendering
ORDERED_COMPONENTS: tuple[str, ...] = tuple(
    name for name, _ in sorted(
        COMPONENT_ROLES.items(), key=lambda kv: kv[1].series_position
    )
)

assert set(COMPONENT_ROLES.keys()) == set(COMPONENT_NAMES), (
    f"topology.COMPONENT_ROLES does not match Phase 1 taxonomy.\n"
    f"Missing: {set(COMPONENT_NAMES) - set(COMPONENT_ROLES.keys())}\n"
    f"Extra:   {set(COMPONENT_ROLES.keys()) - set(COMPONENT_NAMES)}"
)
assert len(ORDERED_COMPONENTS) == 13, "Expected 13 series components"


def tier_colour(confidence: str) -> str:
    """Diagram block fill colour per confidence level (Section 3.9)."""
    return {
        "fitted_tier_a": "#2ecc71",          # green  — Tier A Weibull
        "fitted_tier_b": "#3498db",           # blue   — Tier B exponential
        "posterior_informed": "#e67e22",      # orange — Tier C Bayesian
        "assumed_placeholder": "#95a5a6",     # grey   — Tier C placeholder
        "not_yet_sourced": "#c0392b",         # red    — should not appear
    }.get(confidence, "#bdc3c7")
