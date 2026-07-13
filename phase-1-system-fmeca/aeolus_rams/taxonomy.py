"""
aeolus_rams.taxonomy
======================
Section 1.3's ReliaWind-informed component taxonomy, plus the ordered
keyword rules used as the *fallback* layer of the failure-mode tagger
(component_tagger.py) for any free-text description that isn't already
covered by the curated, hand-reviewed lookup table.

Design notes
------------
- `PRIORITY_ORDER` operationalises Section 1.4's instruction to "tag the
  most specific/severe element as primary" for compound log entries: when
  a description matches keywords for more than one component, the
  component that ranks *higher* in this list wins the primary slot. The
  ordering is derived directly from the Severity (S) anchors in Section
  1.6 (fmeca.py's SEVERITY_DETECTION_DEFAULTS) — i.e. "more severe" is
  operationalised as "higher default Severity score" — with narrower,
  more specific systems breaking ties ahead of broad/generic ones.
- Keyword lists are intentionally ORDER-SENSITIVE at the rule-table level:
  rare/specific categories are declared before generic ones so a broad
  keyword occurring inside a compound entry can't swallow a more specific
  match (e.g. "yaw carbon brush" must not fall through to a generic
  "brush"/"grounding" rule ahead of the "yaw" rule).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Component:
    name: str
    subsystem_group: str
    covers: str
    in_scope: bool = True


#: Section 1.3 — the default taxonomy. Adjust only with documented
#: reasoning grounded in your own feature_description.csv / event log
#: (Section 1.3's closing instruction) — don't add categories you don't
#: have evidence for.
COMPONENT_TAXONOMY: tuple[Component, ...] = (
    Component(
        "Pitch System", "Rotor",
        "Blade pitch motors, encoders, control cards, slip rings, backup "
        "batteries, pitch bearing lubrication",
    ),
    Component(
        "Main/Rotor Bearing", "Drivetrain",
        "Main shaft support bearing",
    ),
    Component(
        "Gearbox", "Drivetrain",
        "Gears, gear oil supply/lubrication, oil cooling circuit",
    ),
    Component(
        "Generator", "Drivetrain/Electrical",
        "Generator core and generator-side bearing",
    ),
    Component(
        "Mechanical Brake", "Nacelle",
        "Brake disc, caliper, dedicated electrical/hydraulic supply",
    ),
    Component(
        "Hydraulic System", "Nacelle (shared)",
        "Pumps, accumulators, fluid supply feeding pitch and brake "
        "actuation",
    ),
    Component(
        "Converter", "Power Conversion",
        "Power electronics, fuses/filters",
    ),
    Component(
        "Transformer", "Power Conversion",
        "Step-up transformer",
    ),
    Component(
        "Yaw System", "Nacelle",
        "Yaw drive, yaw slip ring/grease system",
    ),
    Component(
        "SCADA/Communication", "Control & Safety",
        "Turbine controller fieldbus/communication, remote monitoring link",
    ),
    Component(
        "Electrical Safety System", "Control & Safety",
        "Safety chain relay, RCD (residual current device)",
    ),
    Component(
        "Grounding/Lightning Protection", "Nacelle/Rotor",
        "Lightning conduction path, grounding brushes",
    ),
    Component(
        "Cooling System", "Nacelle",
        "Non-gearbox cooling circuits (e.g. water cooling loop)",
    ),
)

COMPONENT_NAMES: tuple[str, ...] = tuple(c.name for c in COMPONENT_TAXONOMY)

#: Sentinel labels used by the tagger. Kept as constants so every module
#: compares against the same string rather than re-typing literals.
UNCLASSIFIED = "Unclassified — Review Manually"
NO_DESCRIPTION = "Unclassified / No Description"

#: Severity-derived primary/secondary tie-break order for compound entries.
#: See module docstring. Kept here (rather than in fmeca.py) because it is
#: a taxonomy-level policy, not a scoring computation.
PRIORITY_ORDER: tuple[str, ...] = (
    "Main/Rotor Bearing",           # S=10
    "Mechanical Brake",             # S=9, safety-critical redundant layer
    "Gearbox",                      # S=9
    "Generator",                    # S=9
    "Grounding/Lightning Protection",  # latent catastrophic risk, narrow/specific
    "Pitch System",                 # S=8, safety-critical (overspeed layer)
    "Electrical Safety System",     # S=8, safety-critical trip layer
    "Transformer",                  # S=8
    "Hydraulic System",             # S=7, shared support system
    "SCADA/Communication",          # S=6
    "Converter",                    # S=6
    "Cooling System",               # S=5
    "Yaw System",                   # S=5
)

assert set(PRIORITY_ORDER) == set(COMPONENT_NAMES), (
    "PRIORITY_ORDER must contain every taxonomy component exactly once"
)


# ---------------------------------------------------------------------------
# Keyword fallback rules
# ---------------------------------------------------------------------------
# Ordered mapping: component -> list of lowercase substrings. Evaluated
# top-to-bottom; see module docstring for why order matters. This is a
# FIRST-PASS tool, not a substitute for reading the actual text (Section
# 1.8's closing caveat) — it is the fallback layer behind
# component_tagger.CURATED_OVERRIDES, exercised only for descriptions that
# curated lookup doesn't already cover.

KEYWORD_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Grounding/Lightning Protection", (
        # Deliberately NOT a bare "carbon brush"/"carbonbrush" keyword:
        # that phrase is ambiguous between a rotor/spinner grounding brush
        # and a yaw slip-ring brush (see the curated "yaw carbon brush"
        # entry in component_tagger.py, tagged Yaw System). Keeping these
        # phrases specific avoids the generic term swallowing a more
        # specific "yaw ..." match declared later in this rule table.
        "grounding", "lightning", "earthing", "spinner_carbonbrush",
        "spinner carbon brush",
    )),
    ("Mechanical Brake", (
        "rotorbrake", "rotor brake", "brake disc", "brake caliper",
        "rotorbremse",
    )),
    ("Main/Rotor Bearing", (
        "main bearing", "rotor bearing", "rotorbearing", "main shaft bearing",
    )),
    ("Gearbox", (
        "gearbox", "gear oil", "gear tooth",
    )),
    ("Pitch System", (
        "pitch", "axis 1", "axis 2", "axis 3", "blade control", "slip ring",
        "beckhoff",
    )),
    ("Hydraulic System", (
        "hydraulic", "accumulator",
    )),
    ("Generator", (
        "generator",
    )),
    ("Converter", (
        "converter", "igbt", "umrichter",
    )),
    ("Transformer", (
        "transformer",
    )),
    ("Yaw System", (
        "yaw",
    )),
    ("SCADA/Communication", (
        "communication", "fieldbus", "scada", "fault bk", "nc300", "nc310",
    )),
    ("Cooling System", (
        "cooling", "cooler",
    )),
    ("Electrical Safety System", (
        "rcd", "safety chain", "relay", "dguv",
    )),
)

_covered = {name for name, _ in KEYWORD_RULES}
assert _covered == set(COMPONENT_NAMES), (
    f"KEYWORD_RULES is missing rules for: {set(COMPONENT_NAMES) - _covered}"
)


def component_by_name(name: str) -> Component:
    """Look up a Component by name; raises KeyError with a helpful message."""
    for c in COMPONENT_TAXONOMY:
        if c.name == name:
            return c
    raise KeyError(f"'{name}' is not in COMPONENT_TAXONOMY. Known: {COMPONENT_NAMES}")
