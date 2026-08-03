"""
aeolus_rams_phase4.config
==========================
Central configuration: all 15 component parameter rows, scenario
definitions, sensitivity sweep ranges, and filesystem paths.

Every numeric constant carries its source. Component MTBFs are taken
verbatim from Phase 3's committed outputs (component_rt_table.csv) and
config.BALANCE_OF_PLANT. MTTR values and access fractions come from
Carroll et al. (2016) and Dinwoodie et al. (2015).

Cross-validation checkpoint (verified against phase-3-rbd/outputs/):
    λ_system = Σ(1/MTBF_i) for 13 turbine components
             = 0.002184 /day  [Phase 3: 0.002183967231261054 /day ✓]
    MTBF_sys = 1/λ_system = 457.88 days  [Phase 3: 457.882327942524 days ✓]
"""

from __future__ import annotations

from pathlib import Path

from .component_sampler import ComponentParams


# ---------------------------------------------------------------------------
# Filesystem paths
# ---------------------------------------------------------------------------

DEFAULT_MTBF_TABLE_PATH = Path(
    "../phase-2-weibull-mtbf-hazard/outputs/mtbf_table.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs")

PHASE3_IMPORTANCE_PATH = Path(
    "../phase-3-rbd/outputs/importance_table.csv"
)


# ---------------------------------------------------------------------------
# Farm topology (from Phase 3 config.py and farm_rbd.py)
# ---------------------------------------------------------------------------

#: 22 turbines in Farm C (offshore Germany).
#: Source: Gück et al. (2024), Table 1 — CARE-to-Compare dataset.
FARM_N_TURBINES: int = 22

#: k=15 of 22 turbines required for contractual power delivery (≈ 68% capacity).
#: Source: Phase 3 spec, consistent with typical grid-connection contract terms.
FARM_K_MIN_TURBINES: int = 15

#: Primary mission time for Phase 4 (1 year = 365.25 days).
T_MISSION_DAYS: float = 365.25


# ---------------------------------------------------------------------------
# Per-component simulation parameters — 13 turbine components
#
# MTBFs: verbatim from phase-3-rbd/outputs/component_rt_table.csv.
# MTTRs and access fractions: Carroll et al. (2016) Table 2 + 3,
# supplemented by Dinwoodie et al. (2015) Appendix B.
#
# Access fraction tiers:
#   0.85 — light electrical, no crane, CTV in calm sea (Hs < 1.5 m → ~310d/yr)
#   0.60 — minor mechanical, CTV required (Hs < 1.5 m → ~219d/yr)
#   0.30 — major crane lift, crane vessel required (Hs < 2.0 m + Vw < 10 m/s → ~110d/yr)
# ---------------------------------------------------------------------------

TURBINE_COMPONENTS: list[ComponentParams] = [

    # ── Tier A: exponential fitted, AIC-preferred ─────────────────────────
    ComponentParams(
        name="Pitch System",
        mtbf_days=1936.3776678104537,   # Phase 3 exact value
        mttr_raw_days=3.0,              # Carroll (2016) Table 2: pitch/blade median 3d
        access_fraction=0.60,           # CTV-accessible, minor repair
        confidence="fitted_tier_a",
    ),

    # ── Tier B: exponential fitted ────────────────────────────────────────
    ComponentParams(
        name="Hydraulic System",
        mtbf_days=1844.703703703704,    # Phase 3 exact value
        mttr_raw_days=5.0,              # Carroll (2016) Table 2: hydraulics median 5d
        access_fraction=0.60,
        confidence="fitted_tier_b",
    ),

    # ── Tier C: posterior-informed (Bayesian, CARE data) ──────────────────
    ComponentParams(
        name="Gearbox",
        mtbf_days=28033.147886957817,
        mttr_raw_days=21.0,             # Carroll (2016) Table 3: gearbox major repair 21d
        access_fraction=0.30,           # Crane vessel required
        confidence="posterior_informed",
    ),
    ComponentParams(
        name="Main/Rotor Bearing",
        mtbf_days=29293.568135823352,
        mttr_raw_days=28.0,             # Carroll (2016) Table 3: main bearing 28d
        access_fraction=0.30,           # Crane vessel required
        confidence="posterior_informed",
    ),
    ComponentParams(
        name="SCADA/Communication",
        mtbf_days=37150.250660361286,
        mttr_raw_days=1.0,              # Carroll (2016): SCADA typically <1d; use 1d
        access_fraction=0.85,           # Remote/electrical, no vessel mobilisation
        confidence="posterior_informed",
    ),
    ComponentParams(
        name="Converter",
        mtbf_days=37289.56293551928,
        mttr_raw_days=5.0,              # Carroll (2016) Table 2: power electronics 5d
        access_fraction=0.60,
        confidence="posterior_informed",
    ),
    ComponentParams(
        name="Generator",
        mtbf_days=42203.017130032,
        mttr_raw_days=14.0,             # Carroll (2016) Table 3: generator 14d; crane req.
        access_fraction=0.30,           # Crane vessel for generator swap
        confidence="posterior_informed",
    ),

    # ── Tier C: assumed_placeholder (Option A from Phase 3) ───────────────
    # Source for MTBFs: phase-3-rbd/aeolus_rams_phase3/config.PLACEHOLDER_MTBF
    ComponentParams(
        name="Yaw System",
        mtbf_days=4300.0,
        mttr_raw_days=5.0,              # Carroll (2016): yaw motor/gear ~5d
        access_fraction=0.60,
        confidence="assumed_placeholder",
    ),
    ComponentParams(
        name="Mechanical Brake",
        mtbf_days=4400.0,
        mttr_raw_days=3.0,              # Carroll (2016): brake caliper/disc ~3d
        access_fraction=0.60,
        confidence="assumed_placeholder",
    ),
    ComponentParams(
        name="Electrical Safety System",
        mtbf_days=5200.0,
        mttr_raw_days=2.0,              # Carroll (2016): protective relay ~2d (fast)
        access_fraction=0.80,           # Electrical, CTV in mild seas
        confidence="assumed_placeholder",
    ),
    ComponentParams(
        name="Transformer",
        mtbf_days=6000.0,
        mttr_raw_days=14.0,             # Carroll (2016) Table 3: transformer 14d; crane
        access_fraction=0.30,           # Step-up transformer requires crane
        confidence="assumed_placeholder",
    ),
    ComponentParams(
        name="Cooling System",
        mtbf_days=11000.0,
        mttr_raw_days=3.0,              # Carroll (2016): cooling pump/heat exchanger ~3d
        access_fraction=0.60,
        confidence="assumed_placeholder",
    ),
    ComponentParams(
        name="Grounding/Lightning Protection",
        mtbf_days=14600.0,
        mttr_raw_days=3.0,              # Carroll (2016): grounding conductors ~3d
        access_fraction=0.60,
        confidence="assumed_placeholder",
    ),
]


# ---------------------------------------------------------------------------
# Balance-of-plant components (Phase 3: config.BALANCE_OF_PLANT)
# ---------------------------------------------------------------------------

BOP_COMPONENTS: list[ComponentParams] = [
    ComponentParams(
        name="Offshore Substation",
        mtbf_days=18000.0,              # Phase 3: Stehly et al. (2018) NREL/TP-5000-74598
        mttr_raw_days=60.0,             # Dinwoodie (2015): substation major outage ~60d
        access_fraction=0.30,           # Crane/heavy-lift vessel
        confidence="assumed_bop",
        is_bop=True,
    ),
    ComponentParams(
        name="Export Cable",
        mtbf_days=1300.0,               # Phase 3: Walgern et al. (2026); Faulstich et al. (2011)
        mttr_raw_days=45.0,             # Dinwoodie (2015): cable repair incl. lay/bury 45d
        access_fraction=0.30,           # Cable lay vessel, weather-gated
        confidence="assumed_bop",
        is_bop=True,
    ),
]


# ---------------------------------------------------------------------------
# Cross-validation constants (from Phase 3 outputs — must not be recomputed)
# ---------------------------------------------------------------------------

#: Phase 3 committed λ_system (sum of 1/MTBF for all 13 turbine components).
#: Source: phase-3-rbd/outputs/system_reliability_table.csv
PHASE3_LAMBDA_SYSTEM: float = 0.002183967231261054  # /day

#: Phase 3 committed MTBF_system = 1 / λ_system.
PHASE3_MTBF_SYSTEM: float = 457.882327942524  # days

#: Phase 3 committed R_turbine(1yr) = exp(-λ_system × 365.25).
PHASE3_R_TURBINE_1YR: float = 0.4503662982516767

#: Tolerance for the Phase 3 boundary-condition test (test_zero_repair_time_gives_R_t).
#: 0.8% allows for Monte Carlo noise at n=50,000 simulations.
PHASE3_BRIDGE_TOLERANCE: float = 0.008


# ---------------------------------------------------------------------------
# Scenario MTTR multipliers (Section 4.5)
# ---------------------------------------------------------------------------
# Applied uniformly to all component mttr_effective_days.
# Models logistics capability, not component physics.

SCENARIOS: dict[str, dict] = {
    "baseline": {
        "mttr_multiplier": 1.0,
        "label": "Baseline",
        "description": (
            "Shared charter vessel, weather-gated access as per Carroll (2016) "
            "literature MTTR values. Representative of current offshore O&M practice."
        ),
        "color": "#2196F3",  # blue
    },
    "optimised": {
        "mttr_multiplier": 0.6,
        "label": "Optimised",
        "description": (
            "Dedicated fast-response Service Operations Vessel (SOV), improved "
            "weather forecasting, pre-positioned spare parts. Reduces effective "
            "MTTR by ~40% across all work categories."
        ),
        "color": "#4CAF50",  # green
    },
    "degraded": {
        "mttr_multiplier": 2.5,
        "label": "Degraded",
        "description": (
            "Severe North Sea winters, vessel sharing with adjacent farms, "
            "long queue delays. Multiplies effective MTTR by 2.5× — represents "
            "the operational risk tail."
        ),
        "color": "#F44336",  # red
    },
}


# ---------------------------------------------------------------------------
# Sensitivity sweep parameters (Section 4.4.5)
# ---------------------------------------------------------------------------

import numpy as np

#: Sweep 1 — Pitch System MTBF (highest IC component from Phase 3).
SWEEP_PITCH_MTBF: np.ndarray = np.linspace(800.0, 5000.0, 22)

#: Sweep 2 — Hydraulic System MTBF (highest IC by absolute lambda from Phase 3).
SWEEP_HYDRAULIC_MTBF: np.ndarray = np.linspace(800.0, 5000.0, 22)

#: Sweep 3 — Export Cable MTBF (dominant BoP risk; Phase 3 bop_sensitivity_table).
#: Range 500–5,000 days matches Phase 3's bop_sensitivity_table range.
SWEEP_CABLE_MTBF: np.ndarray = np.linspace(500.0, 5000.0, 22)

#: Sweep 4 — MTTR multiplier (applied to ALL components simultaneously).
#: Range 0.4–3.5 spans "dedicated SOV" to "severe access constraint".
SWEEP_MTTR_MULT: np.ndarray = np.linspace(0.4, 3.5, 22)

#: Number of simulations per sweep point (lower than main run for speed).
SWEEP_N_SIMULATIONS: int = 2000

#: Number of simulations for main scenario runs.
MAIN_N_SIMULATIONS: int = 10_000

#: Random seed for all Monte Carlo runs.
RANDOM_SEED: int = 42