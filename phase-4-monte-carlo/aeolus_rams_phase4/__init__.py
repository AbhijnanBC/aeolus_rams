"""
AEOLUS-RAMS — Phase 4: Monte Carlo Simulation
===============================================

Produces the farm-level availability distribution that Phase 3's static
R(t) model could not: a histogram of annual farm availability values across
10,000 simulated years, accounting for component repair and weather-gated
offshore vessel access.

Phase 3 cross-validation entry point:
    from aeolus_rams_phase4.turbine_state import simulate_turbine
    from aeolus_rams_phase4.config import TURBINE_COMPONENTS_ZERO_REPAIR

    # With MTTR=0 on all components, A_turbine must equal R_turbine(T) from Phase 3.
    # See tests/test_turbine_state.py::test_zero_repair_time_gives_R_t for the
    # exact numerical check that enforces this boundary condition.

Module map
----------
config              All 15 component parameter rows + scenario/sweep definitions
component_sampler   ComponentParams dataclass + failure_repair_sequence() generator
turbine_state       Event-driven single-turbine renewal simulation
farm_state          N-turbine joint-timeline farm aggregation (exact, not binomial approx)
monte_carlo         M-simulation orchestrator + summary statistics
sensitivity         Single-parameter sweep engine
scenarios           Baseline / Optimised / Degraded param-list builders
plots               Figures 1–4 (matplotlib)
reporting           Phase 4 Markdown report
pipeline            CLI entry point

Quick start
-----------
    python -m aeolus_rams_phase4.pipeline \\
        --mtbf-table ../phase-2-weibull-mtbf-hazard/outputs/mtbf_table.csv \\
        --output-dir outputs/
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("aeolus-rams-phase4")
except PackageNotFoundError:
    __version__ = "1.0.0-phase4"

__all__ = ["__version__"]