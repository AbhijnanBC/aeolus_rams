# phase-4-monte-carlo

**Status: Complete**

AEOLUS-RAMS Phase 4 — Monte Carlo Simulation.

Produces the farm-level availability distribution that Phase 3's static
R(t) model could not: a histogram of annual farm availability values
across 10,000 simulated years, with repair and offshore weather-gated
access modelled explicitly.

## Phase 3 → 4 Key Numbers

| Phase 3 | Phase 4 (Baseline) |
|---|---|
| R_turbine(1yr) = 0.4504 (P(never failed)) | A_turbine ≈ 0.974 (fraction of time up) |
| R_farm(1yr) = 0.0181 | A_farm ≈ 0.963 |

## Quick Start

```bash
cd phase-4-monte-carlo
pip install -e ".[dev]"

# Smoke test (1,000 simulations, ~2 min, no sweeps):
aeolus-rams-phase4 --n-simulations 1000 --skip-sweep --skip-convergence

# Full run (10,000 simulations, ~30 min with sweeps):
aeolus-rams-phase4 --output-dir outputs/ --verbose

# Tests:
pytest tests/ -v