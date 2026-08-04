# phase-5-fta

**Status: Complete**

AEOLUS-RAMS Phase 5 — Fault Tree Analysis (FTA).

Answers a fundamentally different question from Phase 4:

| Phase 4 | Phase 5 |
|---|---|
| What fraction of time is the farm delivering power? | Given a load-rejection demand, what is P(catastrophic overspeed)? |
| A_farm ≈ 0.928 (dominated by export cable) | Q_top ≈ 7.3×10⁻⁴/yr (dominated by G3b safety relay + G2a hydraulic) |

## Verified Numbers (pre-computed before any code)

From Phase 3 `component_rt_table.csv`:

## Quick Start

```bash
pip install -e ".[dev]"

# Run tests first (mandatory before full run)
pytest tests/ -v

# Full run (with probabilistic FTA, ~3 min):
aeolus-rams-phase5 --rt-table ../phase-3-rbd/outputs/component_rt_table.csv

# Fast run (skip probabilistic FTA, ~10s):
aeolus-rams-phase5 --skip-probabilistic