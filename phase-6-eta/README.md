# AEOLUS-RAMS Phase 6 — Event Tree Analysis (ETA)

Builds on Phase 5's fault tree outputs to compute **absolute consequence
frequencies**, place results on the **IEC 61400-1 Ed.4 risk matrix**, and
produce a complete **bow-tie diagram** and **ALARP argument** for the turbine
overspeed hazard.

## Quick Start

```bash
pip install -e ".[dev]"
aeolus-rams-phase6 --phase5-dir ../phase-5-fta/outputs --output-dir outputs/
pytest tests/ -v