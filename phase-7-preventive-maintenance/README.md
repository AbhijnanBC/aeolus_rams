# phase-7-preventive-maintenance

**Status: Complete**

AEOLUS-RAMS Phase 7 — Preventive Maintenance & RCM.

This is the "so what" phase. Six phases of characterisation converted into
**concrete, mathematically justified maintenance decisions** for all 13 components.

## Key Findings

| Finding | Value | Source |
|---|---|---|
| Bearing T* (Barlow-Proschan) | **23.0 yr** | β=2.0, η=25000d, Cp/Cf=0.10, savings=46% |
| Gearbox T* (Barlow-Proschan) | **22.3 yr** | β=1.8, η=24000d, Cp/Cf=0.10, savings=39% |
| Generator T* | **>25yr design life** → CBM | β=1.5, T*=54.5yr |
| G3b τ_SIL2 (IEC 61511) | **26.4 days ≈ monthly** | PFDavg ≤ 10⁻³ for SIL-2 |
| G3b current PFDavg | 0.0271 (<SIL-1) | No proof-test plan currently |
| Pitch System | **CBM (ML anomaly score)** | β=0.728<1, Phase 2 Tier A |

## PFDavg Formula Correction

The Phase 7 plan document used the wrong formula:
- ❌ Wrong: `PFDavg = [1-exp(-λτ)]/(λτ)` — this equals `(1-PFDavg)`, not PFDavg
- ✅ Correct: `PFDavg = 1 - [1-exp(-λτ)]/(λτ) ≈ λτ/2` (IEC 61508-6 Eq. B.4)

Verified: `PFDavg_exact(7.571e-5/d, 730d) = 0.027132 = Q_G3b (Phase 5) ✓`

## Quick Start

```bash
pip install -e ".[dev]"

# Run tests first (mandatory)
pytest tests/ -v

# Full run (reads Phase 5 gate_Q_table.csv):
aeolus-rams-phase7 \
    --gate-q-table ../phase-5-fta/outputs/gate_Q_table.csv \
    --output-dir outputs/