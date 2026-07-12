# AEOLUS-RAMS

An integrated, data-driven RAMS (Reliability, Availability, Maintainability, Safety)
case study of an offshore wind farm — built one phase at a time.

## Progress

- [ ] Phase 0 — Prerequisites & Environment Setup
- [ ] Phase 1 — System Definition + FMECA
- [ ] Phase 2 — Weibull / MTBF / MTTF / Hazard Rate
- [ ] Phase 3 — RBD
- [ ] Phase 4 — Monte Carlo
- [ ] Phase 5 — FTA
- [ ] Phase 6 — ETA
- [ ] 🛑 Core complete
- [ ] Phase 7 — Preventive Maintenance
- [ ] Enhancement — Bow-Tie + ALARP
- [ ] Enhancement — Bayesian Updating
- [ ] Enhancement — Reliability Growth
- [ ] Enhancement — Survival / Cox PH
- [ ] Enhancement — Dashboard

## Data Sources

See `DATA.md` for exact dataset provenance, download instructions, and citation requirements.

## Setup

See `phase-0` notes below, or run:
\`\`\`
python -m venv venv
source venv/bin/activate   # or venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
python test_environment.py
\`\`\`