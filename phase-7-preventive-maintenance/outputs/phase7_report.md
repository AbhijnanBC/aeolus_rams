# AEOLUS-RAMS — Phase 7 Report: Preventive Maintenance & RCM
*Generated 2026-08-05 02:26 UTC — aeolus_rams_phase7 v1.0.0*

## 0. Executive Summary

Phase 7 converts six phases of characterisation into **concrete, mathematically justified maintenance decisions** for all 13 components. Three distinct engineering findings drive three distinct maintenance categories:

**1. Age-Replacement (Barlow-Proschan):** Main/Rotor Bearing T* = 23.0 yr (savings 46.3% vs. run-to-failure). Gearbox T* = 22.3 yr (savings 39.4%). Generator T* > 25 yr design life → CBM preferred.

**2. Failure-Finding (IEC 61511):** G3b Safety Relay requires proof-testing at **τ ≤ 26 days (0.9 months) to maintain SIL-2 (PFDavg ≤ 10⁻³).** Current state: PFDavg = 0.0271 (implicitly τ=730d, no explicit proof-test plan). Without monthly proof-testing, Phase 6's ALARP Option 1 SIL-2 relay replacement provides no lasting SIL-2 benefit.

**3. Condition-Based Maintenance:** Pitch System (β=0.7285 < 1, Phase 2). No finite optimal replacement age exists — C(T) is monotonically decreasing. The ML anomaly score from the SCADA pipeline IS the Phase 7 maintenance trigger. Trigger rule: score ≥ μ_rolling + 2σ_rolling (30-day baseline).

## 1. Phase 1–6 Inherited Inputs

| Phase | Key Input | Phase 7 Use |
|---|---|---|
| Phase 1 | FMECA RPN: Bearing=480, Gearbox=324, Pitch=360 | Confirms severity for age-replacement + CBM priority |
| Phase 2 | Pitch β=0.7285 < 1 (Tier A, 10 TBF) | CBM is the only optimal strategy for Pitch System |
| Phase 2 | Bearing/Gearbox: insufficient TBF → literature β | Justifies use of literature Weibull for age-replacement |
| Phase 3 | λ_system=0.002184/d, MTBF=457.9d | Baseline for proportional hazards CBM model |
| Phase 4 | A_farm≈0.963, export cable dominant | Confirms BoP risk is separate from turbine PM scope |
| Phase 5 | G3b.Q=0.027132 (FV rank 1=0.822) | Input to failure-finding λ_G3b derivation |
| Phase 6 | ALARP Option 1: G3b→SIL-2 (79.1% Q_top reduction) | Phase 7 derives τ_required to maintain SIL-2 |

## 2. Mathematical Framework

### 2.1 Barlow-Proschan Age-Replacement

For a component with Weibull reliability R(t) = exp(-(t/η)^β), the long-run expected cost rate under age-replacement at interval T is:

```
C(T) = [Cp·R(T) + Cf·(1-R(T))] / ∫₀ᵀ R(t)dt
```

**T* exists if and only if β > 1.** For β ≤ 1, C(T) is monotonically decreasing — no minimum exists, and PM at any fixed interval is suboptimal. This is the mathematical proof that calendar-based PM on the Pitch System is wrong.

### 2.2 IEC 61511 Failure-Finding PFDavg

For a dormant device with constant failure rate λ proof-tested at interval τ:

```
PFDavg = 1 - [1 - exp(-λτ)] / (λτ)    [exact, IEC 61508-6 Eq. B.4]
       ≈ λτ / 2                         [first-order, valid for λτ < 0.1]
```

> **Note:** A plan document circulated earlier incorrectly omitted the leading `1 -`. The correct formula is implemented and verified: PFDavg_correct(λ_G3b=7.571e-05/d, τ=730d) = 0.027132 = Q_G3b = 0.027132 ✓

## 3. Age-Replacement Results

| component          |   beta |   mttf_years |   cp_cf_ratio |   T_star_years |   C_star_per_day |   C_no_PM_per_day |   savings_fraction | exceeds_design_life   |
|:-------------------|-------:|-------------:|--------------:|---------------:|-----------------:|------------------:|-------------------:|:----------------------|
| Main/Rotor Bearing | 2.0000 |      60.6590 |        0.1000 |        23.0290 |          36.3370 |           67.7030 |             0.4630 | False                 |
| Gearbox            | 1.8000 |      58.4340 |        0.1000 |        22.2520 |          56.7710 |           93.7080 |             0.3940 | False                 |
| Generator          | 1.5000 |      93.9200 |        0.1500 |        54.5470 |          19.4360 |           23.3210 |             0.1670 | True                  |

![Figure 1 — Cost-Rate Curves](outputs\cost_rate_curves.png)

> **Generator finding:** T* = ~54yr >> 25yr design life. The Barlow-Proschan analysis confirms that age-replacement at a specific calendar interval is not economically viable for the Generator. Vibration monitoring + insulation resistance testing is the preferred strategy.

## 4. Failure-Finding Results — G3b Safety Relay

**Input from Phase 5:** Q_G3b = 0.027132 (gate_Q_table.csv)
**Derived:** λ_G3b = 7.5710e-05 /day (from PFDavg_exact at τ_implied=730d)
**Current PFDavg:** 0.0271 (no explicit proof-test plan → effectively <SIL-1)
**SIL-2 requires:** PFDavg ≤ 1e-03
**Required τ_max:** **26.4 days (0.9 months) — implement monthly proof-test**

**Operational implication:** After Phase 6 ALARP Option 1 (replace with SIL-2 relay, Q_G3b → 10⁻³), the SIL-2 rating is only maintained if the relay is proof-tested at intervals ≤ 26 days. Without proof-testing, latent faults accumulate and the PFDavg drifts back toward the current unsafe value within months.

![Figure 2 — PFDavg vs. Proof-Test Interval](outputs\pfd_curve.png)

|   tau_days |   tau_months |   PFDavg_exact |   PFDavg_approx | SIL_achieved   |
|-----------:|-------------:|---------------:|----------------:|:---------------|
|     7.0000 |       0.2000 |         0.0003 |          0.0003 | SIL-2          |
|    14.0000 |       0.5000 |         0.0005 |          0.0005 | SIL-2          |
|    26.0000 |       0.9000 |         0.0010 |          0.0010 | SIL-2          |
|    27.0000 |       0.9000 |         0.0010 |          0.0010 | SIL-1          |
|    30.0000 |       1.0000 |         0.0011 |          0.0011 | SIL-1          |
|    30.0000 |       1.0000 |         0.0011 |          0.0011 | SIL-1          |
|   365.0000 |      12.0000 |         0.0137 |          0.0138 | BELOW SIL-1    |
|   365.0000 |      12.0000 |         0.0137 |          0.0138 | BELOW SIL-1    |

## 5. Condition-Based Maintenance — Pitch System

**Phase 2 finding:** β = 0.7285 < 1. Weibull hazard rate h(t) = (β/η)(t/η)^(β−1) is **decreasing** with age. The component is not wearing out — it is in an infant-mortality or random failure regime. Barlow-Proschan C(T) is monotonically decreasing (see Panel D, Figure 1): the longer you run without PM, the lower the cost rate. No calendar-based PM interval is ever cost-optimal.

**CBM trigger (Phase 7 starting rule):**  Trigger maintenance when anomaly_score(t) ≥ rolling_mean(30d) + 2σ_rolling.  This is the Proportional Hazards CBM framework (Jardine et al. 2006) adapted for the available SCADA anomaly score signal.  **Tuning:** k=2.0 is the starting threshold. Collect operational data on score-to-failure time to calibrate γ (PH coefficient) and the optimal k*.

## 6. Complete RCM Maintenance Schedule

| component                                   | rcm_category    | maintenance_interval_label   | trigger_rule                                                                                                              |
|:--------------------------------------------|:----------------|:-----------------------------|:--------------------------------------------------------------------------------------------------------------------------|
| Main/Rotor Bearing                          | AGE_REPLACEMENT | 23.0 yr                      | Replace at T* = 23.0 yr (Barlow-Proschan optimal, β=2.0, Cp/Cf=0.10). Savings vs run-to-failure: 46.3%.…                  |
| Gearbox                                     | AGE_REPLACEMENT | 22.3 yr                      | Replace at T* = 22.3 yr (Barlow-Proschan optimal, β=1.8, Cp/Cf=0.10). Savings vs run-to-failure: 39.4%.…                  |
| Generator                                   | CONDITION_BASED | On-condition                 | T* = 54.5 yr > design life (25 yr). Condition-based monitoring preferred: vibration analysis (accelerometer), oil debris… |
| Electrical Safety System (G3b Safety Relay) | FAILURE_FINDING | 26 d (0.9 months)            | Proof-test at τ ≤ 26 d (IEC 61511 SIL-2 requirement). Derived: λ_G3b = 7.571e-05/d from Q_G3b=0.0271 at τ_implied=730d. … |
| Pitch System                                | CONDITION_BASED | On-condition                 | ML anomaly score ≥ μ_rolling + 2σ_rolling (30-day rolling baseline). β = 0.7285 < 1 (Phase 2 Tier A Weibull MLE fit, 10 … |
| Hydraulic System                            | TIME_DIRECTED   | 18 months                    | Hydraulic fluid replacement + pump/filter inspection at 18-month intervals. Sample oil at 9 months to check viscosity/co… |
| Mechanical Brake                            | TIME_DIRECTED   | 24 months                    | Brake disc measurement + caliper overhaul + 24VAC supply verification at 24-month intervals. Replace disc if thickness <… |
| Transformer                                 | CONDITION_BASED | 12 months (annual DGA)       | Annual dissolved gas analysis (DGA) of insulating oil. Maintenance triggered if key gas thresholds per IEC 60599 are exc… |
| Yaw System                                  | TIME_DIRECTED   | 12 months                    | Yaw bearing grease replenishment + slew ring gap inspection + yaw motor current logging annually.…                        |
| SCADA/Communication                         | RUN_TO_FAILURE  | On failure                   | Replace BK1120/NC300 module on failure. Maintain minimum 1 spare module on-site. Monitor communication health via SCADA … |
| Converter                                   | RUN_TO_FAILURE  | On failure                   | Replace power electronics module on failure. Monitor IGBT junction temperature via SCADA thermal alarms. Keep 1 replacem… |
| Cooling System                              | TIME_DIRECTED   | 24 months                    | Coolant flush + heat exchanger inspection + pump bearing check at 24-month intervals. Verify coolant inhibitor concentra… |
| Grounding/Lightning Protection              | TIME_DIRECTED   | 12 months                    | Visual inspection of grounding brushes + earth continuity test (resistance < 1Ω) annually. Check lightning receptor tip … |

![Figure 3 — RCM Summary Chart](outputs\rcm_summary_chart.png)

## 7. Assumptions and Limitations

**Literature Weibull parameters (Tier C):** β and η for Bearing, Gearbox, Generator are from published studies. They are NOT fitted to the AEOLUS dataset (insufficient TBF observations — 0 direct failures per Phase 2 tiering). T* values are illustrative of the optimal-PM framework; specific numeric T* depends on the assumed β/η. Sensitivity figure shows T* is robust to ±50% variation in Cp/Cf.

**Exponential failure model for dormant devices:** PFDavg_exact assumes constant failure rate (exponential) for G3b. If the relay degrades with age (β>1), PFDavg increases faster than the formula predicts and a shorter τ would be required. IEC 61508 assumes exponential for electronic hardware; for electromechanical relays this is conservative.

**100% proof-test effectiveness:** The failure-finding formula assumes that every proof-test detects and repairs all latent faults. Partial-coverage tests (e.g., only testing one relay channel) require the imperfect-testing extension: PFDavg_adjusted = PFDavg_exact / coverage.

**CBM γ parameter unknown:** The Proportional Hazards coefficient γ linking anomaly score to hazard uplift has not been estimated (requires historical degradation-to-failure data). k=2σ is a starting threshold; the cost-optimal k* must be tuned with operational experience. Phase 8+ should collect score-to-failure time series.

**Age-replacement costs (USD) are ballpark estimates:** Replacement costs are from the NREL offshore cost model (2020). Actual costs depend on vessel availability, weather, supply-chain, and specific turbine model. The cost-ratio sensitivity plot (Figure 4) shows that T* is robust to ±50% cost variation for β=2.0 and β=1.8.

## 8. Phase 7 → Phase 8 Handoff

Phase 8 (Spare Parts Optimisation) takes Phase 7's maintenance schedule as input to determine optimal inventory levels for each component category:

**Age-replacement components:** Bearing replacement at T*=23yr, Gearbox at T*=22yr — planned procurement with 12-month lead time acceptable. Spare part buffer: 1 unit each (low demand rate, high unit cost).
**Time-directed components:** Hydraulic fluid, brake discs, coolant — consumables with predictable demand. Reorder at safety stock = mean demand × lead time + z×σ.
**Failure-finding (G3b):** Monthly proof-test → 12 tests/year. If test reveals failure, replace SIL-2 relay module (MTTR=2d with on-site spare). Maintain 1 spare relay module on-site at all times.
**CBM (Pitch System):** Triggered maintenance → stochastic demand. Poisson approximation: expected triggers ≈ 2–4/year/turbine based on k=2σ rule applied to typical SCADA anomaly score distributions.

## 9. Definition of Done

- [x] config.py: literature Weibull (β, η) for Bearing, Gearbox, Generator with full citation
- [x] config.py: replacement costs (Cp, Cf) with NREL source for all AR components
- [x] age_replacement.py: cost_rate() and optimal_replacement_age() correct per Barlow-Proschan
- [x] age_replacement.py: β≤1 guard returns T*=None with explanatory warning
- [x] failure_finding.py: PFDavg_exact() uses CORRECT IEC 61508 formula (verified by test)
- [x] failure_finding.py: derive_lambda_from_Q() verified: PFDavg(λ_G3b, 730d) = Q_G3b
- [x] τ_SIL2 ≈ 26d (0.9 months) committed to failure_finding_table.csv
- [x] cbm_trigger.py: rolling z-score trigger function with sensitivity sweep
- [x] rcm_schedule.csv: all 13 components with category, interval, trigger rule, data source
- [x] age_replacement_table.csv: T*, C*, savings for Bearing, Gearbox, Generator
- [x] cost_ratio_sensitivity.csv: T* vs Cp/Cf sweep [0.01, 0.50] for all AR components
- [x] cost_rate_curves.png: C(T) for 3 AR components + Pitch β<1 illustration (4-panel)
- [x] pfd_curve.png: PFDavg vs. τ with SIL-1/2/3 lines and τ_SIL2 annotated
- [x] rcm_summary_chart.png: all 13 components on one visual
- [x] phase7_report.md: includes formula correction note, T*>design-life finding, τ_required, CBM framework