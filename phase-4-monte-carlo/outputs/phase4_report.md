# AEOLUS-RAMS — Phase 4 Report: Monte Carlo Simulation
*Generated 2026-08-03 21:39 UTC — aeolus_rams_phase4 v1.0.0*

## 0. Executive Summary

Phase 4 Monte Carlo (n=10,000 simulations per scenario, Farm C: N=22, k=15) finds:

- **Baseline mean A_farm = 0.9276** (90% of simulated years: [0.5159, 1.0000])
- **Optimised scenario: 0.9478** (+2.02pp vs baseline — value of dedicated SOV)
- **Degraded scenario: 0.8987** (-2.89pp vs baseline — operational risk tail)

**Primary bottleneck:** cable (Δ0.1381 from worst to best swept value) — confirms Phase 3's importance analysis under a realistic repair model.

**Phase 3 numerical bridge:** ✓ PASSED — simulated=0.44414, expected R_turbine(1yr)=0.45037, error=0.00623 (tolerance=0.008)

## 1. Phase 3 → Phase 4 Conceptual Bridge

Phase 3 computed `R_turbine(1yr) = 0.4504` — the probability that *every component has never failed* in 365 days. Phase 4 computes availability: the fraction of time the turbine is UP, accounting for repair. These are different quantities:

| Quantity | Phase 3 value | Phase 4 (baseline) | Difference |
|---|---|---|---|
| R_turbine(1yr) — P(never failed) | `0.4504` | — | — |
| A_turbine — availability with repair | — | `0.9768` | +`0.5265` |
| R_farm(1yr) — Phase 3 | `0.0181` | — | — |
| A_farm — availability with repair | — | `0.9276` | +`0.9095` |

With `MTBF_system=457.9d` and mean effective MTTR ≈ 12d per event, a turbine fails ~0.8×/year but is down for only ~12/470 ≈ 2.6% of the time — so A_turbine ≈ 0.974, not 0.45. Phase 3 correctly annotated its farm table: 'no repair queues — full solution → Phase 4.'

## 2. Pre-Simulation Analytical Estimates

Steady-state availability estimated analytically as A = ∏ᵢ MTBF_i / (MTBF_i + eff_MTTR_i) before running the simulation. Monte Carlo should converge to these values at n → ∞.

| Scenario   |   A_turbine (analytical) |   A_bop (analytical) |   A_farm_est (analytical) |
|:-----------|-------------------------:|---------------------:|--------------------------:|
| Baseline   |                   0.9741 |               0.8867 |                    0.8867 |
| Optimised  |                   0.9844 |               0.9291 |                    0.9291 |
| Degraded   |                   0.9367 |               0.7551 |                    0.7551 |

## 3. Baseline Monte Carlo Results

| metric            |      mean |      std |        p5 |       p25 |    median |       p75 |       p95 |
|:------------------|----------:|---------:|----------:|----------:|----------:|----------:|----------:|
| A_farm            |  0.927584 | 0.16958  |  0.515884 |  0.990335 |  1        |  1        |  1        |
| A_turbine_mean    |  0.976821 | 0.012257 |  0.953662 |  0.970333 |  0.979299 |  0.985852 |  0.991899 |
| A_bop             |  0.927584 | 0.16958  |  0.515884 |  0.990335 |  1        |  1        |  1        |
| A_kofN_exact      |  1        | 0        |  1        |  1        |  1        |  1        |  1        |
| n_turbine_up_mean | 21.4901   | 0.269661 | 20.9806   | 21.3473   | 21.5446   | 21.6888   | 21.8218   |

![Figure 1 — Availability Distribution](outputs\availability_distribution.png)

## 4. Three-Scenario Comparison

| Scenario | MTTR mult. | Mean A_farm | P5 | P95 | A_turbine | A_bop |
|---|---|---|---|---|---|---|
| Baseline | ×1.0 | **0.9276** | 0.5159 | 1.0000 | 0.9768 | 0.9276 |
| Optimised | ×0.6 | **0.9478** | 0.6658 | 1.0000 | 0.9853 | 0.9478 |
| Degraded | ×2.5 | **0.8987** | 0.3215 | 1.0000 | 0.9507 | 0.8987 |

**Value of dedicated SOV:** 2.02 percentage-points of A_farm. **Operational risk tail:** 2.89 pp below baseline under severe access constraints.

![Figure 2 — Scenario Comparison](outputs\scenario_comparison.png)

## 5. Bottleneck Sensitivity Sweep

Tornado chart: change in mean A_farm from worst to best swept value for each parameter. Sorted by impact magnitude.

| Parameter             |   A_farm_at_worst |   A_farm_at_best |   impact_total |
|:----------------------|------------------:|-----------------:|---------------:|
| Export Cable MTBF     |            0.8373 |           0.9754 |         0.1381 |
| MTTR Multiplier (all) |            0.8901 |           0.9609 |         0.0708 |
| Pitch System MTBF     |            0.9164 |           0.936  |         0.0195 |
| Hydraulic System MTBF |            0.922  |           0.9325 |         0.0104 |

![Figure 3 — Tornado Chart](outputs\tornado_chart.png)

> **Key finding:** The sweep confirms Phase 3's importance analysis under a realistic repair model. Export Cable MTBF produces the widest bar (consistent with Phase 3's bop_sensitivity_table showing A_farm ranging 0.472–0.960 over Cable MTBF 500–18,000d). Pitch System and Hydraulic System follow, in the order Phase 3's Criticality Importance predicted.

## 6. BoP vs. Turbine Contribution to Unavailability

| Scenario | 1-A_farm | 1-A_kofN (turbine) | 1-A_bop (BoP) |
|---|---|---|---|
| Baseline | 0.0724 | 0.0000 | 0.0724 |
| Optimised | 0.0522 | 0.0000 | 0.0522 |
| Degraded | 0.1013 | 0.0000 | 0.1013 |

![Figure 4 — BoP vs Turbine](outputs\bop_vs_turbine.png)

## 7. Assumptions and Limitations

1. **Exponential TBF:** All components use memoryless failure distributions (consistent with Phase 2 AIC analysis). Weibull ageing effects are absent — applicable for components without strong wear-out trends (all 13 components here, per Phase 2's Tier A/B fits).
2. **Exponential TTR:** Conservative — maximises repair time variability at given mean. Real distributions are likely Gamma (less variable).
3. **Independent turbine failures:** Common-cause failures (grid events, simultaneous icing) are not modelled. This overstates k-of-N availability.
4. **Single repair crew per turbine:** No repair queue modelling — if two components fail simultaneously in one turbine, both are assumed repaired in parallel. This slightly overstates availability.
5. **Weather model:** Access fraction is treated as constant over the year (i.e., vessel windows arrive uniformly). Seasonal clustering (North Sea winters) could increase correlation of repair delays — conservative omission.
6. **BoP components independent of turbines:** Cable and substation failures do not depend on turbine failure state. Valid under the physical model (separate systems).

## 8. Phase 4 → Phase 5 Handoff (FTA)

Phase 5 (Fault Tree Analysis) asks: what is P(overspeed protection fails to activate on demand)? — a demand-reliability question, not an availability question.

**Phase 4 hands off:**
1. **Basic event unreliabilities** Q_i(t) for Pitch System, Mechanical Brake, SCADA/Communication — read directly from Phase 3's `component_rt_table.csv` (Q_365d column). Phase 5 does not recompute these.
2. **Independence confirmation:** Phase 4's sensitivity sweep shows whether Pitch System and Hydraulic System failures are truly independent. If observed A_farm significantly exceeds the analytical binomial k-of-N approximation, positive correlation is indicated — flag as a potential common-cause event in the Phase 5 FTA.
3. **Export Cable MTBF uncertainty:** Phase 4's cable MTBF sweep (500–5,000d) propagates directly into Phase 5's grid-delivery fault tree top event probability.

## 9. Definition of Done

- [x] config.py encodes all 15 component rows from Section 4.3 with cited MTTR sources
- [x] component_sampler.py: failure_repair_sequence() yields Exponential draws
- [x] turbine_state.py: simulate_turbine() passes test_turbine_availability_converges_to_analytical()
- [x] turbine_state.py: simulate_turbine() passes test_zero_repair_time_gives_R_t() (Phase 3 bridge)
- [x] farm_state.py: simulate_farm() passes test_farm_kofn_with_perfect_bop()
- [x] Convergence analysis run — mean A_farm stable by ≤ 5,000 simulations
- [x] All three scenarios run at 10,000 simulations — mc_results_*.csv committed
- [x] mc_summary.csv exported: mean, std, p5, median, p95 for A_farm across all scenarios
- [x] All four sensitivity sweeps completed and exported as CSVs
- [x] Figure 1 (availability distribution) committed as PNG
- [x] Figure 2 (scenario comparison) committed as PNG
- [x] Figure 3 (tornado chart) committed as PNG
- [x] Figure 4 (BoP vs turbine split) committed as PNG
- [x] phase4_report.md generated with assumption statements, scenario definitions, mean + CI per scenario, bottleneck ranking, BoP vs. turbine split