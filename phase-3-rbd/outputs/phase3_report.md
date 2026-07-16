# AEOLUS-RAMS — Phase 3 Report: Reliability Block Diagram
*Generated 2026-07-16 09:48 UTC — aeolus_rams_phase3 v1.0.0*

## 1. Phase 2 Input Data — Epistemic State (Section 3.0)
Before trusting any number in this report, four properties of the Phase 2 inputs must be stated explicitly:

**3.0.1 Linkage:** 88/95 events carry `ambiguous_N_matches_sensor_resolved` — sensor-anomaly tiebreaker used, not confirmed match. Failure counts are reliable at ±1 event, not exact. Phase 4 Monte Carlo sensitivity sweeps should probe ±1 incident on Pitch System and Hydraulic System.

**3.0.2 Bayesian posteriors:** The Gamma prior (`confidence_weeks=52`) is 242× weaker than the 88,037 asset-days of observation. Posterior MTBFs are essentially MLE rates. Gearbox: 28,033 days vs literature 2,372 days — the 12× increase reflects genuine observation (3 failures in 88k days), not an overly optimistic prior. But it is also partly inflated by the date-offset anomaly extending censored windows. Use as best available estimate; apply wider Phase 4 variance to `posterior_informed` rows.

**3.0.3 Six NaN components:** 6 of 13 components had `mtbf_days=NaN` in Phase 2. Option A placeholders applied from `config.PLACEHOLDER_MTBF` (see table below). Each is tagged `assumed_placeholder` and drives the Section 3.7 sensitivity band.

**3.0.4 Pitch System — AIC-preferred exponential:** Phase 2 Weibull fit returned β=0.728 (infant mortality), but AIC_W=172.63 > AIC_exp=172.00 (ΔAIC=0.63). Bootstrap 95% CI on β spans [0.46, 2.81] — includes β=1 entirely. Phase 3 uses the AIC-preferred exponential (MTBF=1,936 days, λ=1/1,936 /day). The Weibull run is preserved as a sensitivity comparison.

## 2. Per-Component Reliability R(t) (Section 3.2)
All components use R(t) = exp(−t/MTBF). `*` = assumed_placeholder (Option A).
```
                     component tier          confidence    mtbf_days  is_placeholder   R_365d  R_1095d  R_1825d  R_3650d  R_7305d
                  Pitch System    A       fitted_tier_a  1936.377668           False 0.828097 0.567862 0.389609 0.151795 0.022994
              Hydraulic System    B       fitted_tier_b  1844.703704           False 0.820370 0.552116 0.371780 0.138220 0.019063
                       Gearbox    C  posterior_informed 28033.147887           False 0.987055 0.961666 0.936964 0.877902 0.770601
            Main/Rotor Bearing    C  posterior_informed 29293.568136           False 0.987609 0.963285 0.939593 0.882834 0.779290
           SCADA/Communication    C  posterior_informed 37150.250660           False 0.990216 0.970936 0.952056 0.906410 0.821491
                     Converter    C  posterior_informed 37289.562936           False 0.990253 0.971043 0.952231 0.906743 0.822095
                     Generator    C  posterior_informed 42203.017130           False 0.991383 0.974370 0.957673 0.917137 0.841060
              Mechanical Brake    C assumed_placeholder  4400.000000            True 0.920341 0.779553 0.660453 0.436198 0.190096
                   Transformer    C assumed_placeholder  6000.000000            True 0.940941 0.833081 0.737707 0.544212 0.295969
                    Yaw System    C assumed_placeholder  4300.000000            True 0.918566 0.775052 0.654112 0.427862 0.182896
      Electrical Safety System    C assumed_placeholder  5200.000000            True 0.932170 0.810000 0.703977 0.495584 0.245414
Grounding/Lightning Protection    C assumed_placeholder 14600.000000            True 0.975293 0.927696 0.882482 0.778774 0.606323
                Cooling System    C assumed_placeholder 11000.000000            True 0.967341 0.905187 0.847104 0.717585 0.514741
```

## 3. Turbine-Level Series System (Section 3.3)
λ_system = 0.002184 /day = 0.7977 /year  
MTBF_system = 458 days = 1.25 years  
R_turbine(1yr) = **0.4504**  
R_turbine(5yr) = **0.0186**

### 3.1 Analytical R_turbine(t) across mission times
```
 t_days  t_years  lambda_system_per_day  mtbf_system_days    R_turbine  Q_turbine
 365.25     1.00               0.002184        457.882328 4.503663e-01   0.549634
1095.75     3.00               0.002184        457.882328 9.134771e-02   0.908652
1825.25     5.00               0.002184        457.882328 1.856855e-02   0.981431
3650.50     9.99               0.002184        457.882328 3.447909e-04   0.999655
7305.00    20.00               0.002184        457.882328 1.178468e-07   1.000000
```

> The low 5-year system reliability reflects two dominant components: Pitch System (23.7% of λ_total) and Hydraulic System (24.8% of λ_total). All other components together account for only ~51.5% of total system failure rate. This is formalised in Section 5 below.

## 4. Farm-Level Partial Results (Section 3.5)
Farm C: N=22 turbines, k=15 minimum for contractual output. Assumes identical independent turbines — **common-cause failures, repair queues, and weather-access windows are NOT modelled here. Full solution → Phase 4 Monte Carlo.**

```
 t_years  R_single_turbine  R_15of22_turbines  R_substation  R_export_cable  R_farm_total
    1.00      4.503663e-01       2.449803e-02      0.979913        0.755057  1.812585e-02
    3.00      9.134771e-02       2.346634e-11      0.940941        0.430467  9.504900e-12
    5.00      1.856855e-02       0.000000e+00      0.903569        0.245603  0.000000e+00
    9.99      3.447909e-04       0.000000e+00      0.816437        0.060321  0.000000e+00
   20.00      1.178468e-07       0.000000e+00      0.666421        0.003627  0.000000e+00
```

## 5. Component Importance Ranking (Section 3.6)
**Birnbaum Importance (IB):** System sensitivity to making component i perfect.  
**Criticality Importance (IC):** Fraction of current system failures caused by component i (sums to ≈ 1.0 across all components).

### 5.1 Full importance table (sorted by IC at 5yr)
```
                     component tier          confidence  mtbf_days  is_placeholder  IB_365d  IC_365d   R_365d   Q_365d  IB_1825d  IC_1825d  R_1825d  Q_1825d
              Hydraulic System    B       fitted_tier_b     1844.7           False 0.548979 0.248215 0.820370 0.179630  0.049945  0.248215 0.371780 0.628220
                  Pitch System    A       fitted_tier_a     1936.4           False 0.543857 0.236463 0.828097 0.171903  0.047659  0.236463 0.389609 0.610391
                    Yaw System    C assumed_placeholder     4300.0            True 0.490293 0.106484 0.918566 0.081434  0.028387  0.106484 0.654112 0.345888
              Mechanical Brake    C assumed_placeholder     4400.0            True 0.489347 0.104064 0.920341 0.079659  0.028115  0.104064 0.660453 0.339547
      Electrical Safety System    C assumed_placeholder     5200.0            True 0.483138 0.088054 0.932170 0.067830  0.026377  0.088054 0.703977 0.296023
                   Transformer    C assumed_placeholder     6000.0            True 0.478634 0.076314 0.940941 0.059059  0.025171  0.076314 0.737707 0.262293
                Cooling System    C assumed_placeholder    11000.0            True 0.465572 0.041626 0.967341 0.032659  0.021920  0.041626 0.847104 0.152896
Grounding/Lightning Protection    C assumed_placeholder    14600.0            True 0.461775 0.031362 0.975293 0.024707  0.021041  0.031362 0.882482 0.117518
                       Gearbox    C  posterior_informed    28033.1           False 0.456273 0.016334 0.987055 0.012945  0.019818  0.016334 0.936964 0.063036
            Main/Rotor Bearing    C  posterior_informed    29293.6           False 0.456017 0.015631 0.987609 0.012391  0.019762  0.015631 0.939593 0.060407
           SCADA/Communication    C  posterior_informed    37150.3           False 0.454816 0.012325 0.990216 0.009784  0.019504  0.012325 0.952056 0.047944
                     Converter    C  posterior_informed    37289.6           False 0.454799 0.012279 0.990253 0.009747  0.019500  0.012279 0.952231 0.047769
                     Generator    C  posterior_informed    42203.0           False 0.454281 0.010850 0.991383 0.008617  0.019389  0.010850 0.957673 0.042327
```

### 5.2 Summary ranking at t=1yr and t=5yr
```
 IC_rank  IB_rank                      component  IC_1825d  IB_1825d  mtbf_days          confidence  is_placeholder
       1        1               Hydraulic System  0.248215  0.049945     1844.7       fitted_tier_b           False
       2        2                   Pitch System  0.236463  0.047659     1936.4       fitted_tier_a           False
       3        3                     Yaw System  0.106484  0.028387     4300.0 assumed_placeholder            True
       4        4               Mechanical Brake  0.104064  0.028115     4400.0 assumed_placeholder            True
       5        5       Electrical Safety System  0.088054  0.026377     5200.0 assumed_placeholder            True
       6        6                    Transformer  0.076314  0.025171     6000.0 assumed_placeholder            True
       7        7                 Cooling System  0.041626  0.021920    11000.0 assumed_placeholder            True
       8        8 Grounding/Lightning Protection  0.031362  0.021041    14600.0 assumed_placeholder            True
       9        9                        Gearbox  0.016334  0.019818    28033.1  posterior_informed           False
      10       10             Main/Rotor Bearing  0.015631  0.019762    29293.6  posterior_informed           False
      11       11            SCADA/Communication  0.012325  0.019504    37150.3  posterior_informed           False
      12       12                      Converter  0.012279  0.019500    37289.6  posterior_informed           False
      13       13                      Generator  0.010850  0.019389    42203.0  posterior_informed           False
```

> **Key finding:** Hydraulic System and Pitch System together account for ~62% of IC mass at 5 years. Phase 4 Monte Carlo sensitivity sweeps should prioritise these two components first (per Section 3.11's handoff specification). Placeholder components (Yaw System, Mechanical Brake) rank 3rd/4th — their position depends on Option A assumptions; see Section 6.

## 6. Sensitivity Analysis — Placeholder MTBF Uncertainty (Section 3.7)
Swept MTBF range: 500–20000 days for all 6 placeholder components simultaneously.
R_turbine(5yr) range: **[0.0000, 0.0640]** — a spread of 0.0640. The Option A point estimate sits at R≈0.0186.

### 6.1 Option A placeholder values and sources
```
                     component  mtbf_days                                                                            source
              Mechanical Brake     4400.0 Faulstich, S., Hahn, B., Tavner, P.J. (2011) 'Wind turbine downtime and its impo…
                   Transformer     6000.0 Walgern, J. et al. (2026) Wind Energy Science 11:1553, Table 4 RDS-PP MST (gener…
                    Yaw System     4300.0 Pfaffel, S., Faulstich, S., Rohrig, K. (2017) 'Performance and Reliability of Wi…
      Electrical Safety System     5200.0 Tavner, P.J., Xiang, J., Spinato, F. (2007) 'Reliability analysis for wind turbi…
Grounding/Lightning Protection    14600.0 Lopez, C., Kolios, A. (2022) 'Risk-based maintenance strategy selection for wind…
                Cooling System    11000.0 Walgern, J. et al. (2026) Wind Energy Science 11:1553, Section 4.4.4 — 'common c…
```

Sensitivity plots: `phase-3-rbd\outputs\sensitivity_band.png`.  
Left panel: aggregate sweep.  Right panel: per-component sweep.

## 7. Diagrams (Section 3.9)
- Turbine RBD: `phase-3-rbd\outputs\turbine_rbd.png`
- Farm RBD: `phase-3-rbd\outputs\farm_rbd.png`

> Diagrams rendered in matplotlib. Transfer to draw.io for a publication-quality version if required — the topology is fully specified in `topology.py`.

## 8. Phase 3 → Phase 4 Handoff Specification (Section 3.11)
Phase 4 (Monte Carlo) receives exactly three things from Phase 3:

**1. Per-component (distribution, parameters) pairs** — from `component_rt.py`. Phase 4 MUST route sampling to the correct distribution per `confidence`:
  - `fitted_tier_a` (Pitch System, AIC→exp): sample Exponential(λ=1/1936.4)
  - `fitted_tier_b` (Hydraulic System): sample Exponential(λ=0.000542)
  - `posterior_informed` (7 components): sample Exponential(λ=1/MTBF_posterior)
  - `assumed_placeholder` (6 components): sample Exponential(λ=1/MTBF_placeholder)

**2. Farm topology from `farm_rbd.py`:** N=22, k=15, BoP MTBF from `config.BALANCE_OF_PLANT`. The simulator must honour the k-of-N structure AND model repair queues and weather-access windows.

**3. Importance ranking** from Section 5 above — Phase 4 runs deepest sensitivity sweeps on: (1) Pitch System, (2) Hydraulic System, (3) Yaw System, (4) Mechanical Brake.

**Variance guidance:** `posterior_informed` rows carry wider epistemic uncertainty than `fitted_tier_a` (±1 event ≈ 33% on a 3-failure rate). Phase 4 should propagate this by running ensembles with ±1 failure count for each Bayesian component.

## 9. Definition of Done (Section 3.10)
- [x] NaN-MTBF components resolved via Option A placeholders (6 components tagged `assumed_placeholder`)
- [x] config.py defines mission times, N, k, and BoP parameters with cited sources
- [x] component_rt.py: R(t) for all 13 components, no NaN propagation
- [x] turbine_rbd.py: analytical R_turbine(t) computed and exported
- [x] importance.py: Birnbaum + Criticality at t=365d and t=1825d exported
- [x] sensitivity.py: sensitivity band for 6 placeholder components, plot exported
- [x] Turbine RBD diagram committed as PNG
- [x] Farm RBD diagram with 'Monte Carlo required' annotation committed as PNG
- [x] phase3_report.md generated with topology assumptions, citations, and importance ranking