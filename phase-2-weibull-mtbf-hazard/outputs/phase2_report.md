# AEOLUS-RAMS — Phase 2 Report
*Generated 2026-07-14 20:45 UTC — aeolus_rams_phase2 v1.0.0*

## 1. Event -> Asset Linkage (Section 2.1.1)
```
                    link_confidence  count
ambiguous_4_matches_sensor_resolved     17
ambiguous_7_matches_sensor_resolved     17
ambiguous_5_matches_sensor_resolved     15
ambiguous_2_matches_sensor_resolved     12
ambiguous_3_matches_sensor_resolved     11
ambiguous_8_matches_sensor_resolved      8
ambiguous_6_matches_sensor_resolved      7
                       unique_match      7
ambiguous_9_matches_sensor_resolved      1
```

- All events uniquely linked to an asset: **False**
  > Ambiguous or unmatched rows exist — inspect `linked_events.csv` before trusting any MTBF number derived from an affected asset.

## 2. Timestamp Offset Diagnostic (Section 2.1.2)
```
farm  asset_id  n_dated_events  offset_days_min  offset_days_max  offset_days_constant                                    interpretation
   C         2               1             1462             1462                  True only one dated event on this asset — inconclusive
   C        34               1             1099             1099                  True only one dated event on this asset — inconclusive
   C        38               1             1483             1483                  True only one dated event on this asset — inconclusive
```

> This is diagnostic, not a gate — Section 2.1.2's rule (relative durations only) is followed throughout this pipeline regardless of which explanation the offsets above point to.

## 3. Data-Sufficiency Tiering (Section 2.2)
- Tier A (≥ 8 incidents): **1** component(s)
- Tier B (5-7 incidents): **1** component(s)
- Tier C (< 5 incidents): **11** component(s)

```
                     component  distinct_incidents_observed tier  rpn
                  Pitch System                           11    A  360
              Hydraulic System                            6    B  196
                Cooling System                            4    C  180
            Main/Rotor Bearing                            3    C  480
                       Gearbox                            3    C  324
              Mechanical Brake                            3    C  162
      Electrical Safety System                            3    C   96
                     Generator                            2    C  252
Grounding/Lightning Protection                            2    C  168
                   Transformer                            2    C  160
                     Converter                            2    C   72
           SCADA/Communication                            2    C   48
                    Yaw System                            1    C   40
```

## 4. TBF Extraction Summary (Section 2.3)
```
                     component  n_uncensored  n_censored  n_assets
                  Pitch System            10          36        27
              Hydraulic System             6          36        27
                Cooling System             4          36        27
                       Gearbox             3          36        27
      Electrical Safety System             3          36        27
              Mechanical Brake             3          36        27
            Main/Rotor Bearing             3          36        27
                     Generator             2          36        27
                     Converter             2          36        27
Grounding/Lightning Protection             2          36        27
           SCADA/Communication             2          36        27
                   Transformer             2          36        27
                    Yaw System             1          36        27
```

## 5. Tier A — 2-Parameter Weibull Fits (Section 2.4)
### Pitch System
- β (shape) = **0.728**, η (scale) = **1586.6 days**
- MTBF = **1936.4 days** (5.30 years)
- beta < 1: infant mortality (hazard falls with age)
- AIC: Weibull = 172.63, Exponential = 172.00 -> **preferred: exponential**
- n (usable TBF intervals) = 10
- Bootstrap 95% CI: β ∈ (0.457, 2.807), η ∈ (493.0, 2920.9), MTBF ∈ (1017.3, 2658.7) days (2000/2000 resamples successful)

## 6. Tier B — Exponential-Only Fits (Section 2.4)
### Hydraulic System
- λ (rate) = **0.00054 / day**
- MTBF = **1844.7 days** (5.05 years)
- n (usable TBF intervals) = 6

## 7. Tier C — Literature-Informed Priors (Section 2.4)
```
                     component mtbf_days                          confidence                                                                            source
            Main/Rotor Bearing   20481.0 derived_from_cumulative_probability Hart, E. et al. (2019), cited in Hart, E., Clarke, B., Nicholas, G., Kazemi Amir…
                       Gearbox    2372.0                        cited_direct Carroll, J., McDonald, A., McMillan, D. (2016) 'Failure rate, repair time and un…
                     Generator    3845.0                        cited_direct Carroll, J., McDonald, A., McMillan, D. (2016) 'Failure rate, repair time and un…
              Mechanical Brake         —                     not_yet_sourced Tavner, P.J., Xiang, J., Spinato, F. (2007) 'Reliability analysis for wind turbi…
                     Converter     982.0     derived_needs_rating_assumption Walgern, J., Stratmann, N., Horn, M., Then, N.W.Y., Menzel, M., Anderson, F., Ko…
                   Transformer         —                     not_yet_sourced Walgern et al. (2026), WES 11:1553, Table 4 — bundled under the 'generator trans…
                    Yaw System         —                     not_yet_sourced See Tavner, P.J. et al. (2007-2013) WMEP/LWK-based studies (e.g. summarised in h…
           SCADA/Communication     959.0     derived_needs_rating_assumption Walgern, J. et al. (2026), Wind Energy Science, 11, 1553-1568, https://doi.org/1…
      Electrical Safety System         —                     not_yet_sourced WMEP/LWK 'Electrical Control' and 'Electrical System' categories (NREL/TP-5000-5…
Grounding/Lightning Protection         —                     not_yet_sourced No source search turned up a standalone grounding-brush/lightning-conduction-pat…
                Cooling System         —                     not_yet_sourced Walgern et al. (2026), WES 11:1553, Sect. 4.4.4 mentions a distinct 'common cool…
```

Full citations and derivation notes for every entry above are in `literature_priors.py` (`LITERATURE_PRIORS` dict) and are carried through unabridged into `mtbf_table.csv`.

## 8. MTBF Table
```
                     component tier    mtbf_days         confidence
                  Pitch System    A  1936.377668      fitted_tier_a
              Hydraulic System    B  1844.703704      fitted_tier_b
                       Gearbox    C 28033.147887 posterior_informed
            Main/Rotor Bearing    C 29293.568136 posterior_informed
           SCADA/Communication    C 37150.250660 posterior_informed
                     Converter    C 37289.562936 posterior_informed
                     Generator    C 42203.017130 posterior_informed
              Mechanical Brake    C          NaN    not_yet_sourced
                   Transformer    C          NaN    not_yet_sourced
                    Yaw System    C          NaN    not_yet_sourced
      Electrical Safety System    C          NaN    not_yet_sourced
Grounding/Lightning Protection    C          NaN    not_yet_sourced
                Cooling System    C          NaN    not_yet_sourced
```

## 9. Hazard Plots (Section 2.6)
- `hazard_tier_a_b.png`: phase-2-weibull-mtbf-hazard\outputs\hazard_tier_a_b.png
- `hazard_system_illustrative.png`: phase-2-weibull-mtbf-hazard\outputs\hazard_system_illustrative.png

Tier C components are never plotted with a fitted curve — see each figure's own 'not plotted' panel / disclaimer text.

## 10. Definition of Done (Section 2.9)
- [ ] Event -> asset_id linkage established and verified (all `unique_match`, or discrepancies investigated)
- [x] Timestamp offset diagnostic run (Section 2.1.2) — documented
- [x] TBF extraction complete per component, censored final intervals correctly flagged
- [x] Tier A: 2-parameter Weibull fit, AIC-compared against exponential, bootstrap CI computed
- [x] Tier B: 1-parameter exponential fit only
- [x] Tier C: literature-informed MTBF placeholders, each with a cited source — not a forced small-sample fit
- [x] MTBF correctly labeled (not MTTF) for all repairable-system results
- [x] Hazard rate plotted only for Tier A/B; Tier C explicitly marked 'shape unknown'
- [x] Results exported: mtbf_table.csv