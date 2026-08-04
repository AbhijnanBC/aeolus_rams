# AEOLUS-RAMS Phase 6 — Event Tree Analysis (ETA) Report

---

## Executive Summary

Phase 6 completes the AEOLUS-RAMS safety analysis by converting the
Phase 5 fault tree's conditional failure probabilities into **absolute
consequence frequencies**, placing them on the **IEC 61400-1 Ed.4 risk
matrix**, and delivering a complete **bow-tie diagram** and **ALARP
argument**.

**Central finding (λ_IE = 0.5/turbine/yr, independent):**

| Branch | P(branch\|IE) | Frequency | IEC 61400-1 placement |
|---|---|---|---|
| Branch 1: SAFE (controlled) | 0.8281 | 0.414/turbine/yr | Cat E × F1 — Acceptable |
| Branch 2: SAFE (brake save) | 0.1485 | 0.0742/turbine/yr | Cat D × F2 — Acceptable |
| Branch 3: NEAR-MISS (SCADA) | 0.0226 | 1.1323e-02/turbine/yr | Cat C × F3 — ALARP |
| **Branch 4: CATASTROPHIC** | **7.6900e-04** | **3.85e-04/turbine/yr** | **Cat A × F4 — ALARP** |

The **catastrophic overspeed risk (Cat A × F4) is in the ALARP region** —
not automatically acceptable. A formal ALARP case is required.

**ALARP recommendation (highest leverage):**
Option 1: Duplicate Safety Relay (G3b → SIL-2) reduces Q_top by 79% and
moves the risk from Cat A × F4 (ALARP) to
Cat A × F5 (**ACCEPTABLE**).

**Cross-phase comparison:** The BoP export cable failure (Phase 4,
Δ_availability = 5.73 pp) appears on the same risk matrix as Cat D × F1
(ALARP, economic consequence only) — fundamentally different from the
catastrophic overspeed risk. Both must be managed, but under different
ALARP frameworks (economic optimisation vs. IEC 61508 safety integrity).

---

## 1. Phase 5 Inheritance

Phase 6 reads the following values directly from `gate_Q_table.csv`
(mission time t = 365 days, one-year probability of failure on demand):

| Gate | Node name | Q(365d) | Physical meaning |
|---|---|---|---|
| G1 | Pitch System fails to feather | 0.171903 | P(primary protection fails on demand) |
| G2 | Mechanical brake fails to engage | 0.136207 | P(secondary protection fails on demand) |
| G3 | SCADA overspeed trip fails to activate | 0.032843 | P(tertiary protection fails on demand) |
| **TE** | **Catastrophic overspeed** | **7.690041e-04** | **P(all three fail simultaneously)** |

CCF uplift factor at β = 0.1 (from `ccf_sensitivity.csv`):
**1.577175×**, giving Q_top_ccf = 1.2129e-03.

**Critical note (Phase 5 subcause sensitivity finding):** The
`subcause_sensitivity.csv` showed Δ=0.0% for all sub-event fraction
combinations. This is mathematically expected — the OR gate normalises
sub-event Q values so the gate result equals the parent component Q
exactly. The gate Q values above are used with full confidence as branch
probabilities; sub-cause fractions affect only MCS ordering, not Q_top.

---

## 2. Initiating Events

### IE-1: Grid Fault / Sudden Load Rejection (primary, λ_IE_central = 0.5/yr)

A grid fault causes the turbine to lose electrical load instantly; aerodynamic
torque continues while generator torque drops to zero, causing immediate
overspeed. This is the standard design basis event for overspeed protection
per IEC 61400-1 and NORSOK Z-013.

Frequency estimate: 0.2–2.0 faults/turbine/yr (Troldborg et al. 2019,
DTU Wind Energy Report E-0183; Hau, "Wind Turbines" 3rd ed., Ch. 19).
**Central estimate used: λ_IE = 0.5/turbine/yr.**

**This is the single highest-uncertainty input in Phase 6.** All consequence
frequencies scale linearly with λ_IE. The sensitivity sweep below covers
the full [0.05, 2.0] range — results at the pessimistic end (λ_IE = 2.0)
cross into the UNACCEPTABLE region and require design change.

### IE-2: Extreme Gust Beyond Cut-Out (secondary, λ_IE = 0.05/yr)

An extreme operating gust (EOG) above the emergency-stop threshold can cause
transient overspeed. Rate: ≈0.05/turbine/yr for an IEC Class I site
(per IEC 61400-1 Ed.4 load case EWS). The event tree structure is identical
to IE-1; only λ_IE differs. IE-2 contributes ~10% of IE-1's consequence
frequency and is reported as a parallel calculation in the frequency table.

### λ_IE sensitivity range

The sweep covers: [np.float64(0.05), np.float64(0.1), np.float64(0.2), np.float64(0.5), np.float64(1.0), np.float64(2.0)].

---

## 3. Event Tree Structure

```
IE occurs  ──►  G1 activates?  ──►  G2 activates?  ──►  G3 activates?  ──►  Outcome
                │                                                         
   P(G1 ok)─────┴────────────────────────────────────────────────────────►  Branch 1: SAFE
   P(G1 fail)                                                             
              │                                                           
   P(G2 ok)───┴─────────────────────────────────────────────────────────►  Branch 2: SAFE
   P(G2 fail)                                                             
                           │                                              
   P(G3 ok)────────────────┴────────────────────────────────────────────►  Branch 3: NEAR-MISS
   P(G3 fail)                                                             
                                            └──────────────────────────►  Branch 4: CATASTROPHIC
```

Branch probabilities at λ_IE = 0.5/turbine/yr (independent):

| Branch label | Severity | P(branch\|IE) | λ/turbine/yr | Return period (yr) | Freq cat | Risk |
|---|---|---|---|---|---|---|
| Branch_1_SAFE_controlled | E | 8.2810e-01 | 4.140e-01 | 2 | F1 | ACCEPTABLE |
| Branch_2_SAFE_brake | D | 1.4849e-01 | 7.424e-02 | 13 | F2 | ACCEPTABLE |
| Branch_3_NEAR_MISS_SCADA | C | 2.2645e-02 | 1.132e-02 | 88 | F2 | ALARP |
| Branch_4_CATASTROPHIC | A | 7.6900e-04 | 3.845e-04 | 2,601 | F4 | ALARP |

---

## 4. Consequence Frequency Table — Central Estimate

Branch frequencies at λ_IE = 0.5/turbine/yr:


| branch_label | severity_category | P_branch | lambda_per_turbine | lambda_per_farm | freq_category | risk_acceptability | ccf_adjusted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Branch_1_SAFE_controlled | E | 0.8280970000000811 | 0.41404850000004056 | 9.109067000000893 | F1 | ACCEPTABLE | False |
| Branch_2_SAFE_brake | D | 0.14848853847210472 | 0.07424426923605236 | 1.6333739231931519 | F2 | ACCEPTABLE | False |
| Branch_3_NEAR_MISS_SCADA | C | 0.02264545745727002 | 0.01132272872863501 | 0.24910003202997022 | F2 | ALARP | False |
| Branch_4_CATASTROPHIC | A | 0.0007690040705441505 | 0.00038450203527207524 | 0.008459044775985655 | F4 | ALARP | False |
| Branch_1_SAFE_controlled | E | 0.8280970000000811 | 0.41404850000004056 | 9.109067000000893 | F1 | ACCEPTABLE | True |
| Branch_2_SAFE_brake | D | 0.14848853847210472 | 0.07424426923605236 | 1.6333739231931519 | F2 | ACCEPTABLE | True |
| Branch_3_NEAR_MISS_SCADA | C | 0.02220160718874318 | 0.01110080359437159 | 0.24421767907617498 | F2 | ALARP | True |
| Branch_4_CATASTROPHIC | A | 0.0012128543390709933 | 0.0006064271695354966 | 0.013341397729780926 | F4 | ALARP | True |


### Catastrophic Branch Sensitivity to λ_IE


| lambda_IE | ccf_adjusted | lambda_per_turbine | lambda_per_farm | P_one_plus_25yr_farm | freq_category | risk_acceptability |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 | False | 3.8450203527207524e-05 | 0.0008459044775985655 | 0.02092556917383459 | F5 | ACCEPTABLE |
| 0.05 | True | 6.064271695354966e-05 | 0.0013341397729780926 | 0.03280339935663734 | F5 | ACCEPTABLE |
| 0.1 | False | 7.690040705441505e-05 | 0.001691808955197131 | 0.041413258902420313 | F5 | ACCEPTABLE |
| 0.1 | True | 0.00012128543390709933 | 0.002668279545956185 | 0.06453073570392365 | F4 | ALARP |
| 0.2 | False | 0.0001538008141088301 | 0.003383617910394262 | 0.08111145979192169 | F4 | ALARP |
| 0.2 | True | 0.00024257086781419865 | 0.00533655909191237 | 0.12489725555735776 | F4 | ALARP |
| 0.5 | False | 0.00038450203527207524 | 0.008459044775985655 | 0.19061139049214415 | F4 | ALARP |
| 0.5 | True | 0.0006064271695354966 | 0.013341397729780926 | 0.28361313467881366 | F4 | ALARP |
| 1.0 | False | 0.0007690040705441505 | 0.01691808955197131 | 0.3448900787989396 | F4 | ALARP |
| 1.0 | True | 0.0012128543390709933 | 0.026682795459561853 | 0.4867898591952845 | F3 | UNACCEPTABLE |
| 2.0 | False | 0.001538008141088301 | 0.03383617910394262 | 0.5708309911439404 | F3 | UNACCEPTABLE |
| 2.0 | True | 0.0024257086781419865 | 0.053365590919123705 | 0.7366153513752041 | F3 | UNACCEPTABLE |

---

## 5. Risk Matrix Placement (IEC 61400-1 Ed.4 Annex K)

| Risk point | Severity | Frequency | Cell | Acceptability |
|---|---|---|---|---|
| Branch 4: Catastrophic overspeed (central) | A (Extreme) | F4 (3.85e-04/yr) | A × F4 | **ALARP** |
| Branch 4: With CCF uplift (β=0.10) | A (Extreme) | F4 | A × F4 | **ALARP** |
| Branch 4: λ_IE = 2.0/yr (pessimistic) | A (Extreme) | F3 | A × F3 | **UNACCEPTABLE** |
| Branch 3: Near-miss (SCADA trip) | C (Significant) | F2 | C × F2 | ALARP |
| BoP export cable failure (Phase 4) | D (Minor — economic) | F1 (0.281/yr) | D × F1 | ALARP |

**Structural finding:**
The turbine overspeed (Branch 4) and BoP cable failure (Phase 4) are
in **fundamentally different risk categories** on the same matrix.
Overspeed is a low-frequency, catastrophic-consequence safety risk requiring
IEC 61508-style protection. BoP cable is a higher-frequency, economic-
consequence availability risk requiring Phase 7 economic optimisation.
The risk matrix makes this distinction explicit and defensible.

---

## 6. ALARP Argument

The Cat A × F4 overspeed risk requires a formal ALARP case. Three options
are evaluated in Phase 5 FV importance order (highest leverage first).

### Option 1 — Option 1: Duplicate Safety Relay (G3b → SIL-2)
- **Target:** G3b: Safety relay/electrical fault (FV rank 1, FV = 0.8221)
- **Intervention:** Replace single-channel safety relay with dual-channel SIL-2 certified module (IEC 61508 HFT=1). Estimated Q_G3b: 1×10⁻³ (SIL-2 PFDavg upper bound per IEC 61508 Table 4).
- **Effect:** Q_G3 0.03284 → 0.00686
  (79% reduction in Q_top)
- **New Q_top:** 1.607e-04
- **Risk matrix:** Cat A × F4 (ALARP)
  → Cat A × F5 (**ACCEPTABLE**)
- **Cost category:** Low
- **Verdict:** ✅ **RECOMMENDED — moves risk to ACCEPTABLE in one measure.**
  SIL-2 relay modules are inexpensive relative to turbine downtime or blade
  replacement; the ALARP argument is straightforward.

### Option 2 — Option 2: Hydraulic Accumulator Backup (G2a → ×0.2)
- **Target:** G2a: Hydraulic supply loss (FV rank 2, FV = 0.7718)
- **Intervention:** Install independent nitrogen-charged hydraulic accumulator maintaining ≥60 s brake pressure after pump loss. Estimated Q_G2a reduction factor: ~5× (independent pressure source with hydraulic isolation valve).
- **Effect:** Q_G2 0.13621 → 0.05273
  (61% reduction in Q_top)
- **New Q_top:** 2.977e-04
- **Risk matrix:** Cat A × F4 (ALARP)
  → Cat A × F4 (**ALARP**)
- **Cost category:** Medium
- **Verdict:** ⚠️ Significant improvement but insufficient alone to cross
  the F4→F5 boundary at the central λ_IE estimate.
  Recommended as a secondary measure alongside Option 1.

### Option 3 — Option 3: Enhanced Encoder Testing Protocol (G1a → ×0.5)
- **Target:** G1a: Control/Encoder fault (FV rank 3, FV = 0.4500)
- **Intervention:** Periodic offline integrity testing of pitch encoder and controller during planned maintenance (6-month interval). Estimated Q_G1a reduction: ~50% (halves latent fault exposure interval per IEC 61400-26-2 test interval analysis).
- **Effect:** Q_G1 0.17190 → 0.13491
  (22% reduction in Q_top)
- **New Q_top:** 6.035e-04
- **Risk matrix:** Cat A × F4 → Cat A × F4
  (**ALARP**)
- **Cost category:** Low
- **Verdict:** ℹ️ Modest improvement; cannot cross F4→F5 alone.
  Low cost — implement as good practice during planned maintenance stops.

### ALARP Conclusion

Option 1 alone achieves the ACCEPTABLE threshold (Cat A × F5) at the central
λ_IE estimate. The combination of Options 1 + 2 + 3 achieves a combined
Q_top reduction of ~102%,
providing defence-in-depth and robustness against the λ_IE uncertainty.

Note: if λ_IE = 2.0/yr (pessimistic grid-fault environment), even Option 1
leaves the risk at Cat A × F4 (ALARP boundary). Sourcing a site-specific
grid-fault frequency from ENTSO-E or the project's grid code is the single
highest-priority data improvement for this analysis.

---

## 7. Bow-Tie Integration Summary

The bow-tie diagram (`bowtie.png`) combines:
- **Left**: Phase 5 FTA — 7 basic events grouped under OR gates G1/G2/G3, coloured by data confidence tier, CCF hydraulic pathway annotated
- **Centre**: Turbine overspeed hazard box with Q_top (independent and CCF-adjusted)
- **Right**: Phase 6 ETA — four consequence branches with conditional probabilities, absolute frequencies, and risk matrix placement

This is the first diagram in the AEOLUS project that shows causes AND consequences on a single plot, making it the synthesis deliverable of the two-phase safety analysis.

---

## 8. Bridge to Phase 7 (Preventive Maintenance / RCM)

Phase 7 shifts from **safety** (Phases 5–6) to **economics** — optimal
maintenance intervals to minimise total cost per unit time.

Two explicit inheritance items:

**Input:** Phase 6's ALARP analysis identified the safety relay (G3b) as the
highest-leverage single intervention. Phase 7's RCM analysis must determine
whether the relay belongs in:
- **Time-directed PM** (scheduled replacement at fixed interval), or
- **Failure-finding PM** (periodic proof-test to detect latent failures).
Per NORSOK Z-008 and IEC 61511, protective devices belong in the
failure-finding category — the proof-test interval is derived differently
from age-replacement.

**Context:** Pitch System (Weibull β=0.728, Phase 2) and Hydraulic System
were both high-importance items in Phase 4 (availability) and Phase 5 (FTA).
Phase 7's cost-optimization should present two cases:
- β < 1 (as fitted): no optimal age-replacement interval exists —
  on-condition maintenance only.
- β > 1 (physically plausible for aging mechanical actuators):
  age-replacement is justified; Barlow-Proschan formula gives the
  cost-optimal interval.

---

## Appendix A — Definition of Done Checklist

- [x] `config.py` reads gate Q values from Phase 5 `gate_Q_table.csv` — no hard-coded Q values
- [x] `event_tree.py`: four-branch tree verified (branch probabilities sum to 1.0)
- [x] `consequence_frequency_table.csv`: absolute frequencies at all λ_IE values, with and without CCF
- [x] `risk_matrix.py`: all branches + BoP cable plotted on IEC 61400-1 risk matrix
- [x] `alarp.py`: three options quantified; Option 1 moves risk to ACCEPTABLE
- [x] `bowtie.png`: FTA basic events left, ETA branches right, CCF pathway annotated
- [x] `phase6_report.md`: includes λ_IE sensitivity, risk matrix placement, ALARP summary, cross-phase finding
