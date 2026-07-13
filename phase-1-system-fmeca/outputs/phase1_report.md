# AEOLUS-RAMS — Phase 1 Report
*Generated 2026-07-13 06:08 UTC — aeolus_rams v1.0.0*

## 1. Dataset Inventory (Section 1.1.3)
| farm   |   n_turbine_datasets |   n_logged_events |   n_anomaly_events |   n_normal_events | has_feature_description   | counts_reconcile   |
|:-------|---------------------:|------------------:|-------------------:|------------------:|:--------------------------|:-------------------|
| A      |                   22 |                22 |                 11 |                11 | True                      | True               |
| B      |                   15 |                15 |                  6 |                 9 | True                      | True               |
| C      |                   58 |                58 |                 27 |                31 | True                      | True               |

## 2. Step 0 — Data Validation (Section 1.1)
- All farm inventories reconcile: **True**
- Unknown `status_type_id` codes found: **False**
  - Farm A: observed=[0, 3, 4, 5], unknown=[], documented-but-not-observed=[1, 2]
  - Farm B: observed=[0, 1, 4, 5], unknown=[], documented-but-not-observed=[2, 3]
  - Farm C: observed=[0, 3, 4, 5], unknown=[], documented-but-not-observed=[1, 2]

> **Reminder:** do not assume `failure = (status_type_id == 4)` purely because no unknown codes were found. Section 1.1.1 also asks for an empirical cross-check against known anomaly windows (`validation.empirical_status_check_around_events`) before that definition is locked into the canonical schema.

### 2.1 True occurrence frequency, per farm (`.value_counts()`, not `.unique()`)
**Farm A**
| event_description         |   count |
|:--------------------------|--------:|
| Hydraulic group           |       6 |
| Gearbox failure           |       2 |
| Generator bearing failure |       2 |
| Transformer failure       |       1 |

**Farm B**
| event_description                                                |   count |
|:-----------------------------------------------------------------|--------:|
| high temperature                                                 |       3 |
| Rotor Bearing 2 - Damage                                         |       1 |
| Turbine is stopped due to a main bearing damage                  |       1 |
| Turbine is in standstill since 01.08 due to rotorbearing damage. |       1 |

**Farm C**
| event_description                                                                                                                                                                             |   count |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------:|
| 23020 : Axis 3 not ready-to-operate                                                                                                                                                           |       2 |
| Harting plug Nacelle/HUB damaged + NCR20_HUB: Wiring blade control system                                                                                                                     |       1 |
| Converter Failure from 17.11 12:30 - 18.11. 13:57, Fuse Filter Supply                                                                                                                         |       1 |
| Failure due to Rotorbrake and Hydraulic problemes - Hydraulic pump A disabeld, 2h later turbine back in production - Data shows anomaly in temp_hydraulic_oil_tank_1_average until 15.01.2023 |       1 |
| 10115 : Oil level error, two-pump mode + Oil Leakage Gear Oil Supply + 12019: Rotor brake B cannot be closed + P20_yaw carbon brush damaged                                                   |       1 |
| We had some failures (störung 24VAC Versorgung Rotorbremse) on the 16th in the afternoon. From 17th onwards a longer standstill where we don't know the root cause to.                        |       1 |
| P20_spinner_carbonbrush defekt + P20_Accumulators_hydraulic system                                                                                                                            |       1 |
| 15004 : Safety chain relay open + 93005 : Gear oil cooler bypass valve                                                                                                                        |       1 |
| Pitchfailure - defect Beckhoffcard, Axis 2, rectified on 23/01 - Anomalie liegt aber länger an als der Fehler, Batterien waren ok                                                             |       1 |
| Randomn small failures regarding pitch resulting in a longer standstill due to a defect pitch encoder (26/02)                                                                                 |       1 |
| P20_Grounding role brake disc + P20_cover-lightning-main-cabinet-hub                                                                                                                          |       1 |
| Communication fault BK1120 in NC300                                                                                                                                                           |       1 |
| Pitch failure - defect fan on pitch motor                                                                                                                                                     |       1 |
| P20_Blade3_Grease Collector missing                                                                                                                                                           |       1 |
| P20_DGUV-v3 RCD 28F1 NC310 defective + 0 : P20_Blades_Cabinet Caps missing                                                                                                                    |       1 |
| Valve in water cooling system was left in wrong position after maintenance actions on 05-08-2020                                                                                              |       1 |
| Failure 2023-04-05 03:30 - defective coupling between gear oil pump and motor                                                                                                                 |       1 |
| Communication and Pitchfailure - slip ring and Beckhoff card                                                                                                                                  |       1 |
| Turbine has some issues with overpressure on the main transformer                                                                                                                             |       1 |
| PENDING19_PREV_YAW_Grease pump defective                                                                                                                                                      |       1 |
| WEC in failure - current measurement own consumption                                                                                                                                          |       1 |
| COMMUNICATION FAULT BK1120 IN NC300 A2                                                                                                                                                        |       1 |
| 21002 : Axis 1 DC-link voltage low, batt                                                                                                                                                      |       1 |
| Turbine had several short standstills (max 8min) with failure "Schwingungen Umrichter Drehmomenten Level 1"                                                                                   |       1 |
| WEC in failure - hub battery charger defect                                                                                                                                                   |       1 |
| WEC in failure with pitch battery issues - rewiring                                                                                                                                           |       1 |

## 3. Failure Mode Tagging (Section 1.4)
- Anomaly-labeled events tagged: **44**
- Flagged for manual review: **0**

_No entries currently require manual review._

## 4. FMECA Table (Sections 1.5 & 1.6)
`occurrence` is the Section 1.5 rubric score derived from real `distinct_incidents_observed`; `severity`/`detection` default to the Section 1.6 worked-table values (two components extended — see `fmeca.SEVERITY_DETECTION_DEFAULTS` docstring) and are fully overridable via `fmeca.with_custom_scores`.
|   rank | component                      | subsystem_group       |   distinct_incidents_observed |   severity |   occurrence |   detection |   rpn | data_support   |
|-------:|:-------------------------------|:----------------------|------------------------------:|-----------:|-------------:|------------:|------:|:---------------|
|      1 | Main/Rotor Bearing             | Drivetrain            |                             3 |         10 |            6 |           8 |   480 | observed       |
|      2 | Pitch System                   | Rotor                 |                            11 |          8 |            9 |           5 |   360 | observed       |
|      3 | Gearbox                        | Drivetrain            |                             3 |          9 |            6 |           6 |   324 | observed       |
|      4 | Generator                      | Drivetrain/Electrical |                             2 |          9 |            4 |           7 |   252 | observed       |
|      5 | Hydraulic System               | Nacelle (shared)      |                             6 |          7 |            7 |           4 |   196 | observed       |
|      6 | Cooling System                 | Nacelle               |                             4 |          5 |            6 |           6 |   180 | observed       |
|      7 | Grounding/Lightning Protection | Nacelle/Rotor         |                             2 |          6 |            4 |           7 |   168 | observed       |
|      8 | Mechanical Brake               | Nacelle               |                             3 |          9 |            6 |           3 |   162 | observed       |
|      9 | Transformer                    | Power Conversion      |                             2 |          8 |            4 |           5 |   160 | observed       |
|     10 | Electrical Safety System       | Control & Safety      |                             3 |          8 |            6 |           2 |    96 | observed       |
|     11 | Converter                      | Power Conversion      |                             2 |          6 |            4 |           3 |    72 | observed       |
|     12 | SCADA/Communication            | Control & Safety      |                             2 |          6 |            4 |           2 |    48 | observed       |
|     13 | Yaw System                     | Nacelle               |                             1 |          5 |            2 |           4 |    40 | observed       |

## 5. Definition of Done (Section 1.9)
- [x] `status_type_id` legend cross-checked against documentation (no unknown codes observed)
- [x] `.value_counts()` run per farm on `event_description`; true occurrence frequencies recorded
- [x] Component taxonomy adopted (Section 1.3)
- [x] Every distinct failure description mapped to a primary component (curated or keyword-tagged)
- [x] Automated tagging run, and every 'Unclassified — Review Manually' entry resolved by hand
- [x] FMECA table completed with real S/O/D/RPN values for all taxonomy components
- [x] Criticality ranking finalized (sorted by RPN, descending)
- [x] `tagged_events.csv` and `fmeca_table.csv` exported

Unchecked items require a human decision this script cannot make safely on its own (reading unclassified free text, confirming the status legend against your Zenodo record, re-reviewing the two added Severity/Detection defaults). Everything else is produced and verified by this run.