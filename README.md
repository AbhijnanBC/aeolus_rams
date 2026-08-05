Here is a comprehensive, portfolio-ready `README.md` for the AEOLUS-RAMS repository. It synthesizes the entire 8-phase architecture, highlights the advanced mathematical and data science techniques you implemented, and is perfectly structured for CS admissions committees or engineering recruiters.

---

# AEOLUS-RAMS 🌬️

**Reliability, Availability, Maintainability, and Safety (RAMS) Predictive Architecture**

AEOLUS-RAMS is a production-grade, 8-phase reliability engineering pipeline designed to ingest raw wind turbine SCADA data and output validated predictive maintenance strategies. Built against the CARE-to-Compare wind turbine benchmark dataset, this project bridges the gap between raw anomaly logs and advanced statistical survival analysis, ultimately delivering cost-optimized Condition-Based Maintenance (CBM) thresholds.

## 🚀 Technical Highlights

* **Automated FMECA Engine:** NLP-based hybrid curated/keyword tagging system that parses free-text maintenance logs into canonical component taxonomy.


* **Rigorous Survival Analysis:** Extracts right-censored Time-Between-Failures (TBF) and fits Weibull/Exponential distributions with strict sample-size guarding.


* **Bayesian Parameter Updating:** Blends literature-informed MTBF priors with small-sample operational data via Gamma-Poisson conjugate modeling.


* **Dynamic Predictive Modeling:** Utilizes Cox Proportional Hazards to calculate hazard ratios against live SCADA covariates (wind speed, vibration, temperature).
* **Monte Carlo Optimization:** Executes multi-phase simulations to derive financially optimal alert thresholds (e.g., $k^*=4.00\sigma$) balancing preventive costs against catastrophic failure risks.

---

## 🏗️ The 8-Phase Architecture

The codebase is strictly modularized into 8 sequential phases, each building mathematically on the outputs of its predecessor:

### Phase 1: System Definition & FMECA

Turns raw CARE dataset logs into a ranked, evidence-backed Failure Mode, Effects, and Criticality Analysis (FMECA) table. It handles dataset inventory validation, true occurrence frequency aggregation, and automated failure-mode tagging.

### Phase 2: Weibull, MTBF & Hazard Rates

Performs strict data-sufficiency tiering (Tiers A, B, C) to fit 2-parameter Weibull or Exponential distributions to the FMECA data. Includes bootstrap confidence intervals, right-censoring management, and Bayesian inference for small-sample failure modes.

### Phase 3: Reliability Block Diagrams (RBD)

Models the logical system topology (series/parallel configurations) to extrapolate individual component MTBFs into subsystem and farm-level reliability metrics.

### Phase 4: Monte Carlo Availability

Runs stochastic simulations across operational scenarios to quantify system availability, mapping specific component downtimes to overall yield losses and generating sensitivity tornado charts.

### Phase 5: Fault Tree Analysis (FTA)

Constructs boolean logic trees to perform top-down root-cause analysis for catastrophic system states. Includes Minimal Cut Set (MCS) extraction and probabilistic importance ranking.

### Phase 6: Event Tree Analysis (ETA)

Executes forward-looking consequence modeling (bowtie diagrams) to evaluate risk matrices and ALARP (As Low As Reasonably Practicable) mitigation strategies.

### Phase 7: Preventive Maintenance

Computes Barlow-Proschan cost-rate curves to determine optimal age-replacement schedules, balancing the cost of planned interventions against the severe financial penalties of run-to-failure events.

### Phase 8: Dynamic Reliability & Predictive Modeling

The capstone of the pipeline. Integrates live SCADA covariates using `lifelines` Cox Proportional Hazards modeling, updates structural parameters via Bayesian grid approximation, and performs Monte Carlo sweeps to define exact, cost-optimal Condition-Based Maintenance (CBM) standard-deviation thresholds.

---

## 📊 Dataset Requirements

This pipeline is built to process the **CARE-to-Compare wind turbine SCADA benchmark dataset** (Gueck, Roelofs & Faulstich, 2024).

* The data loader automatically resolves standard CARE archive structures, handling the doubled-folder nesting and non-standard `;` CSV delimiters transparently.


* **Note:** The CARE dataset itself is gigabytes in size and licensed separately via Zenodo. It is not included in this repository. Ensure your raw data is placed in `data/raw/care` before executing the pipeline.



---

## 💻 Installation & Usage

**1. Clone the repository and setup the environment:**

```bash
git clone https://github.com/yourusername/aeolus-rams.git
cd aeolus-rams
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

```

**2. Execute the Pipeline:**
Each phase is packaged as an executable module. Run them sequentially.

```bash
# Example: Running Phase 1[cite: 1]
python -m aeolus_rams.pipeline --data-root data/raw/care --output-dir phase-1-system-fmeca

# Example: Running Phase 2[cite: 1]
python -m aeolus_rams_phase2.pipeline --phase1-dir phase-1-system-fmeca --output-dir phase-2-weibull-mtbf-hazard

```

*Every phase generates a detailed `phaseX_report.md` alongside comprehensive CSV tables and diagnostic Matplotlib visualizations.*

---

## 🗂️ Project Structure

```text
AEOLUS-RAMS/
├── phase-1-system-fmeca/           # NLP Tagging & Risk Prioritization
├── phase-2-weibull-mtbf-hazard/    # Survival Analysis & Bayesian Inference
├── phase-3-rbd/                    # Logical Topology Modeling
├── phase-4-monte-carlo/            # Stochastic Availability Simulation
├── phase-5-fta/                    # Root Cause Logic Trees
├── phase-6-eta/                    # Consequence Mitigation Matrix
├── phase-7-preventive-maintenance/ # Barlow-Proschan Cost Optimization
└── phase-8-dynamic-modeling/       # Cox PH & CBM Thresholding

```

---

## 👨‍💻 Author

**Abhijnan B C**
