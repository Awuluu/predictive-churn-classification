# Predictive Churn Classification System

> End-to-end machine learning system that predicts customer account churn, explains each prediction with SHAP, and delivers a risk-tiered action list for Customer Success teams.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Binary_Classifier-orange)
![MLflow](https://img.shields.io/badge/MLflow-Tracking_%26_Registry-purple)
![Databricks](https://img.shields.io/badge/Databricks-Azure-red)
![Delta Lake](https://img.shields.io/badge/Delta_Lake-Managed_Tables-green)

---

## Problem Statement

Customer churn directly erodes ARR and strains retention teams that can’t manually prioritize 600+ accounts. This project builds an early-warning system that:

* Identifies which accounts are likely to cancel **before** renewal
* Explains **why** each account is at risk (top-3 SHAP drivers)
* Ranks accounts into actionable risk tiers (Low / Medium / High / Critical)
* Outputs a ready-to-use CSM action list with dollar-value exposure

---

## Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. RESEARCH NOTEBOOK (34 cells)                                          │
│  EDA → Leakage Detection → Feature Engineering → Model Training          │
│  → Hyperparameter Tuning → SHAP Analysis → Threshold Optimization        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. TRAINING NOTEBOOK (11 cells)  │  Cadence: Quarterly                    │
│  Preprocessing → 50-Trial Optuna Bayesian Search (5-fold Stratified CV)   │
│  → Champion/Challenger Gate → Register to Unity Catalog                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. SCORING NOTEBOOK (7 cells)    │  Cadence: Weekly                      │
│  Load Champion from UC → Apply Frozen Medians → Score All Accounts       │
│  → SHAP Explanations → Risk Tiers → CSM Action List                     │
└─────────────────────────────────────────────────────────────────────────────┘
Tech Stack
Layer	Tool
Algorithm	XGBoost (binary classifier, native categorical handling, scale_pos_weight for class imbalance)
Hyperparameter Tuning	Optuna (Bayesian TPE sampler, 50 trials, 5-fold Stratified K-Fold CV, early stopping)
Explainability	SHAP TreeExplainer — global feature importance + per-account top-3 risk drivers and protective factors
Experiment Tracking	MLflow (experiment logging, artifact storage, model signatures)
Model Registry	Unity Catalog Model Registry (versioning, @champion alias, champion/challenger promotion)
Deployment Wrapper	MLflow PyFunc custom class embedding threshold + SHAP + scoring_config.json
Data	Delta Lake managed tables on Azure ADLS, Databricks SQL
Preprocessing	pandas, scikit-learn, NumPy
Visualization	matplotlib, seaborn
Feature Engineering
8 derived features built from domain knowledge:

Feature	Logic
Total_Products_Owned	Count of active product families per account
Is_Multi_Product_Customer	Binary: owns more than one product family
Has_Platform	Binary: has Tenable One (platform product)
ARR_Concentration	Ratio of largest product ARR to total ARR
Total_Overage_Flags	Count of overage indicators across products
Has_Any_Overage	Binary: any overage flag present
Has_Support_Activity	Binary: non-zero support case count
Contract_Days_Remaining_Ratio	Days remaining / total contract term
Key Design Decisions
Leakage detection: Account Status and Ending Enterprise ARR identified as target leakage via crosstab analysis — dropped before training
Missing value strategy: XGBoost learns default split direction for NaN natively; categorical nulls filled with "Unknown"; numeric nulls imputed with training medians frozen at train time
Training/serving skew prevention: Medians and feature order stored in scoring_config.json, loaded at inference — never recomputed from scoring data
Threshold optimization: Swept validation set for F1-max → production threshold = 0.41
Champion/challenger gate: New model must beat current champion on held-out test set before promotion is allowed
Results
Model Performance (Held-Out Test Set)
Metric	Value
AUC	0.858
F1 Score	0.709
Precision	0.694
Recall	0.725
Accuracy	0.819
Production Scoring Output
606 active accounts scored
48 flagged as at-risk (7.9%)
~$321K ARR identified at risk
Each prediction includes top-3 SHAP risk drivers + protective factors
Accounts bucketed into Low / Medium / High / Critical risk tiers
Project Structure
predictive-churn-classification/
├── notebooks/
│   ├── 01_research.py              # EDA, leakage detection, feature engineering, training, SHAP
│   ├── 02_training_pipeline.py      # Production retraining with Optuna + champion/challenger
│   └── 03_scoring_pipeline.py       # Weekly inference, SHAP explanations, CSM action list
├── config/
│   └── scoring_config.json          # Frozen medians, feature order, threshold
├── docs/
│   ├── technical_walkthrough.md     # Cell-by-cell explanation
│   └── presentation_guide.md       # Executive summary deck notes
└── README.md
What I Learned
Leakage is subtle: Columns that look like valid features (Account Status) can perfectly predict the target and silently inflate metrics — crosstab analysis before feature selection is non-negotiable
Threshold ≠ 0.5: Optimizing the decision threshold on validation data improved F1 by 8+ points over the default
Frozen preprocessing matters: Computing medians from scoring data instead of training data introduces training/serving skew — serializing a scoring_config.json at train time prevents this
SHAP makes models actionable: Raw predictions alone aren’t useful to CSMs; per-account explanations (“this account is at risk because ARR dropped 40% and they have no platform product”) drive actual intervention
