# Customer Churn Intelligence & Explainable AI (XGBoost + SHAP + Streamlit)

An end-to-end Machine Learning system for predicting customer churn risk, analyzing model explainability using SHAP (SHapley Additive exPlanations), quantifying Customer Lifetime Value (CLV) revenue loss risk, generating automated retention recommendations, tracking model champion-challenger leaderboards, evaluating model fairness, forecasting multi-period churn trajectories, detecting data drift, and serving real-time predictions via an interactive 5-tab Streamlit dashboard.

---

## 🚀 Features

- **Synthetic Realistic Dataset Generator**: Generates realistic customer demographics, usage metrics, support tickets, billing details, and contract structures ([`data/generate_data.py`](file:///D:/Projects/AIML/data/generate_data.py)).
- **Input Schema Validation & Preprocessing**: Ensures data quality with column validation, clean encoding (One-Hot), feature scaling (`StandardScaler`), and missing value handling ([`src/data_loader.py`](file:///D:/Projects/AIML/src/data_loader.py)).
- **Benchmark Model Training & Hyperparameter Tuning**: Trains Logistic Regression, Random Forest, and XGBoost classifiers with `RandomizedSearchCV` cross-validation ([`src/train.py`](file:///D:/Projects/AIML/src/train.py)).
- **Model Evaluation Visualizer**: Plots and saves ROC Curve comparisons and Confusion Matrix heatmaps to `reports/figures/` ([`src/evaluator.py`](file:///D:/Projects/AIML/src/evaluator.py)).
- **Customer CLV & Revenue Loss Calculator**: Quantifies 24-month Customer Lifetime Value (CLV) and total portfolio revenue at risk ([`src/clv_calculator.py`](file:///D:/Projects/AIML/src/clv_calculator.py)).
- **Explainable AI (SHAP)**: Global feature importance bar plots, individual customer prediction attribution, and downloadable SHAP visual export ([`src/explainability.py`](file:///D:/Projects/AIML/src/explainability.py)).
- **Actionable Retention Recommendation Engine**: Generates targeted customer retention strategies based on SHAP risk factors ([`src/recommender.py`](file:///D:/Projects/AIML/src/recommender.py)).
- **Model Artifact Versioning & Registry**: Saves versioned model artifacts and metadata manifest ([`src/model_registry.py`](file:///D:/Projects/AIML/src/model_registry.py)).
- **What-If Scenario Analyzer**: Simulates counterfactual modifications to a customer profile and re-evaluates churn risk ([`src/what_if_analyzer.py`](file:///D:/Projects/AIML/src/what_if_analyzer.py)).
- **Prediction Audit Logger**: Appends every prediction to `logs/predictions_audit.csv` for traceability ([`src/auditor.py`](file:///D:/Projects/AIML/src/auditor.py)).
- **Data Drift Detector**: Compares inference distribution against training baseline using Kolmogorov-Smirnov tests for numerical features and max category shift for categoricals ([`src/drift_detector.py`](file:///D:/Projects/AIML/src/drift_detector.py)).
- **Decision Threshold Optimizer**: Sweeps classification thresholds 0.10–0.90, optimizing for F1-Score and saving `models/optimal_threshold.json` ([`src/threshold_optimizer.py`](file:///D:/Projects/AIML/src/threshold_optimizer.py)).
- **Batch Intervention Report Generator**: Ranks at-risk customers by CLV, assigns P1/P2/P3 priority, and maps intervention playbooks ([`src/report_generator.py`](file:///D:/Projects/AIML/src/report_generator.py)).
- **Customer Cohort Segmentation Engine**: K-Means clustering into 4 behavioral segments (High-Value Loyal, At-Risk Spenders, Budget Churn Risk, Inactive Low-Spend) ([`src/segmenter.py`](file:///D:/Projects/AIML/src/segmenter.py)).
- **Model Fairness & Bias Checker**: Audits predictions across demographic groups using Demographic Parity (4/5ths rule) and Equal Opportunity (TPR parity) ([`src/fairness_checker.py`](file:///D:/Projects/AIML/src/fairness_checker.py)).
- **Dataset Column Profiler & Health Checker**: Evaluates missing value percentages, constant features, high cardinality, and skewness distribution metrics ([`src/data_profiler.py`](file:///D:/Projects/AIML/src/data_profiler.py)).
- **Fine-Grained Risk Band Classifier**: Categorizes churn probabilities into 5 actionable risk tiers (Safe, Low, Moderate, High, Critical) with confidence margin calculation ([`src/risk_classifier.py`](file:///D:/Projects/AIML/src/risk_classifier.py)).
- **Multi-Period Churn Trajectory Projector**: Forecasts 12-month churn risk trajectories comparing baseline trends against proactive retention decay models ([`src/trend_projector.py`](file:///D:/Projects/AIML/src/trend_projector.py)).
- **Model Champion-Challenger Leaderboard**: Persists benchmark rankings in `models/leaderboard.json`, tracking active champion models by ROC-AUC and version differentials ([`src/model_leaderboard.py`](file:///D:/Projects/AIML/src/model_leaderboard.py)).
- **Customer Retention Campaign ROI Simulator**: Simulates campaign economics, break-even customer volume, payback ratio, and optimal tier-based budget allocation ([`src/retention_roi.py`](file:///D:/Projects/AIML/src/retention_roi.py)).
- **Automated Model Retraining Pipeline**: Monitors data drift triggers, trains candidate challenger models, compares against active champion, and conditionally promotes model artifacts ([`src/retrain_pipeline.py`](file:///D:/Projects/AIML/src/retrain_pipeline.py)).
- **Customer Survival Curve & Hazard Rate Simulator**: Actuarial customer retention survival modeling ($S(t)$), hazard functions ($h(t)$), and median half-life tenure forecasting ([`src/survival_simulator.py`](file:///D:/Projects/AIML/src/survival_simulator.py)).
- **Model Probability Calibration Analyzer**: Evaluates prediction reliability using Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Brier Score ([`src/calibrator.py`](file:///D:/Projects/AIML/src/calibrator.py)).
- **Customer Churn Root-Cause Diagnostic Engine**: Translates complex feature attributions into strategic business friction categories with departmental action playbooks ([`src/root_cause_analyzer.py`](file:///D:/Projects/AIML/src/root_cause_analyzer.py)).
- **Automated Executive Intelligence Report Generator**: Compiles C-suite styled, standalone executive HTML intelligence reports with dark-mode styling and KPI cards ([`src/executive_reporter.py`](file:///D:/Projects/AIML/src/executive_reporter.py)).
- **Customer Churn Uplift & Sensitivity Modeler**: Maps customers into causal uplift quadrants (Persuadables, Sure Things, Lost Causes, Sleeping Dogs) to maximize outreach efficiency ([`src/uplift_modeler.py`](file:///D:/Projects/AIML/src/uplift_modeler.py)).
- **Automated Churn Alert & Incident Dispatcher**: Real-time rule engine monitoring VIP customer exposure, portfolio churn surges, data drift, and calibration decay with Slack/Teams payloads ([`src/alert_dispatcher.py`](file:///D:/Projects/AIML/src/alert_dispatcher.py)).
- **Retention A/B Test Power & Significance Calculator**: Statistical experimentation engine for calculating test sample sizes and evaluating two-proportion hypothesis tests ([`src/ab_test_calculator.py`](file:///D:/Projects/AIML/src/ab_test_calculator.py)).
- **Centralized Configuration & Logging**: Manages system paths, feature definitions, and pipeline logs ([`src/config.py`](file:///D:/Projects/AIML/src/config.py)).
- **Automated Test Suite**: 130 comprehensive unit tests across all analytical and ML modules ([`tests/`](file:///D:/Projects/AIML/tests/)).
- **Interactive 5-Tab Streamlit Web Dashboard** ([`app.py`](file:///D:/Projects/AIML/app.py)):
  - **📊 Executive Overview & EDA**: Summary KPIs and interactive Plotly distributions.
  - **🔍 Individual Customer Predictor**: Profile form, fine-grained risk band badge, projected 24-Mo CLV, revenue at risk, SHAP breakdown, retention recommendations, root-cause diagnostics, uplift sensitivity badge, What-If simulator, 12-month churn trajectory forecast, and 24-month survival curve.
  - **📂 Batch CSV Predictor**: Upload customer CSV files with schema validation, portfolio CLV financial loss summary, risk distribution donut chart, cohort segmentation explorer, model fairness audit, retention campaign ROI simulator, A/B test power planner, executive HTML report export, dual CSV exports, and intervention tasklist report.
  - **🤖 Model Performance & Metrics**: Model evaluation comparison tables, ROC-AUC benchmarks, Global SHAP importance charts, Champion-Challenger leaderboard, probability calibration reliability metrics, and automated retraining pipeline trigger.
  - **📈 Data Drift & Audit Logs**: Model registry metadata, operational alert rules monitor, per-feature KS drift analysis with upload comparison, prediction audit log viewer, and data quality profiler.

---

## 📁 Repository Structure

```text
D:\Projects\AIML\
├── data/
│   ├── generate_data.py       # Synthetic dataset generator script
│   └── customer_churn.csv     # Target customer churn dataset
├── models/
│   ├── xgboost_model.pkl      # Active XGBoost classifier artifact
│   ├── preprocessor.pkl       # Active Scikit-Learn ColumnTransformer artifact
│   ├── xgboost_v1.0.0.pkl    # Versioned model artifact
│   ├── preprocessor_v1.0.0.pkl
│   ├── registry.json          # Model version manifest
│   ├── leaderboard.json       # Champion-challenger model leaderboard
│   ├── optimal_threshold.json # Optimal classification threshold
│   ├── best_params.json       # Hyperparameter optimization log
│   └── metrics.json           # Model evaluation performance metrics
├── reports/
│   └── figures/
│       ├── confusion_matrices.png # Evaluation confusion matrix heatmap
│       └── roc_curves.png         # Benchmark model ROC Curves comparison
├── logs/
│   ├── pipeline.log           # System execution log file
│   └── predictions_audit.csv  # Per-prediction audit trail
├── src/
│   ├── __init__.py
│   ├── config.py              # Centralized configuration & logging manager
│   ├── data_loader.py         # Schema validation, preprocessing & split pipeline
│   ├── train.py               # Model training, benchmarking & hyperparameter tuning (CLI)
│   ├── evaluator.py           # Model evaluation plotting & metrics helper
│   ├── clv_calculator.py      # Customer Lifetime Value & revenue loss calculator
│   ├── explainability.py      # SHAP feature importance & local explanations
│   ├── recommender.py         # Automated customer retention recommendation engine
│   ├── model_registry.py      # Model artifact versioning & metadata registry
│   ├── model_leaderboard.py   # Champion-challenger leaderboard tracker
│   ├── what_if_analyzer.py    # Counterfactual what-if scenario simulator
│   ├── auditor.py             # Prediction audit logger
│   ├── drift_detector.py      # Feature distribution drift detector (KS-test)
│   ├── threshold_optimizer.py # Classification threshold optimizer (F1-Score)
│   ├── report_generator.py   # Batch customer intervention report generator
│   ├── segmenter.py           # Customer behavioral cohort segmentation engine
│   ├── fairness_checker.py    # Model demographic parity and equal opportunity auditor
│   ├── data_profiler.py       # Dataset column profiler and data quality checker
│   ├── risk_classifier.py     # Fine-grained risk band classifier and confidence scorer
│   ├── trend_projector.py     # Multi-period churn trajectory projector
│   ├── retention_roi.py       # Customer retention campaign financial ROI simulator
│   ├── retrain_pipeline.py    # Automated model retraining pipeline and challenger promoter
│   ├── survival_simulator.py  # Customer survival curve and hazard rate simulator
│   ├── calibrator.py          # Model calibration and reliability analyzer (ECE, MCE)
│   ├── root_cause_analyzer.py # Customer churn root-cause diagnostic engine
│   ├── executive_reporter.py  # Automated executive HTML intelligence report generator
│   ├── uplift_modeler.py      # Customer churn uplift & incrementality sensitivity modeler
│   ├── alert_dispatcher.py    # Automated operational alert rules & webhook dispatcher
│   └── ab_test_calculator.py  # Retention A/B test power & hypothesis test calculator
├── tests/
│   ├── test_config.py
│   ├── test_data_loader.py
│   ├── test_train.py
│   ├── test_explainability.py
│   ├── test_clv_calculator.py
│   ├── test_recommender.py
│   ├── test_model_registry.py
│   ├── test_model_leaderboard.py
│   ├── test_what_if.py
│   ├── test_auditor.py
│   ├── test_drift_detector.py
│   ├── test_threshold_optimizer.py
│   ├── test_report_generator.py
│   ├── test_segmenter.py
│   ├── test_fairness_checker.py
│   ├── test_data_profiler.py
│   ├── test_risk_classifier.py
│   ├── test_trend_projector.py
│   ├── test_retention_roi.py
│   ├── test_retrain_pipeline.py
│   ├── test_survival_simulator.py
│   ├── test_calibrator.py
│   ├── test_root_cause_analyzer.py
│   ├── test_executive_reporter.py
│   ├── test_uplift_modeler.py
│   ├── test_alert_dispatcher.py
│   └── test_ab_test_calculator.py
├── app.py                     # 5-tab Streamlit web application
├── requirements.txt           # Python dependencies
└── README.md                  # Documentation
```

---

## 🛠️ Quickstart & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python data/generate_data.py
```

### 3. Run Automated Tests
```bash
python -m pytest tests/ -v
```

### 4. Train Models

**Basic training:**
```bash
python src/train.py
```

**With hyperparameter tuning:**
```bash
python src/train.py --tune
```

**With custom version tag:**
```bash
python src/train.py --version v1.2.0
```

### 5. Launch Streamlit Web Application
```bash
streamlit run app.py
```

---

## 🖥️ CLI Reference — `src/train.py`

| Flag | Type | Description |
|---|---|---|
| `--tune` | flag | Enable XGBoost `RandomizedSearchCV` hyperparameter tuning |
| `--version` | string | Semantic version tag to register model artifact (e.g. `v1.1.0`) |
| `--eval` | flag | Generate and save ROC Curve and Confusion Matrix figures |

---

## 📊 Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | ~0.818 | ~0.662 | ~0.384 | ~0.486 | ~0.847 |
| **Random Forest** | ~0.820 | ~0.775 | ~0.277 | ~0.408 | ~0.845 |
| **XGBoost (Best)** | **~0.802+** | **~0.594+** | **~0.366+** | **~0.453+** | **~0.841+** |