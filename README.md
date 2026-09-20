# Customer Churn Intelligence & Explainable AI (XGBoost + SHAP + Streamlit)

An end-to-end Machine Learning system for predicting customer churn risk, analyzing model explainability using SHAP (SHapley Additive exPlanations), generating automated retention recommendations, and serving real-time predictions via an interactive Streamlit dashboard.

---

## 🚀 Features

- **Synthetic Realistic Dataset Generator**: Generates realistic customer demographics, usage metrics, support tickets, billing details, and contract structures ([`data/generate_data.py`](file:///D:/Projects/AIML/data/generate_data.py)).
- **Input Schema Validation & Preprocessing**: Ensures data quality with column validation, clean encoding (One-Hot), feature scaling (`StandardScaler`), and missing value handling ([`src/data_loader.py`](file:///D:/Projects/AIML/src/data_loader.py)).
- **Benchmark Model Training & Hyperparameter Tuning**: Trains Logistic Regression, Random Forest, and XGBoost classifiers with optional `RandomizedSearchCV` cross-validation ([`src/train.py`](file:///D:/Projects/AIML/src/train.py)).
- **Explainable AI (SHAP)**: Global feature importance bar plots and individual customer prediction attribution ([`src/explainability.py`](file:///D:/Projects/AIML/src/explainability.py)).
- **Actionable Retention Recommendation Engine**: Generates targeted customer retention strategies based on SHAP risk factors ([`src/recommender.py`](file:///D:/Projects/AIML/src/recommender.py)).
- **Automated Test Suite**: Full unit test coverage across data loading, training, SHAP, and recommender modules ([`tests/`](file:///D:/Projects/AIML/tests/)).
- **Interactive Streamlit Web Dashboard** ([`app.py`](file:///D:/Projects/AIML/app.py)):
  - **Executive Overview & EDA**: Summary KPIs and interactive Plotly distributions.
  - **Single Customer Predictor**: Input profile form, risk status badge, SHAP breakdown, and automated retention recommendations.
  - **Batch CSV Predictor**: Upload customer CSV files with schema validation for bulk scoring and CSV export.
  - **Model Performance & Metrics**: Model evaluation comparison tables, ROC-AUC benchmarks, and Global SHAP importance charts.

---

## 📁 Repository Structure

```text
D:\Projects\AIML\
├── data/
│   ├── generate_data.py       # Synthetic dataset generator script
│   └── customer_churn.csv     # Target customer churn dataset
├── models/
│   ├── xgboost_model.pkl      # Saved XGBoost classifier artifact
│   ├── preprocessor.pkl       # Saved Scikit-Learn ColumnTransformer artifact
│   ├── best_params.json       # Hyperparameter optimization log
│   └── metrics.json           # Model evaluation performance metrics
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Schema validation, preprocessing & split pipeline
│   ├── train.py               # Model training, benchmarking & hyperparameter tuning
│   ├── explainability.py      # SHAP feature importance & local explanations
│   └── recommender.py         # Automated customer retention recommendation engine
├── tests/
│   ├── test_data_loader.py    # Unit tests for data loading & validation
│   ├── test_train.py          # Unit tests for model training & evaluation
│   ├── test_explainability.py # Unit tests for SHAP explainer
│   └── test_recommender.py   # Unit tests for retention recommendation engine
├── app.py                     # Streamlit web application
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
python -m unittest discover -s tests -p "test_*.py"
```

### 4. Train Models & Tune Hyperparameters
```bash
python src/train.py
```

### 5. Launch Streamlit Web Application
```bash
streamlit run app.py
```

---

## 📊 Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | ~0.818 | ~0.662 | ~0.384 | ~0.486 | ~0.847 |
| **Random Forest** | ~0.820 | ~0.775 | ~0.277 | ~0.408 | ~0.845 |
| **XGBoost (Best)** | **~0.802+** | **~0.594+** | **~0.366+** | **~0.453+** | **~0.841+** |