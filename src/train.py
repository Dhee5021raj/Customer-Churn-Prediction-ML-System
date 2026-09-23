import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import (
    load_raw_data, prepare_train_test_data, save_preprocessor
)


def evaluate_model(model: Any, X_test: np.ndarray, y_test: pd.Series) -> Dict[str, Any]:
    """Calculate key evaluation metrics for a trained classifier."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    metrics = {
        "accuracy": float(round(accuracy_score(y_test, y_pred), 4)),
        "precision": float(round(precision_score(y_test, y_pred, zero_division=0), 4)),
        "recall": float(round(recall_score(y_test, y_pred, zero_division=0), 4)),
        "f1_score": float(round(f1_score(y_test, y_pred, zero_division=0), 4)),
        "roc_auc": float(round(roc_auc_score(y_test, y_prob), 4)),
        "confusion_matrix": cm
    }
    return metrics


from sklearn.model_selection import RandomizedSearchCV


def tune_xgboost_hyperparameters(X_train: np.ndarray, y_train: pd.Series, n_iter: int = 10, random_state: int = 42) -> XGBClassifier:
    """Optimize XGBoost hyperparameters using RandomizedSearchCV with 3-Fold Stratified Cross-Validation."""
    param_dist = {
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [3, 4, 5, 6],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "scale_pos_weight": [1, 2, 3]
    }
    
    base_xgb = XGBClassifier(random_state=random_state, eval_metric="logloss")
    search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=3,
        random_state=random_state,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    
    os.makedirs("models", exist_ok=True)
    with open("models/best_params.json", "w") as f:
        json.dump(search.best_params_, f, indent=4)
        
    print(f"Hyperparameter tuning complete. Best params saved to models/best_params.json: {search.best_params_}")
    return search.best_estimator_


def train_and_evaluate_models(tune_hyperparams: bool = False) -> Dict[str, Dict[str, Any]]:
    """Train benchmark ML models and select best XGBoost classifier."""
    print("Loading data and running preprocessor...")
    df = load_raw_data("data/customer_churn.csv")
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_train_test_data(df)
    
    # Save preprocessor artifact
    save_preprocessor(preprocessor, "models/preprocessor.pkl")
    print("Fitted preprocessor saved to models/preprocessor.pkl")

    if tune_hyperparams:
        print("\nRunning hyperparameter tuning for XGBoost...")
        tuned_xgb = tune_xgboost_hyperparameters(X_train, y_train)
    else:
        tuned_xgb = XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )

    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, max_depth=10),
        "XGBoost": tuned_xgb
    }

    results = {}
    best_xgboost = None

    print("\nTraining and evaluating models:")
    print("-" * 65)
    print(f"{'Model':<22} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<6} | {'F1':<6} | {'ROC-AUC':<7}")
    print("-" * 65)

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        
        print(f"{name:<22} | {metrics['accuracy']:<8.4f} | {metrics['precision']:<9.4f} | {metrics['recall']:<6.4f} | {metrics['f1_score']:<6.4f} | {metrics['roc_auc']:<7.4f}")
        
        if name == "XGBoost":
            best_xgboost = model

    # Save best XGBoost model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_xgboost, "models/xgboost_model.pkl")
    print("\nTrained XGBoost model saved to models/xgboost_model.pkl")

    # Save metrics JSON
    with open("models/metrics.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Evaluation metrics saved to models/metrics.json")

    # Register model version artifact
    try:
        from src.model_registry import register_model
        reg_entry = register_model(best_xgboost, preprocessor, results["XGBoost"], version="v1.0.0")
        print(f"Model registered in registry.json: version {reg_entry['version']}")
    except Exception as e:
        print(f"Warning: Could not register model in manifest: {e}")

    # Generate & save evaluation figures
    try:
        from src.evaluator import plot_confusion_matrices, plot_roc_curves
        cm_path = plot_confusion_matrices(results)
        roc_path = plot_roc_curves(models, X_test, y_test)
        print(f"Saved evaluation figures: {cm_path}, {roc_path}")
    except Exception as e:
        print(f"Warning: Could not generate evaluation figures: {e}")

    return results

if __name__ == "__main__":
    train_and_evaluate_models()
