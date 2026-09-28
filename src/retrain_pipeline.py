"""
src/retrain_pipeline.py
-----------------------
Automated model retraining pipeline with drift triggers and champion promotion.

Evaluates dataset drift and performance decay, triggers automated model
training, compares candidate challenger against current champion, and
conditionally promotes and registers new production model artifacts.

Functions:
    check_retraining_trigger(drift_report, min_drifted_features) -> dict
    evaluate_candidate_vs_champion(candidate_metrics, champion_metrics, metric_key) -> dict
    run_retraining_cycle(df, version_tag, tune, min_improvement) -> dict
"""

import os
import joblib
import pandas as pd
from typing import Dict, Any, Optional

from src.config import setup_logger
from src.data_loader import prepare_train_test_data
from src.model_leaderboard import get_champion_model, update_leaderboard
from src.model_registry import register_model

logger = setup_logger("retrain_pipeline")


def check_retraining_trigger(
    drift_report: Dict[str, Any],
    min_drifted_features: int = 2,
) -> Dict[str, Any]:
    """
    Evaluates whether an automated retraining run is warranted based on drift metrics.

    Parameters
    ----------
    drift_report : dict
        Output from `calculate_feature_drift`.
    min_drifted_features : int
        Minimum count of drifted features to trigger retraining.

    Returns
    -------
    dict
        {"should_retrain": bool, "drifted_features": int, "reason": str}
    """
    drifted_count = drift_report.get("drifted_features_count", 0)
    drift_detected = drift_report.get("drift_detected", False)

    if drifted_count >= min_drifted_features:
        reason = f"Drift threshold exceeded: {drifted_count} features drifted (minimum threshold: {min_drifted_features})."
        should_retrain = True
    elif drift_detected:
        reason = "Data distribution drift detected on active features."
        should_retrain = True
    else:
        reason = "Data distribution stable. Retraining not required."
        should_retrain = False

    logger.info(f"Retraining check: should_retrain={should_retrain} ({reason})")
    return {
        "should_retrain": should_retrain,
        "drifted_features": drifted_count,
        "reason": reason,
    }


def evaluate_candidate_vs_champion(
    candidate_metrics: Dict[str, Any],
    champion_metrics: Optional[Dict[str, Any]] = None,
    metric_key: str = "roc_auc",
    min_improvement: float = 0.0,
) -> Dict[str, Any]:
    """
    Compares candidate challenger model against current production champion.

    Parameters
    ----------
    candidate_metrics : dict
        Performance metrics of newly trained candidate model.
    champion_metrics : dict, optional
        Metrics of current active champion from leaderboard.
    metric_key : str
        Primary comparison metric (default: 'roc_auc').
    min_improvement : float
        Minimum delta improvement required to dethrone champion (default: 0.0).

    Returns
    -------
    dict
        {"promoted": bool, "metric": str, "candidate_score": float,
         "champion_score": float, "delta": float, "decision": str}
    """
    candidate_score = float(candidate_metrics.get(metric_key, 0.0))

    if not champion_metrics or metric_key not in champion_metrics:
        # No existing champion, automatically promote candidate
        return {
            "promoted": True,
            "metric": metric_key,
            "candidate_score": round(candidate_score, 4),
            "champion_score": None,
            "delta": round(candidate_score, 4),
            "decision": "Promoted: Initial champion established.",
        }

    champion_score = float(champion_metrics.get(metric_key, 0.0))
    delta = round(candidate_score - champion_score, 4)

    if delta >= min_improvement:
        promoted = True
        decision = f"Promoted: Candidate outperformed champion by {delta:+.4f} on {metric_key}."
    else:
        promoted = False
        decision = f"Rejected: Candidate did not exceed champion by required margin ({delta:+.4f} vs threshold {min_improvement:+.4f})."

    logger.info(f"Challenger evaluation: {decision}")
    return {
        "promoted": promoted,
        "metric": metric_key,
        "candidate_score": round(candidate_score, 4),
        "champion_score": round(champion_score, 4),
        "delta": delta,
        "decision": decision,
    }


def run_retraining_cycle(
    df: pd.DataFrame,
    version_tag: str = "v1.1.0",
    tune: bool = False,
    min_improvement: float = 0.0,
    save_as_active: bool = True,
) -> Dict[str, Any]:
    """
    Executes a complete retraining cycle on input customer data.

    Trains an XGBoost challenger model, benchmarks against current champion,
    and updates registry and leaderboard if promoted.

    Parameters
    ----------
    df : pd.DataFrame
        Customer training dataset.
    version_tag : str
        Semantic version for candidate model.
    tune : bool
        Whether to run hyperparameter tuning.
    min_improvement : float
        Required improvement margin over champion to promote.
    save_as_active : bool
        Whether to overwrite models/xgboost_model.pkl when promoted.

    Returns
    -------
    dict
        Retraining cycle result summary.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from xgboost import XGBClassifier

    logger.info(f"Starting retraining cycle for version {version_tag} (tune={tune})")
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_train_test_data(df)

    if tune:
        from src.train import tune_xgboost_hyperparameters
        best_params = tune_xgboost_hyperparameters(X_train, y_train)
        candidate_model = XGBClassifier(**best_params, random_state=42, eval_metric="logloss")
    else:
        candidate_model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
        )

    candidate_model.fit(X_train, y_train)
    y_pred = candidate_model.predict(X_test)
    y_prob = candidate_model.predict_proba(X_test)[:, 1]

    candidate_metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
    }

    current_champion = get_champion_model()
    eval_result = evaluate_candidate_vs_champion(
        candidate_metrics,
        current_champion,
        metric_key="roc_auc",
        min_improvement=min_improvement,
    )

    if eval_result["promoted"]:
        # Register new version
        reg_entry = register_model(
            candidate_model,
            preprocessor,
            candidate_metrics,
            version=version_tag,
            save_as_active=save_as_active,
        )
        update_leaderboard("XGBoost", version_tag, candidate_metrics)
        logger.info(f"Candidate {version_tag} promoted to production!")
    else:
        reg_entry = None
        logger.info(f"Candidate {version_tag} retained as challenger; champion unchanged.")

    return {
        "version": version_tag,
        "candidate_metrics": candidate_metrics,
        "evaluation": eval_result,
        "registry_entry": reg_entry,
    }
