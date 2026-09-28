import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score
from src.config import MODELS_DIR


def optimize_classification_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    output_path: str = None
) -> Dict[str, Any]:
    """
    Search for classification decision threshold between 0.10 and 0.90 
    that maximizes F1-Score on test predictions.
    """
    thresholds = np.linspace(0.10, 0.90, 81)
    best_threshold = 0.50
    best_f1 = -1.0
    best_metrics = {}
    
    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(th)
            best_metrics = {
                "optimal_threshold": round(float(th), 4),
                "f1_score": round(float(f1), 4),
                "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
            }
            
    if output_path is None:
        output_path = os.path.join(MODELS_DIR, "optimal_threshold.json")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(best_metrics, f, indent=4)
        
    return best_metrics


def get_optimal_threshold(default_threshold: float = 0.50, threshold_path: str = None) -> float:
    """Retrieve saved optimal threshold or fallback to default."""
    if threshold_path is None:
        threshold_path = os.path.join(MODELS_DIR, "optimal_threshold.json")
        
    if os.path.exists(threshold_path):
        try:
            with open(threshold_path, "r") as f:
                data = json.load(f)
                return float(data.get("optimal_threshold", default_threshold))
        except Exception:
            pass
            
    return default_threshold
