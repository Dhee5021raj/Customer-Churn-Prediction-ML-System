"""
src/fairness_checker.py
-----------------------
Model fairness and demographic bias detection module.

Computes group-level fairness metrics across sensitive attributes and
flags potential disparate impact using the 4/5ths rule (threshold: 0.80).

Functions:
    calculate_fairness_metrics(y_true, y_pred, sensitive_col, df) -> dict
    calculate_demographic_parity(selection_rates) -> dict
    calculate_equal_opportunity(tpr_rates) -> dict
    run_fairness_report(y_true, y_pred, df, sensitive_cols) -> dict
"""

import pandas as pd
import numpy as np
from typing import List

from src.config import setup_logger

logger = setup_logger("fairness_checker")

DISPARATE_IMPACT_THRESHOLD = 0.80  # 4/5ths rule


def _group_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes TPR, FPR, FNR, and selection rate for a single group."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # Recall / Equal Opportunity
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    selection_rate = (tp + fp) / len(y_pred) if len(y_pred) > 0 else 0.0

    return {
        "true_positive_rate": round(tpr, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "selection_rate": round(selection_rate, 4),
        "count": len(y_pred),
    }


def calculate_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_col: str,
    df: pd.DataFrame,
) -> dict:
    """
    Calculates per-group fairness metrics for a given sensitive attribute column.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.
    sensitive_col : str
        Column name in df representing the sensitive attribute.
    df : pd.DataFrame
        Customer DataFrame (same row order as y_true / y_pred).

    Returns
    -------
    dict
        {
            "sensitive_col": ...,
            "groups": { group_value: {metrics} },
            "demographic_parity": {...},
            "equal_opportunity": {...},
        }
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if sensitive_col not in df.columns:
        raise ValueError(f"Sensitive column '{sensitive_col}' not found in DataFrame.")

    groups = df[sensitive_col].fillna("Unknown").astype(str).values
    unique_groups = sorted(set(groups))

    group_stats = {}
    for grp in unique_groups:
        mask = groups == grp
        if mask.sum() < 5:
            logger.warning(f"Group '{grp}' has < 5 samples — metrics may be unreliable.")
        group_stats[grp] = _group_metrics(y_true[mask], y_pred[mask])

    selection_rates = {g: s["selection_rate"] for g, s in group_stats.items()}
    tpr_rates = {g: s["true_positive_rate"] for g, s in group_stats.items()}

    dp = calculate_demographic_parity(selection_rates)
    eo = calculate_equal_opportunity(tpr_rates)

    logger.info(f"Fairness check on '{sensitive_col}': DP ratio={dp['ratio']}, EO ratio={eo['ratio']}")

    return {
        "sensitive_col": sensitive_col,
        "groups": group_stats,
        "demographic_parity": dp,
        "equal_opportunity": eo,
    }


def calculate_demographic_parity(selection_rates: dict) -> dict:
    """
    Calculates the Disparate Impact ratio for demographic parity.

    Ratio = min_selection_rate / max_selection_rate.
    Passes the 4/5ths rule if ratio >= 0.80.
    """
    if not selection_rates:
        return {"ratio": None, "passes_4_5ths_rule": False, "min_group": None, "max_group": None}

    min_grp = min(selection_rates, key=selection_rates.get)
    max_grp = max(selection_rates, key=selection_rates.get)
    min_val = selection_rates[min_grp]
    max_val = selection_rates[max_grp]

    ratio = round(min_val / max_val, 4) if max_val > 0 else 1.0

    return {
        "ratio": ratio,
        "passes_4_5ths_rule": ratio >= DISPARATE_IMPACT_THRESHOLD,
        "min_group": min_grp,
        "max_group": max_grp,
        "min_selection_rate": round(min_val, 4),
        "max_selection_rate": round(max_val, 4),
    }


def calculate_equal_opportunity(tpr_rates: dict) -> dict:
    """
    Calculates Equal Opportunity ratio (True Positive Rate parity).

    Ratio = min_TPR / max_TPR. Passes if ratio >= 0.80.
    """
    if not tpr_rates:
        return {"ratio": None, "passes_equal_opportunity": False}

    min_grp = min(tpr_rates, key=tpr_rates.get)
    max_grp = max(tpr_rates, key=tpr_rates.get)
    min_val = tpr_rates[min_grp]
    max_val = tpr_rates[max_grp]

    ratio = round(min_val / max_val, 4) if max_val > 0 else 1.0

    return {
        "ratio": ratio,
        "passes_equal_opportunity": ratio >= DISPARATE_IMPACT_THRESHOLD,
        "min_group": min_grp,
        "max_group": max_grp,
    }


def run_fairness_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df: pd.DataFrame,
    sensitive_cols: List[str],
) -> dict:
    """
    Runs fairness checks across multiple sensitive columns.

    Returns
    -------
    dict
        { col: fairness_result_dict } for each column in sensitive_cols.
    """
    report = {}
    for col in sensitive_cols:
        try:
            report[col] = calculate_fairness_metrics(y_true, y_pred, col, df)
        except ValueError as e:
            logger.warning(str(e))
    return report
