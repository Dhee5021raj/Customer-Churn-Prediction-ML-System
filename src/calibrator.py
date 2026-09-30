"""
src/calibrator.py
-----------------
Model probability calibration and reliability curve analyzer.

Assesses whether predicted churn probabilities accurately reflect empirical
churn likelihood using Expected Calibration Error (ECE), Maximum Calibration
Error (MCE), and Brier Score.

Functions:
    compute_calibration_curve(y_true, y_prob, n_bins) -> dict
    calculate_expected_calibration_error(y_true, y_prob, n_bins) -> dict
    assess_calibration_quality(ece_score) -> str
"""

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from typing import Dict, Any

from src.config import setup_logger

logger = setup_logger("calibrator")


def assess_calibration_quality(ece_score: float) -> str:
    """
    Categorizes calibration reliability from Expected Calibration Error (ECE).

    Parameters
    ----------
    ece_score : float
        Calculated ECE score in [0.0, 1.0].

    Returns
    -------
    str
        "Well-Calibrated" | "Moderately Calibrated" | "Poorly Calibrated"
    """
    if ece_score < 0.05:
        return "Well-Calibrated"
    elif ece_score <= 0.12:
        return "Moderately Calibrated"
    else:
        return "Poorly Calibrated"


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes empirical calibration curve coordinates and bin counts.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0 or 1).
    y_prob : array-like
        Predicted probabilities for the positive class.
    n_bins : int
        Number of probability discretization bins (default: 10).

    Returns
    -------
    dict
        {
            "prob_true": list of float,
            "prob_pred": list of float,
            "n_bins": int,
        }
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) == 0 or len(y_prob) == 0:
        raise ValueError("Inputs y_true and y_prob cannot be empty")

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    return {
        "prob_true": [round(float(p), 4) for p in prob_true],
        "prob_pred": [round(float(p), 4) for p in prob_pred],
        "n_bins": n_bins,
    }


def calculate_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Calculates Expected Calibration Error (ECE), MCE, and Brier Score.

    ECE = sum_{b=1}^B ( |B_b| / N ) * | acc(B_b) - conf(B_b) |
    MCE = max_{b} | acc(B_b) - conf(B_b) |

    Parameters
    ----------
    y_true : array-like
        Ground truth labels (0 or 1).
    y_prob : array-like
        Predicted probabilities.
    n_bins : int
        Number of bins (default: 10).

    Returns
    -------
    dict
        {
            "ece": float,
            "mce": float,
            "brier_score": float,
            "quality": str,
            "n_bins": int,
            "sample_count": int,
        }
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    if n == 0:
        raise ValueError("Inputs cannot be empty")

    brier = float(brier_score_loss(y_true, y_prob))

    # Binning for ECE and MCE
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        mask = bin_indices == i
        bin_size = np.sum(mask)
        if bin_size > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            diff = abs(bin_acc - bin_conf)
            ece += (bin_size / n) * diff
            mce = max(mce, diff)

    quality = assess_calibration_quality(ece)
    logger.info(f"Calibration metrics: ECE={ece:.4f}, MCE={mce:.4f}, Brier={brier:.4f} ({quality})")

    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "brier_score": round(brier, 4),
        "quality": quality,
        "n_bins": n_bins,
        "sample_count": n,
    }
