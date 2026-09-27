"""
src/risk_classifier.py
----------------------
Fine-grained risk band classifier and confidence scorer.

Maps raw churn probabilities to labelled risk bands with confidence context.

Functions:
    classify_risk_band(probability) -> dict
    annotate_dataframe_with_risk_bands(df, prob_col) -> pd.DataFrame
"""

import pandas as pd
from src.config import setup_logger

logger = setup_logger("risk_classifier")

# Band definitions: (lower_bound_inclusive, upper_bound_exclusive, band, label, color_hint)
RISK_BANDS = [
    (0.00, 0.20, "Safe",     "Likely Retained — No Immediate Action",       "green"),
    (0.20, 0.40, "Low",      "Low Churn Risk — Monitor Quarterly",           "teal"),
    (0.40, 0.60, "Moderate", "Moderate Risk — Schedule Check-In",            "yellow"),
    (0.60, 0.80, "High",     "High Churn Risk — Prioritise Retention Offer", "orange"),
    (0.80, 1.01, "Critical", "Critical — Immediate Intervention Required",   "red"),
]

# Distance from band edge required to score as "High" confidence
CONFIDENCE_MARGIN = 0.10


def classify_risk_band(probability: float) -> dict:
    """
    Classifies a single churn probability into a risk band with confidence.

    Parameters
    ----------
    probability : float
        Churn probability in [0.0, 1.0].

    Returns
    -------
    dict with keys:
        band         : str  — "Safe" | "Low" | "Moderate" | "High" | "Critical"
        label        : str  — human-readable action description
        color        : str  — color hint for UI rendering
        confidence   : str  — "High" | "Medium" | "Low"
        probability  : float
    """
    probability = max(0.0, min(1.0, float(probability)))

    selected = RISK_BANDS[-1]  # default to Critical
    for lo, hi, band, label, color in RISK_BANDS:
        if lo <= probability < hi:
            selected = (lo, hi, band, label, color)
            break

    lo, hi, band, label, color = selected
    band_width = hi - lo
    dist_to_lower = probability - lo
    dist_to_upper = hi - probability

    min_dist = round(min(dist_to_lower, dist_to_upper), 6)
    if min_dist >= CONFIDENCE_MARGIN:
        confidence = "High"
    elif min_dist >= CONFIDENCE_MARGIN / 2:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "band": band,
        "label": label,
        "color": color,
        "confidence": confidence,
        "probability": round(probability, 4),
    }


def annotate_dataframe_with_risk_bands(
    df: pd.DataFrame,
    prob_col: str = "churn_probability",
) -> pd.DataFrame:
    """
    Annotates a DataFrame with risk band, label, color, and confidence columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `prob_col`.
    prob_col : str
        Column name holding churn probabilities.

    Returns
    -------
    pd.DataFrame
        Copy of df with added columns: risk_band, risk_label, risk_color, confidence.
    """
    if prob_col not in df.columns:
        raise ValueError(f"Column '{prob_col}' not found in DataFrame.")

    df = df.copy()
    classifications = df[prob_col].apply(classify_risk_band)

    df["risk_band"]  = classifications.apply(lambda x: x["band"])
    df["risk_label"] = classifications.apply(lambda x: x["label"])
    df["risk_color"] = classifications.apply(lambda x: x["color"])
    df["confidence"] = classifications.apply(lambda x: x["confidence"])

    logger.info(
        f"Annotated {len(df)} records with risk bands. "
        f"Distribution: {df['risk_band'].value_counts().to_dict()}"
    )
    return df
