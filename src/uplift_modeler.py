"""
src/uplift_modeler.py
---------------------
Customer churn uplift and incrementality sensitivity modeler.

Classifies customers into classic causal uplift quadrants (Persuadables,
Sure Things, Lost Causes, and Sleeping Dogs) to maximize retention campaign ROI
and suppress wasteful or counter-productive outreach.

Functions:
    calculate_customer_uplift_score(base_prob, treated_prob) -> float
    classify_uplift_quadrant(base_prob, uplift_score) -> dict
    segment_portfolio_by_uplift(df, base_prob_col, treated_prob_col) -> pd.DataFrame
    calculate_uplift_efficiency_metrics(annotated_df) -> dict
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from src.config import setup_logger

logger = setup_logger("uplift_modeler")

UPLIFT_DEFINITIONS = {
    "Persuadables": {
        "description": "High churn risk who respond strongly to intervention.",
        "targeting_priority": "P1 - High Priority Target",
        "action": "Prioritize aggressive retention offer & concierge outreach. Maximum financial ROI.",
        "color": "green",
    },
    "Sure Things": {
        "description": "Customers likely to remain active without costly intervention.",
        "targeting_priority": "P3 - Standard Organic Engagement",
        "action": "Do not offer costly financial incentives. Standard loyalty communications only.",
        "color": "blue",
    },
    "Lost Causes": {
        "description": "High churn risk customers unresponsive to retention offers.",
        "targeting_priority": "P4 - Low ROI Prospect",
        "action": "Avoid expensive promotional discounts. Investigate macro structural product blockers.",
        "color": "gray",
    },
    "Sleeping Dogs": {
        "description": "Customers triggered to cancel when contacted (negative treatment effect).",
        "targeting_priority": "Do Not Disturb (Suppressed)",
        "action": "Strictly suppress from marketing & outreach campaigns. High churn acceleration risk.",
        "color": "red",
    },
}


def calculate_customer_uplift_score(base_prob: float, treated_prob: float) -> float:
    """
    Computes incremental treatment effect tau = P(churn|untreated) - P(churn|treated).

    Positive tau indicates churn risk reduction from intervention.
    """
    base_prob = max(0.0, min(1.0, float(base_prob)))
    treated_prob = max(0.0, min(1.0, float(treated_prob)))
    return round(float(base_prob - treated_prob), 4)


def classify_uplift_quadrant(base_prob: float, uplift_score: float) -> Dict[str, Any]:
    """
    Categorizes a customer into an uplift quadrant based on baseline risk and treatment lift.

    Parameters
    ----------
    base_prob : float
        Baseline probability of churning without treatment (0.0 to 1.0).
    uplift_score : float
        Net reduction in churn probability with treatment.

    Returns
    -------
    dict
        Structured quadrant metadata including quadrant, targeting_priority, action, and color.
    """
    base_prob = max(0.0, min(1.0, float(base_prob)))

    if uplift_score < -0.01:
        quadrant = "Sleeping Dogs"
    elif base_prob >= 0.35 and uplift_score >= 0.08:
        quadrant = "Persuadables"
    elif base_prob < 0.35:
        quadrant = "Sure Things"
    else:
        quadrant = "Lost Causes"

    meta = UPLIFT_DEFINITIONS[quadrant]

    return {
        "quadrant": quadrant,
        "uplift_score": round(float(uplift_score), 4),
        "base_probability": round(float(base_prob), 4),
        "targeting_priority": meta["targeting_priority"],
        "action": meta["action"],
        "color": meta["color"],
    }


def segment_portfolio_by_uplift(
    df: pd.DataFrame,
    base_prob_col: str = "churn_probability",
    treated_prob_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Annotates customer DataFrame with uplift scores and quadrant assignments.

    If treated_prob_col is missing, synthesizes treated_prob from contract/tenure
    treatment elasticity.
    """
    if base_prob_col not in df.columns:
        raise ValueError(f"Base probability column '{base_prob_col}' not found in DataFrame.")

    df_out = df.copy()

    if treated_prob_col and treated_prob_col in df_out.columns:
        treated_probs = df_out[treated_prob_col].values
    else:
        # Realistic retention intervention elasticity model:
        # Base elasticity is ~25% relative risk reduction, modulated by tenure
        base_probs = df_out[base_prob_col].values
        tenure = df_out["tenure"].values if "tenure" in df_out.columns else np.full(len(df_out), 12.0)
        reduction_rate = 0.25 + 0.05 * np.clip(tenure / 24.0, 0.0, 1.0)
        treated_probs = np.clip(base_probs * (1.0 - reduction_rate), 0.01, 0.99)
        df_out["simulated_treated_prob"] = np.round(treated_probs, 4)

    uplift_scores = [
        calculate_customer_uplift_score(b, t)
        for b, t in zip(df_out[base_prob_col].values, treated_probs)
    ]
    quadrants = [
        classify_uplift_quadrant(b, u)
        for b, u in zip(df_out[base_prob_col].values, uplift_scores)
    ]

    df_out["uplift_score"] = uplift_scores
    df_out["uplift_quadrant"] = [q["quadrant"] for q in quadrants]
    df_out["targeting_priority"] = [q["targeting_priority"] for q in quadrants]
    df_out["uplift_action"] = [q["action"] for q in quadrants]

    logger.info(
        f"Segmented {len(df_out)} customers by uplift. "
        f"Distribution: {df_out['uplift_quadrant'].value_counts().to_dict()}"
    )
    return df_out


def calculate_uplift_efficiency_metrics(annotated_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes efficiency gains of targeting Persuadables exclusively vs unsegmented outreach.
    """
    if "uplift_quadrant" not in annotated_df.columns:
        raise ValueError("DataFrame must be annotated with 'uplift_quadrant'.")

    counts = annotated_df["uplift_quadrant"].value_counts().to_dict()
    total = len(annotated_df)

    persuadables = counts.get("Persuadables", 0)
    sure_things = counts.get("Sure Things", 0)
    lost_causes = counts.get("Lost Causes", 0)
    sleeping_dogs = counts.get("Sleeping Dogs", 0)

    # Average uplift score in Persuadables vs entire dataset
    avg_uplift_persuadables = (
        float(annotated_df[annotated_df["uplift_quadrant"] == "Persuadables"]["uplift_score"].mean())
        if persuadables > 0
        else 0.0
    )
    avg_uplift_all = float(annotated_df["uplift_score"].mean()) if "uplift_score" in annotated_df.columns else 0.0

    # Incremental efficiency factor
    targeting_efficiency = (
        round(avg_uplift_persuadables / avg_uplift_all, 2)
        if avg_uplift_all > 0
        else 1.0
    )

    return {
        "total_customers": total,
        "quadrant_counts": {
            "Persuadables": persuadables,
            "Sure Things": sure_things,
            "Lost Causes": lost_causes,
            "Sleeping Dogs": sleeping_dogs,
        },
        "persuadable_percentage": round((persuadables / total) * 100, 1) if total > 0 else 0.0,
        "avg_uplift_persuadables": round(avg_uplift_persuadables, 4),
        "targeting_efficiency_multiplier": targeting_efficiency,
        "recommended_contact_volume": persuadables,
        "budget_waste_prevented_pct": round(((sure_things + lost_causes + sleeping_dogs) / total) * 100, 1) if total > 0 else 0.0,
    }
