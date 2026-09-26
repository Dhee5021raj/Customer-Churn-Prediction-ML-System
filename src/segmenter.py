"""
src/segmenter.py
----------------
K-Means based customer cohort segmentation engine.

Functions:
    segment_customers(df, n_clusters) -> pd.DataFrame with 'customer_segment' column
    get_segment_summary(df_with_segments) -> pd.DataFrame with per-segment aggregated stats
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import NUMERICAL_FEATURES, setup_logger

logger = setup_logger("segmenter")

# Feature subset used for clustering (behavioural + financial signals)
CLUSTER_FEATURES = [
    "monthly_charges",
    "tenure_months",
    "num_support_tickets",
    "monthly_usage_gb",
    "num_products",
]

# Human-readable segment labels assigned by cluster centre profile
SEGMENT_LABEL_MAP = {
    "high_value_loyal": "High-Value Loyal",
    "at_risk_spenders": "At-Risk Spenders",
    "budget_churn_risk": "Budget Churn Risk",
    "inactive_low_spend": "Inactive Low-Spend",
}


def _assign_segment_label(center: np.ndarray, centers: np.ndarray) -> str:
    """
    Assigns a human-readable label to each cluster centre based on relative
    position along monthly_charges (index 0) and tenure_months (index 1).
    """
    avg_charge = center[0]
    avg_tenure = center[1]
    all_charges = centers[:, 0]
    all_tenures = centers[:, 1]

    high_charge = avg_charge >= np.median(all_charges)
    high_tenure = avg_tenure >= np.median(all_tenures)

    if high_charge and high_tenure:
        return SEGMENT_LABEL_MAP["high_value_loyal"]
    elif high_charge and not high_tenure:
        return SEGMENT_LABEL_MAP["at_risk_spenders"]
    elif not high_charge and high_tenure:
        return SEGMENT_LABEL_MAP["inactive_low_spend"]
    else:
        return SEGMENT_LABEL_MAP["budget_churn_risk"]


def segment_customers(df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> pd.DataFrame:
    """
    Segments customers into behavioural cohorts using K-Means clustering.

    Parameters
    ----------
    df : pd.DataFrame
        Customer dataset. Must contain CLUSTER_FEATURES columns.
    n_clusters : int
        Number of K-Means clusters (default: 4).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with an added 'customer_segment' column.
    """
    df = df.copy()

    available_features = [f for f in CLUSTER_FEATURES if f in df.columns]
    if len(available_features) < 2:
        logger.warning("Insufficient cluster features available. Skipping segmentation.")
        df["customer_segment"] = "Unknown"
        return df

    X = df[available_features].fillna(df[available_features].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    cluster_ids = km.fit_predict(X_scaled)

    # Build label map using rank-based assignment across all centres so every
    # cluster always gets a distinct human-readable name regardless of the
    # distribution of the random sample.
    centers_original = scaler.inverse_transform(km.cluster_centers_)

    RANK_LABELS = [
        SEGMENT_LABEL_MAP["high_value_loyal"],    # rank-0: highest combined score
        SEGMENT_LABEL_MAP["at_risk_spenders"],    # rank-1
        SEGMENT_LABEL_MAP["inactive_low_spend"],  # rank-2
        SEGMENT_LABEL_MAP["budget_churn_risk"],   # rank-3: lowest combined score
    ]

    # Score each cluster centre as charge + tenure (both normalised 0-1)
    charges = centers_original[:, 0]
    tenures = centers_original[:, 1]
    ch_range = charges.max() - charges.min() or 1
    te_range = tenures.max() - tenures.min() or 1
    scores = (charges - charges.min()) / ch_range + (tenures - tenures.min()) / te_range

    # Argsort descending → highest score → RANK_LABELS[0]
    ranking = np.argsort(-scores)  # indices sorted by score desc
    label_map = {int(cluster_idx): RANK_LABELS[rank] for rank, cluster_idx in enumerate(ranking)}

    df["customer_segment"] = [label_map[cid] for cid in cluster_ids]
    logger.info(f"Segmented {len(df)} customers into {n_clusters} cohorts: {set(label_map.values())}")
    return df


def get_segment_summary(df_with_segments: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates per-segment statistics for reporting.

    Parameters
    ----------
    df_with_segments : pd.DataFrame
        Output of segment_customers(). Must contain 'customer_segment'.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per segment.
    """
    if "customer_segment" not in df_with_segments.columns:
        raise ValueError("DataFrame must contain 'customer_segment' column. Run segment_customers() first.")

    agg: dict = {
        "customer_count": ("customer_segment", "count"),
    }

    if "monthly_charges" in df_with_segments.columns:
        agg["avg_monthly_charges"] = ("monthly_charges", "mean")
    if "tenure_months" in df_with_segments.columns:
        agg["avg_tenure_months"] = ("tenure_months", "mean")
    if "churn_probability" in df_with_segments.columns:
        agg["avg_churn_probability"] = ("churn_probability", "mean")
    if "clv" in df_with_segments.columns:
        agg["avg_clv"] = ("clv", "mean")
        agg["total_clv_at_risk"] = ("clv", "sum")

    summary = (
        df_with_segments.groupby("customer_segment")
        .agg(**agg)
        .reset_index()
        .sort_values("avg_churn_probability" if "avg_churn_probability" in agg else "customer_count", ascending=False)
    )

    for col in summary.select_dtypes(include="float").columns:
        summary[col] = summary[col].round(3)

    return summary
