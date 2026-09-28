import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List
from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


def calculate_feature_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Compute statistical feature drift between baseline training dataset 
    and current inference dataset using Kolmogorov-Smirnov test for numericals 
    and Distribution Shift for categoricals.
    """
    drift_results = {}
    drifted_count = 0
    total_features = 0

    # Numerical Features (KS-test)
    for col in NUMERICAL_FEATURES:
        if col in baseline_df.columns and col in current_df.columns:
            total_features += 1
            stat, p_value = stats.ks_2samp(baseline_df[col].dropna(), current_df[col].dropna())
            is_drifted = p_value < alpha
            if is_drifted:
                drifted_count += 1
                
            drift_results[col] = {
                "type": "numerical",
                "test": "KS-test",
                "statistic": round(float(stat), 4),
                "p_value": round(float(p_value), 4),
                "is_drifted": is_drifted
            }

    # Categorical Features (Distribution Diff)
    for col in CATEGORICAL_FEATURES:
        if col in baseline_df.columns and col in current_df.columns:
            total_features += 1
            base_dist = baseline_df[col].value_counts(normalize=True)
            curr_dist = current_df[col].value_counts(normalize=True)
            
            # Combine categories
            all_cats = list(set(base_dist.index).union(set(curr_dist.index)))
            max_diff = 0.0
            for cat in all_cats:
                p1 = base_dist.get(cat, 0.0)
                p2 = curr_dist.get(cat, 0.0)
                max_diff = max(max_diff, abs(p1 - p2))
                
            is_drifted = max_diff > 0.15  # Shift threshold > 15%
            if is_drifted:
                drifted_count += 1
                
            drift_results[col] = {
                "type": "categorical",
                "test": "Max Category Shift",
                "max_shift": round(float(max_diff), 4),
                "is_drifted": is_drifted
            }

    return {
        "total_features": total_features,
        "drifted_features_count": drifted_count,
        "drift_detected": drifted_count > 0,
        "feature_details": drift_results
    }
