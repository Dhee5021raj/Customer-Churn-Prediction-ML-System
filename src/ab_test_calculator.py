"""
src/ab_test_calculator.py
-------------------------
Customer retention A/B test power and statistical significance calculator.

Provides experimental sample size estimation and two-proportion z-test evaluation
for customer retention interventions, pricing discounts, and onboarding campaigns.

Functions:
    calculate_sample_size_for_retention_test(...) -> dict
    evaluate_ab_test_results(...) -> dict
"""

import numpy as np
from scipy import stats
from typing import Dict, Any

from src.config import setup_logger

logger = setup_logger("ab_test_calculator")


def calculate_sample_size_for_retention_test(
    baseline_churn_rate: float,
    expected_reduction_pct: float = 0.20,
    alpha: float = 0.05,
    power: float = 0.80,
) -> Dict[str, Any]:
    """
    Computes required sample size per treatment arm for a two-proportion test.

    Parameters
    ----------
    baseline_churn_rate : float
        Current control group churn rate (e.g. 0.25 for 25%).
    expected_reduction_pct : float
        Expected relative percentage reduction in churn (e.g. 0.20 for 20% relative drop).
    alpha : float
        Significance level (Type I error rate, default: 0.05).
    power : float
        Statistical power 1 - beta (default: 0.80).

    Returns
    -------
    dict
        Sample size breakdown per arm and total experiment volume.
    """
    p1 = max(0.01, min(0.99, float(baseline_churn_rate)))
    expected_reduction_pct = max(0.01, min(0.90, float(expected_reduction_pct)))

    p2 = p1 * (1.0 - expected_reduction_pct)
    p_bar = (p1 + p2) / 2.0

    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_beta = stats.norm.ppf(power)

    numerator = (
        z_alpha * np.sqrt(2.0 * p_bar * (1.0 - p_bar))
        + z_beta * np.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    ) ** 2
    denominator = (p1 - p2) ** 2

    n_per_variant = int(np.ceil(numerator / denominator))
    total_n = n_per_variant * 2

    logger.info(
        f"A/B Power calculation: baseline={p1:.2f}, target={p2:.2f}, "
        f"n_per_arm={n_per_variant:,} (Total: {total_n:,})"
    )

    return {
        "sample_size_per_variant": n_per_variant,
        "total_sample_size": total_n,
        "baseline_churn_rate": round(p1, 4),
        "target_churn_rate": round(p2, 4),
        "absolute_reduction": round(float(p1 - p2), 4),
        "relative_reduction_pct": round(expected_reduction_pct * 100, 1),
        "alpha": alpha,
        "power": power,
    }


def evaluate_ab_test_results(
    control_size: int,
    control_churns: int,
    variant_size: int,
    variant_churns: int,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Evaluates completed retention A/B test results using a two-proportion z-test.

    Parameters
    ----------
    control_size : int
        Total customers in Control group (no outreach or baseline).
    control_churns : int
        Number of churned customers in Control group.
    variant_size : int
        Total customers in Variant group (received retention offer).
    variant_churns : int
        Number of churned customers in Variant group.
    alpha : float
        Significance threshold (default: 0.05).

    Returns
    -------
    dict
        Hypothesis test metrics, z-statistic, p-value, and rollout recommendation.
    """
    if control_size <= 0 or variant_size <= 0:
        raise ValueError("Group sizes must be greater than zero.")

    p_control = control_churns / control_size
    p_variant = variant_churns / variant_size

    pooled_p = (control_churns + variant_churns) / (control_size + variant_size)
    se = np.sqrt(pooled_p * (1.0 - pooled_p) * (1.0 / control_size + 1.0 / variant_size))

    if se == 0:
        z_stat = 0.0
        p_val = 1.0
    else:
        z_stat = float((p_control - p_variant) / se)
        # Two-sided p-value
        p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))

    is_sig = p_val < alpha and z_stat > 0
    abs_diff = p_control - p_variant
    rel_diff_pct = (abs_diff / p_control * 100) if p_control > 0 else 0.0

    if is_sig:
        recommendation = (
            f"Statistically Significant Winner: Variant reduced churn by {rel_diff_pct:.1f}% "
            f"(p={p_val:.4f}). Recommended to graduate offer to 100% of target segment."
        )
    elif z_stat < 0 and p_val < alpha:
        recommendation = (
            f"Statistically Significant Negative Effect: Variant increased churn by {abs(rel_diff_pct):.1f}% "
            f"(p={p_val:.4f}). Immediately discontinue offer (Sleeping Dogs triggered)."
        )
    else:
        recommendation = (
            f"Inconclusive Result: Observed difference is not statistically significant (p={p_val:.4f} >= {alpha}). "
            "Continue test to reach target sample power or refine offer value."
        )

    logger.info(f"A/B Evaluation: p_val={p_val:.4f}, z={z_stat:.2f}, significant={is_sig}")

    return {
        "control_churn_rate": round(float(p_control), 4),
        "variant_churn_rate": round(float(p_variant), 4),
        "absolute_reduction": round(float(abs_diff), 4),
        "relative_reduction_pct": round(float(rel_diff_pct), 1),
        "z_statistic": round(z_stat, 4),
        "p_value": round(p_val, 4),
        "is_statistically_significant": is_sig,
        "recommendation": recommendation,
    }
