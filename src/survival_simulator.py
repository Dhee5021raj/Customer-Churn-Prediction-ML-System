"""
src/survival_simulator.py
-------------------------
Actuarial customer retention survival curve and hazard rate simulator.

Models customer survival probability S(t) and instantaneous hazard rate h(t)
over multi-month horizons incorporating contract commitment and tenure seasoning.

Functions:
    simulate_customer_survival_curve(...) -> list[dict]
    calculate_expected_customer_lifetime(survival_curve) -> dict
    compare_contract_survival_curves(...) -> pd.DataFrame
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any

from src.config import setup_logger

logger = setup_logger("survival_simulator")


def simulate_customer_survival_curve(
    base_churn_prob: float,
    tenure_months: int = 12,
    contract_type: str = "Month-to-month",
    periods: int = 24,
) -> List[Dict[str, Any]]:
    """
    Computes expected customer survival curve S(t) and hazard rate h(t).

    Parameters
    ----------
    base_churn_prob : float
        Customer model churn probability (in [0.0, 1.0]).
    tenure_months : int
        Current tenure of the customer in months.
    contract_type : str
        Contract type ('Month-to-month', 'One year', 'Two year').
    periods : int
        Number of forward projection months (default: 24).

    Returns
    -------
    list of dict
        Each dict has: 'month', 'survival_probability', 'hazard_rate', 'cumulative_churn_probability'.
    """
    if periods < 1:
        raise ValueError("periods must be >= 1")

    base_prob = max(0.01, min(0.99, float(base_churn_prob)))
    monthly_base_hazard = -np.log(1.0 - base_prob) / 12.0  # approximate monthly hazard

    curve = []
    cumulative_survival = 1.0

    for t in range(1, periods + 1):
        effective_tenure = tenure_months + t

        # Tenure seasoning effect: hazard reduces logarithmically as customer matures
        tenure_factor = 1.0 / (1.0 + 0.05 * np.log1p(effective_tenure))

        # Contract commitment modifier:
        contract_norm = str(contract_type).lower()
        if "two year" in contract_norm or "2 year" in contract_norm:
            # Low hazard during active term, spike at renewal month 24
            if t % 24 == 0:
                contract_factor = 1.2
            else:
                contract_factor = 0.25
        elif "one year" in contract_norm or "1 year" in contract_norm:
            # Low hazard during term, spike at renewal month 12
            if t % 12 == 0:
                contract_factor = 1.15
            else:
                contract_factor = 0.45
        else:
            # Month-to-month: standard monthly exposure
            contract_factor = 1.0

        monthly_hazard = min(0.90, monthly_base_hazard * tenure_factor * contract_factor)
        monthly_survival_step = 1.0 - monthly_hazard
        cumulative_survival *= monthly_survival_step
        cumulative_survival = max(0.01, min(1.0, cumulative_survival))

        curve.append({
            "month": t,
            "survival_probability": round(float(cumulative_survival), 4),
            "hazard_rate": round(float(monthly_hazard), 4),
            "cumulative_churn_probability": round(float(1.0 - cumulative_survival), 4),
        })

    logger.info(
        f"Simulated survival curve ({contract_type}, {periods}m): "
        f"S(12)={curve[min(11, len(curve)-1)]['survival_probability']:.2f}, "
        f"S(24)={curve[-1]['survival_probability']:.2f}"
    )
    return curve


def calculate_expected_customer_lifetime(
    survival_curve: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Computes actuarial survival statistics from a discrete survival curve.

    Calculates:
        - median_survival_months: month where S(t) drops to <= 0.50
        - restricted_mean_survival_months: area under the survival curve
        - survival_at_12m: S(12)
        - survival_at_24m: S(24)
    """
    if not survival_curve:
        raise ValueError("survival_curve cannot be empty")

    median_months = None
    area_under_curve = 0.0

    for step in survival_curve:
        s = step["survival_probability"]
        m = step["month"]
        area_under_curve += s
        if median_months is None and s <= 0.50:
            median_months = m

    if median_months is None:
        # Customer median survival exceeds projection horizon
        median_months = f">{survival_curve[-1]['month']}"

    s_12 = survival_curve[11]["survival_probability"] if len(survival_curve) >= 12 else survival_curve[-1]["survival_probability"]
    s_24 = survival_curve[23]["survival_probability"] if len(survival_curve) >= 24 else survival_curve[-1]["survival_probability"]

    return {
        "median_survival_months": median_months,
        "restricted_mean_survival_months": round(float(area_under_curve), 1),
        "survival_at_12m": s_12,
        "survival_at_24m": s_24,
    }


def compare_contract_survival_curves(
    base_churn_prob: float,
    tenure_months: int = 12,
    periods: int = 24,
) -> pd.DataFrame:
    """
    Generates comparative survival curves across all standard contract categories.

    Returns
    -------
    pd.DataFrame
        Table with 'month', 'Month-to-month', 'One year', 'Two year' columns.
    """
    contracts = ["Month-to-month", "One year", "Two year"]
    data = {"month": list(range(1, periods + 1))}

    for c in contracts:
        curve = simulate_customer_survival_curve(base_churn_prob, tenure_months, c, periods)
        data[c] = [pt["survival_probability"] for pt in curve]

    df_comp = pd.DataFrame(data)
    return df_comp
