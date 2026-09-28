"""
src/retention_roi.py
--------------------
Customer retention campaign financial ROI and payback simulator.

Calculates campaign economics including net financial benefit, ROI %,
break-even saved customers, and optimal budget distribution across risk tiers.

Functions:
    calculate_campaign_roi(...) -> dict
    simulate_portfolio_retention_roi(df, ...) -> dict
    get_budget_allocation_recommendation(total_budget, df) -> pd.DataFrame
"""

import pandas as pd
from typing import Dict, Any, Optional

from src.config import setup_logger

logger = setup_logger("retention_roi")


def calculate_campaign_roi(
    total_customers_targeted: int,
    avg_clv: float,
    cost_per_contact: float = 5.0,
    offer_incentive_cost: float = 30.0,
    success_rate: float = 0.25,
) -> Dict[str, Any]:
    """
    Computes financial ROI metrics for a retention campaign.

    Parameters
    ----------
    total_customers_targeted : int
        Number of customers contacted in the retention outreach.
    avg_clv : float
        Average customer lifetime value of targeted customers.
    cost_per_contact : float
        Fixed outreach cost per customer (e.g. SMS, agent call time, email).
    offer_incentive_cost : float
        Cost of the promotional discount/gift given to customers who accept.
    success_rate : float
        Fraction of targeted customers successfully saved from churning (0.0 to 1.0).

    Returns
    -------
    dict
        Structured financial summary containing total_cost, revenue_saved,
        net_benefit, roi_pct, break_even_customers, and payback_ratio.
    """
    if total_customers_targeted <= 0:
        return {
            "total_customers_targeted": 0,
            "customers_saved": 0,
            "total_campaign_cost": 0.0,
            "gross_revenue_saved": 0.0,
            "net_financial_benefit": 0.0,
            "roi_percentage": 0.0,
            "break_even_customers": 0,
            "payback_ratio": 0.0,
        }

    success_rate = max(0.0, min(1.0, float(success_rate)))
    customers_saved = round(total_customers_targeted * success_rate)

    outreach_cost = total_customers_targeted * cost_per_contact
    incentive_cost = customers_saved * offer_incentive_cost
    total_campaign_cost = outreach_cost + incentive_cost

    gross_revenue_saved = customers_saved * avg_clv
    net_financial_benefit = gross_revenue_saved - total_campaign_cost

    roi_percentage = (
        (net_financial_benefit / total_campaign_cost * 100)
        if total_campaign_cost > 0
        else 0.0
    )

    break_even_customers = (
        round(total_campaign_cost / avg_clv)
        if avg_clv > 0
        else total_customers_targeted
    )

    payback_ratio = (
        round(gross_revenue_saved / total_campaign_cost, 2)
        if total_campaign_cost > 0
        else 0.0
    )

    return {
        "total_customers_targeted": total_customers_targeted,
        "customers_saved": customers_saved,
        "total_campaign_cost": round(total_campaign_cost, 2),
        "gross_revenue_saved": round(gross_revenue_saved, 2),
        "net_financial_benefit": round(net_financial_benefit, 2),
        "roi_percentage": round(roi_percentage, 1),
        "break_even_customers": break_even_customers,
        "payback_ratio": payback_ratio,
    }


def simulate_portfolio_retention_roi(
    df: pd.DataFrame,
    cost_per_contact: float = 5.0,
    offer_incentive_cost: float = 30.0,
    success_rate: float = 0.25,
    risk_col: str = "risk_tier",
    clv_col: str = "clv",
) -> Dict[str, Any]:
    """
    Simulates retention ROI segmented across risk tiers.

    Parameters
    ----------
    df : pd.DataFrame
        Customer DataFrame containing risk tier and CLV estimates.
    cost_per_contact : float
        Outreach cost per targeted customer.
    offer_incentive_cost : float
        Incentive/discount cost for saved customers.
    success_rate : float
        Expected retention success rate (default: 0.25).
    risk_col : str
        Column denoting risk category.
    clv_col : str
        Column denoting customer lifetime value.

    Returns
    -------
    dict:
        "overall": summary dict of total portfolio ROI
        "tier_breakdown": pd.DataFrame of metrics per risk tier
    """
    if risk_col not in df.columns or clv_col not in df.columns:
        # Fallback if specific columns are missing
        avg_clv = df[clv_col].mean() if clv_col in df.columns else 1200.0
        overall = calculate_campaign_roi(
            len(df), avg_clv, cost_per_contact, offer_incentive_cost, success_rate
        )
        return {"overall": overall, "tier_breakdown": pd.DataFrame([overall])}

    tier_records = []
    for tier, group in df.groupby(risk_col):
        count = len(group)
        tier_clv = group[clv_col].mean()
        metrics = calculate_campaign_roi(
            count, tier_clv, cost_per_contact, offer_incentive_cost, success_rate
        )
        metrics["risk_tier"] = tier
        tier_records.append(metrics)

    tier_df = pd.DataFrame(tier_records)
    # Order columns neatly
    cols = ["risk_tier"] + [c for c in tier_df.columns if c != "risk_tier"]
    tier_df = tier_df[cols]

    total_targeted = len(df)
    total_clv = df[clv_col].mean()
    overall = calculate_campaign_roi(
        total_targeted, total_clv, cost_per_contact, offer_incentive_cost, success_rate
    )

    logger.info(
        f"Simulated retention ROI across {len(tier_df)} tiers. "
        f"Overall Net Benefit: ${overall['net_financial_benefit']:,.2f} (ROI: {overall['roi_percentage']}%)"
    )

    return {"overall": overall, "tier_breakdown": tier_df}


def get_budget_allocation_recommendation(
    total_budget: float,
    df: pd.DataFrame,
    cost_per_contact: float = 5.0,
    offer_incentive_cost: float = 30.0,
    risk_col: str = "risk_tier",
    clv_col: str = "clv",
) -> pd.DataFrame:
    """
    Recommends optimal retention budget allocation prioritizing highest CLV risk.

    High Risk / Critical tiers receive 60% of budget, Medium/Moderate 30%, Low/Safe 10%.
    """
    tiers = df[risk_col].unique() if risk_col in df.columns else ["High Risk", "Medium Risk", "Low Risk"]
    
    # Priority weights
    weight_map = {
        "Critical": 0.40,
        "High Risk": 0.35,
        "High": 0.35,
        "Medium Risk": 0.20,
        "Moderate": 0.20,
        "Low Risk": 0.05,
        "Low": 0.05,
        "Safe": 0.00,
    }

    allocations = []
    total_weight = sum(weight_map.get(t, 0.10) for t in tiers) or 1.0

    for tier in tiers:
        weight = weight_map.get(tier, 0.10) / total_weight
        allocated_budget = round(total_budget * weight, 2)
        unit_cost = cost_per_contact + offer_incentive_cost * 0.25
        targetable_customers = int(allocated_budget / unit_cost) if unit_cost > 0 else 0

        allocations.append({
            "risk_tier": tier,
            "allocated_budget": allocated_budget,
            "budget_share_pct": round(weight * 100, 1),
            "estimated_targetable_customers": targetable_customers,
        })

    alloc_df = pd.DataFrame(allocations).sort_values("allocated_budget", ascending=False)
    return alloc_df
