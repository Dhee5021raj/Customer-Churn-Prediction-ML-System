import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


def calculate_customer_clv(monthly_charges: float, tenure_months: int = 24) -> float:
    """Calculate estimated Customer Lifetime Value (CLV) over a projection horizon."""
    return round(float(monthly_charges) * tenure_months, 2)


def calculate_clv_risk(
    df: pd.DataFrame,
    churn_probabilities: np.ndarray,
    clv_horizon_months: int = 24
) -> pd.DataFrame:
    """
    Annotate customer DataFrame with estimated CLV, predicted risk level, 
    and estimated financial revenue at risk (CLV * Churn Probability).
    """
    df_result = df.copy()
    
    monthly_charges = df_result["monthly_charges"].values
    clv_projected = monthly_charges * clv_horizon_months
    revenue_at_risk = clv_projected * churn_probabilities
    
    df_result["estimated_clv"] = np.round(clv_projected, 2)
    df_result["churn_probability"] = np.round(churn_probabilities, 4)
    df_result["clv_revenue_at_risk"] = np.round(revenue_at_risk, 2)
    
    df_result["risk_tier"] = pd.cut(
        churn_probabilities,
        bins=[-0.01, 0.30, 0.60, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )
    
    return df_result


def get_clv_risk_summary(df_annotated: pd.DataFrame) -> Dict[str, Any]:
    """Calculate financial summary KPIs for customer portfolio."""
    total_clv = float(df_annotated["estimated_clv"].sum())
    total_revenue_at_risk = float(df_annotated["clv_revenue_at_risk"].sum())
    
    high_risk_df = df_annotated[df_annotated["risk_tier"] == "High Risk"]
    high_risk_revenue_loss = float(high_risk_df["clv_revenue_at_risk"].sum())
    high_risk_count = int(len(high_risk_df))
    
    return {
        "total_portfolio_clv": round(total_clv, 2),
        "total_revenue_at_risk": round(total_revenue_at_risk, 2),
        "high_risk_count": high_risk_count,
        "high_risk_revenue_loss": round(high_risk_revenue_loss, 2),
        "at_risk_percentage": round((total_revenue_at_risk / max(total_clv, 1.0)) * 100, 2)
    }
