import pandas as pd
import numpy as np
from typing import Dict, Any, List
from src.clv_calculator import calculate_customer_clv


def simulate_what_if_scenario(
    explainer: Any,
    base_profile: Dict[str, Any],
    modifications: Dict[str, Any],
    scenario_name: str = "Simulated Scenario"
) -> Dict[str, Any]:
    """
    Simulate counterfactual changes to a customer profile and compute net 
    churn risk delta and projected 24-Month CLV revenue savings.
    """
    modified_profile = base_profile.copy()
    modified_profile.update(modifications)
    
    # Recalculate total charges if tenure or monthly charges changed
    monthly_charges = float(modified_profile.get("monthly_charges", 75.0))
    tenure = int(modified_profile.get("tenure", 12))
    modified_profile["total_charges"] = round(monthly_charges * tenure, 2)
    
    # Predict baseline vs modified
    df_base = pd.DataFrame([base_profile])
    df_mod = pd.DataFrame([modified_profile])
    
    X_base = explainer.preprocessor.transform(df_base)
    X_mod = explainer.preprocessor.transform(df_mod)
    
    base_prob = float(explainer.model.predict_proba(X_base)[0][1])
    mod_prob = float(explainer.model.predict_proba(X_mod)[0][1])
    
    risk_delta = round(mod_prob - base_prob, 4)
    
    # CLV Impact
    base_clv = calculate_customer_clv(base_profile.get("monthly_charges", 75.0), tenure_months=24)
    mod_clv = calculate_customer_clv(monthly_charges, tenure_months=24)
    
    base_rev_risk = base_clv * base_prob
    mod_rev_risk = mod_clv * mod_prob
    net_revenue_saved = round(base_rev_risk - mod_rev_risk, 2)
    
    return {
        "scenario_name": scenario_name,
        "base_prob": round(base_prob, 4),
        "mod_prob": round(mod_prob, 4),
        "risk_delta": risk_delta,
        "risk_delta_pct": round(risk_delta * 100, 2),
        "base_rev_risk": round(base_rev_risk, 2),
        "mod_rev_risk": round(mod_rev_risk, 2),
        "net_revenue_saved": net_revenue_saved,
        "modifications": modifications
    }
