import pandas as pd
import numpy as np
from typing import Dict, Any, List


def generate_customer_intervention_tasklist(df_annotated: pd.DataFrame) -> pd.DataFrame:
    """
    Format actionable customer intervention tasklist sorted by 
    highest financial revenue loss risk.
    """
    df_task = df_annotated.copy()
    
    if "clv_revenue_at_risk" not in df_task.columns:
        if "monthly_charges" in df_task.columns and "predicted_churn_prob" in df_task.columns:
            df_task["clv_revenue_at_risk"] = np.round(df_task["monthly_charges"] * 24 * df_task["predicted_churn_prob"], 2)
            
    df_task = df_task.sort_values(by="clv_revenue_at_risk", ascending=False)
    
    # Retention Playbook assignment
    playbooks = []
    priorities = []
    
    for _, row in df_task.iterrows():
        prob = row.get("churn_probability", row.get("predicted_churn_prob", 0.0))
        contract = row.get("contract", "")
        tickets = row.get("num_support_tickets", 0)
        
        if prob >= 0.60:
            priorities.append("P1 - Urgent Intervention")
        elif prob >= 0.30:
            priorities.append("P2 - High Priority")
        else:
            priorities.append("P3 - Low Risk")
            
        playbook_actions = []
        if contract == "Month-to-month":
            playbook_actions.append("Offer 15% 1-Year Contract Discount")
        if tickets >= 3:
            playbook_actions.append("Assign Dedicated CS Agent")
        if row.get("tech_support", "") == "No":
            playbook_actions.append("Grant 3 Mo Free Tech Support")
            
        if not playbook_actions:
            playbook_actions.append("Standard Loyalty Campaign")
            
        playbooks.append(" | ".join(playbook_actions))
        
    df_task["intervention_priority"] = priorities
    df_task["recommended_playbook"] = playbooks
    
    output_cols = [
        col for col in ["customer_id", "intervention_priority", "churn_probability", "predicted_churn_prob",
                        "risk_tier", "clv_revenue_at_risk", "recommended_playbook", "contract", "monthly_charges"]
        if col in df_task.columns
    ]
    
    return df_task[output_cols]
