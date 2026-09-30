"""
src/root_cause_analyzer.py
--------------------------
Customer churn root-cause diagnostic and business friction analyzer.

Translates technical SHAP attributions and customer profile metrics into
actionable strategic business friction themes with departmental action playbooks.

Functions:
    diagnose_customer_root_causes(customer_profile, top_shap_features) -> dict
    diagnose_portfolio_root_causes(df) -> pd.DataFrame
    get_department_action_playbook(root_cause) -> dict
"""

import pandas as pd
from typing import Dict, Any, List, Optional

from src.config import setup_logger

logger = setup_logger("root_cause_analyzer")

# Departmental playbooks mapped to root-cause categories
DEPARTMENT_PLAYBOOKS = {
    "Pricing & Billing Friction": {
        "department": "Billing & Revenue Strategy",
        "action": "Offer annualized billing discount or transition customer from Electronic check to automated autopay incentive.",
        "urgency": "High",
    },
    "Support Experience Dissatisfaction": {
        "department": "Customer Support Operations",
        "action": "Trigger high-priority concierge support outreach; assign dedicated technical account manager.",
        "urgency": "Immediate",
    },
    "Commitment & Onboarding Risk": {
        "department": "Customer Success & Onboarding",
        "action": "Enroll in 90-day onboarding engagement sprint and offer multi-month loyalty extension bundle.",
        "urgency": "High",
    },
    "Addon Protection Deficiency": {
        "department": "Product & Addon Marketing",
        "action": "Provide 30-day complimentary trial for Online Security and Tech Support suite.",
        "urgency": "Medium",
    },
    "Service Under-utilization": {
        "department": "Product Adoption & Engagement",
        "action": "Send tailored product feature walkthroughs, usage milestone notifications, and data tier adjustments.",
        "urgency": "Medium",
    },
}


def get_department_action_playbook(root_cause: str) -> Dict[str, str]:
    """Retrieves action playbook details for a given root cause category."""
    return DEPARTMENT_PLAYBOOKS.get(
        root_cause,
        {
            "department": "Customer Retention Desk",
            "action": "Schedule exploratory check-in call with customer.",
            "urgency": "Medium",
        },
    )


def diagnose_customer_root_causes(
    customer_profile: Dict[str, Any],
    top_shap_features: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Diagnoses customer churn friction drivers into business categories.

    Parameters
    ----------
    customer_profile : dict
        Customer feature dictionary.
    top_shap_features : list of dict, optional
        SHAP features from ChurnExplainer.

    Returns
    -------
    dict
        Structured diagnosis containing primary_root_cause, friction_scores,
        playbook, and severity.
    """
    scores = {
        "Pricing & Billing Friction": 10.0,
        "Support Experience Dissatisfaction": 10.0,
        "Commitment & Onboarding Risk": 10.0,
        "Addon Protection Deficiency": 10.0,
        "Service Under-utilization": 10.0,
    }

    # Profile feature checks
    charges = float(customer_profile.get("monthly_charges", 50))
    if charges > 80:
        scores["Pricing & Billing Friction"] += 35
    elif charges > 65:
        scores["Pricing & Billing Friction"] += 20

    pay_method = str(customer_profile.get("payment_method", ""))
    if "electronic check" in pay_method.lower():
        scores["Pricing & Billing Friction"] += 20

    tickets = int(customer_profile.get("num_support_tickets", 0))
    if tickets >= 3:
        scores["Support Experience Dissatisfaction"] += 50
    elif tickets >= 1:
        scores["Support Experience Dissatisfaction"] += 25

    tech = str(customer_profile.get("tech_support", "")).lower()
    if tech == "no":
        scores["Support Experience Dissatisfaction"] += 15

    contract = str(customer_profile.get("contract", "")).lower()
    if "month-to-month" in contract:
        scores["Commitment & Onboarding Risk"] += 35

    tenure = float(customer_profile.get("tenure", customer_profile.get("tenure_months", 12)))
    if tenure <= 6:
        scores["Commitment & Onboarding Risk"] += 35
    elif tenure <= 12:
        scores["Commitment & Onboarding Risk"] += 20

    sec = str(customer_profile.get("online_security", "")).lower()
    if sec == "no":
        scores["Addon Protection Deficiency"] += 25
    backup = str(customer_profile.get("online_backup", "")).lower()
    if backup == "no":
        scores["Addon Protection Deficiency"] += 20

    usage = float(customer_profile.get("monthly_usage_gb", 50))
    if usage < 25:
        scores["Service Under-utilization"] += 30

    # Boost scores based on SHAP impact if available
    if top_shap_features:
        for f in top_shap_features:
            fname = str(f.get("feature", "")).lower()
            val = float(f.get("shap_value", 0.0))
            if val > 0:
                if any(k in fname for k in ["charge", "pay", "billing"]):
                    scores["Pricing & Billing Friction"] += 20
                elif any(k in fname for k in ["ticket", "support"]):
                    scores["Support Experience Dissatisfaction"] += 20
                elif any(k in fname for k in ["tenure", "contract"]):
                    scores["Commitment & Onboarding Risk"] += 20
                elif any(k in fname for k in ["security", "backup", "protection"]):
                    scores["Addon Protection Deficiency"] += 20
                elif any(k in fname for k in ["usage", "product", "streaming"]):
                    scores["Service Under-utilization"] += 20

    # Normalize scores to 0-100 max range
    max_raw = max(scores.values())
    normalized_scores = {k: round(min(100.0, v), 1) for k, v in scores.items()}

    primary_cause = max(normalized_scores, key=normalized_scores.get)
    highest_score = normalized_scores[primary_cause]

    severity = "High" if highest_score >= 60 else ("Moderate" if highest_score >= 40 else "Low")
    playbook = get_department_action_playbook(primary_cause)

    logger.info(f"Root cause diagnosed: {primary_cause} (score={highest_score}, severity={severity})")

    return {
        "primary_root_cause": primary_cause,
        "highest_friction_score": highest_score,
        "severity": severity,
        "friction_scores": normalized_scores,
        "playbook": playbook,
    }


def diagnose_portfolio_root_causes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Diagnoses and summarizes root causes across an entire customer dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Customer DataFrame with feature columns and optional 'clv'.

    Returns
    -------
    pd.DataFrame
        Ranked summary of root causes with customer count, share %, and average CLV.
    """
    diagnoses = []
    for _, row in df.iterrows():
        prof = row.to_dict()
        diag = diagnose_customer_root_causes(prof)
        diag["customer_id"] = prof.get("customer_id", "")
        diag["clv"] = prof.get("clv", 1200.0)
        diagnoses.append(diag)

    d_df = pd.DataFrame(diagnoses)
    
    summary_records = []
    total = len(d_df)

    for cause, group in d_df.groupby("primary_root_cause"):
        count = len(group)
        share = round((count / total) * 100, 1) if total > 0 else 0.0
        avg_clv = round(float(group["clv"].mean()), 2)
        playbook = get_department_action_playbook(cause)

        summary_records.append({
            "root_cause": cause,
            "affected_customers": count,
            "share_percentage": share,
            "avg_customer_clv": avg_clv,
            "department": playbook["department"],
            "urgency": playbook["urgency"],
        })

    summary_df = pd.DataFrame(summary_records).sort_values("affected_customers", ascending=False).reset_index(drop=True)
    return summary_df
