import pandas as pd
from typing import List, Dict, Any


def generate_retention_recommendations(
    customer_profile: Dict[str, Any],
    top_features: List[Dict[str, Any]] = None
) -> List[str]:
    """
    Generate tailored, actionable customer retention recommendations based on 
    the customer's demographic profile, service usage, and SHAP top risk drivers.
    """
    recs = []
    
    contract = customer_profile.get("contract", "")
    tech_support = customer_profile.get("tech_support", "")
    num_tickets = customer_profile.get("num_support_tickets", 0)
    payment_method = customer_profile.get("payment_method", "")
    monthly_charges = customer_profile.get("monthly_charges", 0.0)
    online_security = customer_profile.get("online_security", "")

    if contract == "Month-to-month":
        recs.append("• **Offer Long-Term Contract Discount**: Present a 15% discount incentive for switching to a 1-Year or 2-Year contract.")
    
    if tech_support == "No" or (top_features and any("tech_support" in f.get("feature", "") and f.get("shap_value", 0) > 0 for f in top_features)):
        recs.append("• **Complimentary Tech Support Upgrade**: Grant 3 months of free premium tech support to improve service satisfaction.")
        
    if num_tickets >= 3:
        recs.append("• **Proactive Customer Success Intervention**: High support ticket volume detected — assign a dedicated representative to resolve open tickets.")
        
    if payment_method == "Electronic check":
        recs.append("• **Auto-Pay Enrollment Incentive**: Offer a $5 monthly bill credit for switching from Electronic Check to automated Credit Card or Bank Transfer.")

    if online_security == "No":
        recs.append("• **Security Add-on Trial**: Offer 60 days of free Online Security protection to increase account lock-in.")

    if monthly_charges > 85 and contract == "Month-to-month":
        recs.append("• **Custom Value Bundle**: High monthly spend detected — propose a customized plan bundle to reduce immediate churn risk.")

    if not recs:
        recs.append("• **Standard Engagement**: Customer exhibits low attrition risk. Maintain standard loyalty communications and promotional offers.")
        
    return recs
