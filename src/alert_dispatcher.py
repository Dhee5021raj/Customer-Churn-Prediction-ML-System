"""
src/alert_dispatcher.py
-----------------------
Automated churn operational alerting and notification webhook dispatcher.

Evaluates operational alerting rules (VIP customer churn exposure, portfolio
churn surge, distribution drift, calibration decay) and formats Slack/Teams webhooks.

Functions:
    evaluate_alert_rules(df, drift_report, calibration_metrics) -> list[dict]
    format_webhook_payload(alert, channel) -> dict
    dispatch_alert_event(alert, log_file) -> dict
"""

import os
import json
from datetime import datetime, timezone
import pandas as pd
from typing import List, Dict, Any, Optional

from src.config import BASE_DIR, setup_logger

logger = setup_logger("alert_dispatcher")

ALERTS_LOG_PATH = os.path.join(BASE_DIR, "logs", "alerts_history.json")


def evaluate_alert_rules(
    df: Optional[pd.DataFrame] = None,
    drift_report: Optional[Dict[str, Any]] = None,
    calibration_metrics: Optional[Dict[str, Any]] = None,
    vip_clv_threshold: float = 2500.0,
    churn_rate_threshold_pct: float = 30.0,
) -> List[Dict[str, Any]]:
    """
    Evaluates system operational alert rules against current data and metrics.

    Returns
    -------
    list of dict
        List of triggered alert dictionaries.
    """
    alerts = []
    now_iso = datetime.now(timezone.utc).isoformat()

    # Rule 1: VIP Customer Churn Exposure
    if df is not None and not df.empty:
        prob_col = "churn_probability" if "churn_probability" in df.columns else None
        clv_col = "clv" if "clv" in df.columns else ("clv_24m" if "clv_24m" in df.columns else None)

        if prob_col and clv_col:
            vip_mask = (df[clv_col] >= vip_clv_threshold) & (df[prob_col] >= 0.65)
            vip_count = int(vip_mask.sum())
            if vip_count > 0:
                vip_clv_total = round(float(df.loc[vip_mask, clv_col].sum()), 2)
                alerts.append({
                    "rule_id": "VIP_CHURN_RISK",
                    "severity": "CRITICAL",
                    "title": f"🚨 {vip_count} High-Value VIP Customers at Risk of Churn",
                    "description": f"{vip_count} customers with CLV >= ${vip_clv_threshold:,.0f} exhibit severe churn risk. Total VIP exposure: ${vip_clv_total:,.2f}.",
                    "timestamp": now_iso,
                    "affected_entities": vip_count,
                    "financial_exposure": vip_clv_total,
                })

        # Rule 2: Overall Portfolio Churn Rate Surge
        if prob_col:
            rate_pct = round(float((df[prob_col] >= 0.50).mean() * 100), 1)
            if rate_pct >= churn_rate_threshold_pct:
                alerts.append({
                    "rule_id": "PORTFOLIO_CHURN_SURGE",
                    "severity": "WARNING",
                    "title": f"⚠️ Portfolio Churn Rate Surge ({rate_pct}%)",
                    "description": f"Portfolio predicted churn rate of {rate_pct}% exceeds operational alert threshold of {churn_rate_threshold_pct}%.",
                    "timestamp": now_iso,
                    "affected_entities": len(df),
                    "churn_rate_pct": rate_pct,
                })

    # Rule 3: Data Distribution Drift Alert
    if drift_report:
        drifted = drift_report.get("drifted_features_count", 0)
        if drifted >= 2:
            alerts.append({
                "rule_id": "DATA_DRIFT_ALERT",
                "severity": "WARNING",
                "title": f"📉 Feature Distribution Drift Detected ({drifted} features)",
                "description": f"{drifted} production features exhibit statistically significant distribution shift compared to training baseline.",
                "timestamp": now_iso,
                "affected_entities": drifted,
            })

    # Rule 4: Model Probability Calibration Degradation
    if calibration_metrics:
        ece = calibration_metrics.get("ece", 0.0)
        quality = calibration_metrics.get("quality", "")
        if ece >= 0.12 or "Poor" in quality:
            alerts.append({
                "rule_id": "CALIBRATION_DEGRADATION",
                "severity": "WARNING",
                "title": f"🎯 Probability Calibration Degraded (ECE: {ece:.4f})",
                "description": f"Model probability calibration has degraded to '{quality}'. Model retraining or Platt scaling recommended.",
                "timestamp": now_iso,
                "ece_score": ece,
            })

    logger.info(f"Evaluated alert rules: {len(alerts)} alerts triggered.")
    return alerts


def format_webhook_payload(
    alert: Dict[str, Any],
    channel: str = "slack",
) -> Dict[str, Any]:
    """
    Formats an alert event dictionary into a Slack or Teams compatible webhook payload.
    """
    severity = alert.get("severity", "INFO")
    color_map = {"CRITICAL": "#EF4444", "WARNING": "#F59E0B", "INFO": "#3B82F6"}
    color = color_map.get(severity, "#64748B")

    if channel.lower() == "teams":
        return {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": color.replace("#", ""),
            "summary": alert.get("title", ""),
            "title": alert.get("title", ""),
            "sections": [{
                "activityTitle": f"Severity: {severity}",
                "text": alert.get("description", ""),
                "facts": [
                    {"name": "Rule ID", "value": alert.get("rule_id", "")},
                    {"name": "Timestamp", "value": alert.get("timestamp", "")},
                ],
            }],
        }

    # Default to Slack Incoming Webhook format
    return {
        "text": f"*{alert.get('title', '')}*",
        "attachments": [{
            "color": color,
            "fields": [
                {"title": "Severity", "value": severity, "short": True},
                {"title": "Rule ID", "value": alert.get("rule_id", ""), "short": True},
                {"title": "Details", "value": alert.get("description", ""), "short": False},
                {"title": "Timestamp", "value": alert.get("timestamp", ""), "short": False},
            ],
        }],
    }


def dispatch_alert_event(
    alert: Dict[str, Any],
    log_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Records an alert event to the alerts history file and logs it.
    """
    if log_file is None:
        log_file = ALERTS_LOG_PATH

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    history = []
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append(alert)
    # Keep last 100 alerts
    history = history[-100:]

    with open(log_file, "w") as f:
        json.dump(history, f, indent=2)

    logger.warning(f"ALERT DISPATCHED [{alert.get('severity')}]: {alert.get('title')}")
    return alert
