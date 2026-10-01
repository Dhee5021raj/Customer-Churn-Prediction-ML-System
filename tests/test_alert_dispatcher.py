"""
tests/test_alert_dispatcher.py
------------------------------
Unit tests for src/alert_dispatcher.py
"""

import os
import unittest
import pandas as pd

from src.alert_dispatcher import (
    evaluate_alert_rules,
    format_webhook_payload,
    dispatch_alert_event,
)


class TestAlertDispatcher(unittest.TestCase):

    def test_evaluate_vip_churn_risk(self):
        df = pd.DataFrame({
            "churn_probability": [0.85, 0.20, 0.75],
            "clv": [3200.0, 500.0, 2800.0],
        })
        alerts = evaluate_alert_rules(df=df)
        vip_alerts = [a for a in alerts if a["rule_id"] == "VIP_CHURN_RISK"]
        self.assertEqual(len(vip_alerts), 1)
        self.assertEqual(vip_alerts[0]["severity"], "CRITICAL")
        self.assertEqual(vip_alerts[0]["affected_entities"], 2)
        self.assertAlmostEqual(vip_alerts[0]["financial_exposure"], 6000.0)

    def test_evaluate_portfolio_churn_surge(self):
        df = pd.DataFrame({
            "churn_probability": [0.70, 0.80, 0.60, 0.10],
            "clv": [1000.0, 1000.0, 1000.0, 1000.0],
        })
        # 3 out of 4 >= 0.50 => 75% churn rate (above 30% threshold)
        alerts = evaluate_alert_rules(df=df)
        surge_alerts = [a for a in alerts if a["rule_id"] == "PORTFOLIO_CHURN_SURGE"]
        self.assertEqual(len(surge_alerts), 1)
        self.assertEqual(surge_alerts[0]["severity"], "WARNING")

    def test_evaluate_data_drift_alert(self):
        drift_rep = {"drifted_features_count": 3}
        alerts = evaluate_alert_rules(drift_report=drift_rep)
        drift_alerts = [a for a in alerts if a["rule_id"] == "DATA_DRIFT_ALERT"]
        self.assertEqual(len(drift_alerts), 1)

    def test_evaluate_calibration_alert(self):
        cal = {"ece": 0.15, "quality": "Poorly Calibrated"}
        alerts = evaluate_alert_rules(calibration_metrics=cal)
        cal_alerts = [a for a in alerts if a["rule_id"] == "CALIBRATION_DEGRADATION"]
        self.assertEqual(len(cal_alerts), 1)

    def test_format_webhook_payload_slack(self):
        alert = {"title": "Test Title", "description": "Test Desc", "severity": "CRITICAL", "rule_id": "TEST"}
        payload = format_webhook_payload(alert, channel="slack")
        self.assertIn("text", payload)
        self.assertIn("attachments", payload)

    def test_format_webhook_payload_teams(self):
        alert = {"title": "Test Title", "description": "Test Desc", "severity": "WARNING", "rule_id": "TEST"}
        payload = format_webhook_payload(alert, channel="teams")
        self.assertEqual(payload["@type"], "MessageCard")
        self.assertIn("sections", payload)

    def test_dispatch_alert_event(self):
        test_log = "logs/test_alerts.json"
        alert = {"title": "Test Alert", "severity": "INFO"}
        try:
            res = dispatch_alert_event(alert, log_file=test_log)
            self.assertTrue(os.path.exists(test_log))
            self.assertEqual(res["title"], "Test Alert")
        finally:
            if os.path.exists(test_log):
                os.remove(test_log)


if __name__ == "__main__":
    unittest.main()
