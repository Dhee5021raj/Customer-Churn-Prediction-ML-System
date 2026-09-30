"""
tests/test_executive_reporter.py
--------------------------------
Unit tests for src/executive_reporter.py
"""

import os
import unittest
import pandas as pd

from src.executive_reporter import (
    generate_executive_html_report,
    save_executive_report,
)


class TestExecutiveReporter(unittest.TestCase):

    def test_generate_executive_html_report_content(self):
        kpis = {
            "total_customers": 1000,
            "churn_rate_pct": 26.5,
            "clv_at_risk": 450000.0,
            "champion_auc": 0.852,
        }
        risk_summary = {"High Risk": 265, "Medium Risk": 310, "Low Risk": 425}
        roi_summary = {
            "total_campaign_cost": 25000.0,
            "gross_revenue_saved": 85000.0,
            "net_financial_benefit": 60000.0,
            "roi_percentage": 240.0,
        }
        root_causes = pd.DataFrame([
            {"root_cause": "Pricing Friction", "affected_customers": 150, "share_percentage": 30.0, "avg_customer_clv": 1400.0, "department": "Billing", "urgency": "High"},
        ])

        html = generate_executive_html_report(kpis, risk_summary, roi_summary, root_causes)
        self.assertIn("Customer Churn Intelligence Report", html)
        self.assertIn("1,000", html)
        self.assertIn("26.5%", html)
        self.assertIn("$450,000", html)
        self.assertIn("High Risk", html)
        self.assertIn("Pricing Friction", html)
        self.assertIn("240.0%", html)
        self.assertIn("<!DOCTYPE html>", html)

    def test_save_executive_report(self):
        html = "<html><body>Test Executive Report</body></html>"
        test_path = "reports/test_report.html"
        try:
            saved_path = save_executive_report(html, test_path)
            self.assertTrue(os.path.exists(saved_path))
            with open(saved_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertEqual(content, html)
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)


if __name__ == "__main__":
    unittest.main()
