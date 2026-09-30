"""
tests/test_root_cause_analyzer.py
---------------------------------
Unit tests for src/root_cause_analyzer.py
"""

import unittest
import pandas as pd

from src.root_cause_analyzer import (
    diagnose_customer_root_causes,
    diagnose_portfolio_root_causes,
    get_department_action_playbook,
)


class TestRootCauseAnalyzer(unittest.TestCase):

    def test_diagnose_support_dissatisfaction(self):
        profile = {
            "monthly_charges": 40.0,
            "num_support_tickets": 5,
            "tech_support": "No",
            "tenure": 24,
            "contract": "Two year",
        }
        res = diagnose_customer_root_causes(profile)
        self.assertEqual(res["primary_root_cause"], "Support Experience Dissatisfaction")
        self.assertEqual(res["severity"], "High")
        self.assertIn("department", res["playbook"])
        self.assertEqual(res["playbook"]["department"], "Customer Support Operations")

    def test_diagnose_pricing_friction(self):
        profile = {
            "monthly_charges": 110.0,
            "payment_method": "Electronic check",
            "num_support_tickets": 0,
            "tenure": 36,
            "contract": "One year",
        }
        res = diagnose_customer_root_causes(profile)
        self.assertEqual(res["primary_root_cause"], "Pricing & Billing Friction")
        self.assertIn("Billing", res["playbook"]["department"])

    def test_diagnose_onboarding_risk(self):
        profile = {
            "monthly_charges": 45.0,
            "num_support_tickets": 0,
            "tenure": 2,
            "contract": "Month-to-month",
        }
        res = diagnose_customer_root_causes(profile)
        self.assertEqual(res["primary_root_cause"], "Commitment & Onboarding Risk")
        self.assertEqual(res["playbook"]["department"], "Customer Success & Onboarding")

    def test_diagnose_portfolio_root_causes(self):
        df = pd.DataFrame([
            {"customer_id": "C1", "monthly_charges": 110.0, "num_support_tickets": 0, "tenure": 30, "contract": "Two year", "clv": 2500},
            {"customer_id": "C2", "monthly_charges": 40.0, "num_support_tickets": 4, "tenure": 20, "contract": "One year", "clv": 1200},
            {"customer_id": "C3", "monthly_charges": 50.0, "num_support_tickets": 0, "tenure": 2, "contract": "Month-to-month", "clv": 800},
        ])
        summary = diagnose_portfolio_root_causes(df)
        self.assertEqual(len(summary), 3)
        self.assertIn("root_cause", summary.columns)
        self.assertIn("affected_customers", summary.columns)
        self.assertIn("share_percentage", summary.columns)
        self.assertIn("department", summary.columns)

    def test_get_department_action_playbook_fallback(self):
        pb = get_department_action_playbook("Unknown Random Cause")
        self.assertIn("department", pb)
        self.assertIn("action", pb)


if __name__ == "__main__":
    unittest.main()
