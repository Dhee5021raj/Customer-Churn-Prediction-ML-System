import unittest
import os
import sys
import pandas as pd
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.explainability import ChurnExplainer
from src.what_if_analyzer import simulate_what_if_scenario


class TestWhatIfAnalyzer(unittest.TestCase):

    def test_simulate_what_if_scenario(self):
        self.assertTrue(os.path.exists("models/xgboost_model.pkl"))
        explainer = ChurnExplainer()
        
        base_profile = {
            "gender": "Female",
            "senior_citizen": 0,
            "partner": "No",
            "dependents": "No",
            "tenure": 6,
            "phone_service": "Yes",
            "multiple_lines": "No",
            "internet_service": "Fiber optic",
            "online_security": "No",
            "online_backup": "No",
            "device_protection": "No",
            "tech_support": "No",
            "streaming_tv": "No",
            "streaming_movies": "No",
            "contract": "Month-to-month",
            "paperless_billing": "Yes",
            "payment_method": "Electronic check",
            "monthly_charges": 85.0,
            "total_charges": 510.0,
            "num_support_tickets": 3
        }
        
        modifications = {"contract": "One year", "tech_support": "Yes"}
        res = simulate_what_if_scenario(explainer, base_profile, modifications, scenario_name="Test Contract Upgrade")
        
        self.assertIn("base_prob", res)
        self.assertIn("mod_prob", res)
        self.assertIn("risk_delta", res)
        self.assertIn("net_revenue_saved", res)
        self.assertLessEqual(res["mod_prob"], res["base_prob"])


if __name__ == "__main__":
    unittest.main()
