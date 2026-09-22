import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clv_calculator import calculate_customer_clv, calculate_clv_risk, get_clv_risk_summary
from data.generate_data import generate_customer_churn_dataset


class TestCLVCalculator(unittest.TestCase):

    def test_calculate_customer_clv(self):
        clv = calculate_customer_clv(monthly_charges=100.0, tenure_months=24)
        self.assertEqual(clv, 2400.0)

    def test_calculate_clv_risk(self):
        df = generate_customer_churn_dataset(num_samples=20, random_state=42)
        probs = np.random.uniform(0.0, 1.0, size=len(df))
        
        annotated_df = calculate_clv_risk(df, probs, clv_horizon_months=24)
        self.assertIn("estimated_clv", annotated_df.columns)
        self.assertIn("churn_probability", annotated_df.columns)
        self.assertIn("clv_revenue_at_risk", annotated_df.columns)
        self.assertIn("risk_tier", annotated_df.columns)

    def test_get_clv_risk_summary(self):
        df = generate_customer_churn_dataset(num_samples=20, random_state=42)
        probs = np.array([0.9 if i % 2 == 0 else 0.1 for i in range(len(df))])
        
        annotated_df = calculate_clv_risk(df, probs)
        summary = get_clv_risk_summary(annotated_df)
        
        self.assertIn("total_portfolio_clv", summary)
        self.assertIn("total_revenue_at_risk", summary)
        self.assertIn("high_risk_count", summary)
        self.assertGreater(summary["total_portfolio_clv"], 0)


if __name__ == "__main__":
    unittest.main()
