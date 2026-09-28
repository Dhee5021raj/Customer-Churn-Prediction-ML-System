"""
tests/test_retention_roi.py
---------------------------
Unit tests for src/retention_roi.py
"""

import unittest
import pandas as pd
import numpy as np

from src.retention_roi import (
    calculate_campaign_roi,
    simulate_portfolio_retention_roi,
    get_budget_allocation_recommendation,
)


class TestRetentionROI(unittest.TestCase):

    def test_calculate_campaign_roi_positive_net(self):
        # 100 targeted, $1000 CLV, $5 contact, $20 incentive, 30% saved
        # saved = 30
        # outreach = 500, incentive = 30 * 20 = 600, total_cost = 1100
        # revenue_saved = 30 * 1000 = 30000
        # net_benefit = 30000 - 1100 = 28900
        res = calculate_campaign_roi(
            total_customers_targeted=100,
            avg_clv=1000.0,
            cost_per_contact=5.0,
            offer_incentive_cost=20.0,
            success_rate=0.30,
        )
        self.assertEqual(res["customers_saved"], 30)
        self.assertEqual(res["total_campaign_cost"], 1100.0)
        self.assertEqual(res["gross_revenue_saved"], 30000.0)
        self.assertEqual(res["net_financial_benefit"], 28900.0)
        self.assertGreater(res["roi_percentage"], 100.0)
        self.assertGreater(res["payback_ratio"], 1.0)
        self.assertLessEqual(res["break_even_customers"], 2)

    def test_calculate_campaign_roi_zero_customers(self):
        res = calculate_campaign_roi(
            total_customers_targeted=0,
            avg_clv=1000.0,
        )
        self.assertEqual(res["total_campaign_cost"], 0.0)
        self.assertEqual(res["net_financial_benefit"], 0.0)
        self.assertEqual(res["roi_percentage"], 0.0)

    def test_simulate_portfolio_retention_roi(self):
        df = pd.DataFrame({
            "risk_tier": ["High Risk"] * 50 + ["Medium Risk"] * 30 + ["Low Risk"] * 20,
            "clv": np.random.uniform(800, 2500, 100),
        })
        sim = simulate_portfolio_retention_roi(df)
        self.assertIn("overall", sim)
        self.assertIn("tier_breakdown", sim)
        self.assertEqual(len(sim["tier_breakdown"]), 3)
        self.assertIn("risk_tier", sim["tier_breakdown"].columns)
        self.assertIn("net_financial_benefit", sim["tier_breakdown"].columns)

    def test_simulate_portfolio_fallback(self):
        # Missing columns fallback
        df = pd.DataFrame({"some_metric": [1, 2, 3]})
        sim = simulate_portfolio_retention_roi(df)
        self.assertIn("overall", sim)
        self.assertEqual(sim["overall"]["total_customers_targeted"], 3)

    def test_get_budget_allocation_recommendation(self):
        df = pd.DataFrame({
            "risk_tier": ["High Risk", "Medium Risk", "Low Risk"],
            "clv": [2000, 1200, 600],
        })
        budget = 10000.0
        alloc = get_budget_allocation_recommendation(budget, df)
        self.assertEqual(len(alloc), 3)
        self.assertAlmostEqual(alloc["allocated_budget"].sum(), budget, places=1)
        self.assertIn("estimated_targetable_customers", alloc.columns)


if __name__ == "__main__":
    unittest.main()
