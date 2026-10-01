"""
tests/test_uplift_modeler.py
----------------------------
Unit tests for src/uplift_modeler.py
"""

import unittest
import pandas as pd
import numpy as np

from src.uplift_modeler import (
    calculate_customer_uplift_score,
    classify_uplift_quadrant,
    segment_portfolio_by_uplift,
    calculate_uplift_efficiency_metrics,
)


class TestUpliftModeler(unittest.TestCase):

    def test_calculate_customer_uplift_score(self):
        tau = calculate_customer_uplift_score(base_prob=0.70, treated_prob=0.45)
        self.assertAlmostEqual(tau, 0.25, places=4)

    def test_classify_persuadable(self):
        res = classify_uplift_quadrant(base_prob=0.65, uplift_score=0.20)
        self.assertEqual(res["quadrant"], "Persuadables")
        self.assertIn("P1", res["targeting_priority"])
        self.assertEqual(res["color"], "green")

    def test_classify_sure_thing(self):
        res = classify_uplift_quadrant(base_prob=0.20, uplift_score=0.05)
        self.assertEqual(res["quadrant"], "Sure Things")
        self.assertIn("Standard", res["targeting_priority"])

    def test_classify_lost_cause(self):
        res = classify_uplift_quadrant(base_prob=0.80, uplift_score=0.02)
        self.assertEqual(res["quadrant"], "Lost Causes")

    def test_classify_sleeping_dog(self):
        res = classify_uplift_quadrant(base_prob=0.30, uplift_score=-0.10)
        self.assertEqual(res["quadrant"], "Sleeping Dogs")
        self.assertIn("Do Not Disturb", res["targeting_priority"])
        self.assertEqual(res["color"], "red")

    def test_segment_portfolio_by_uplift(self):
        df = pd.DataFrame({
            "churn_probability": [0.75, 0.20, 0.85, 0.30],
            "tenure": [12, 24, 6, 36],
        })
        res = segment_portfolio_by_uplift(df)
        self.assertIn("uplift_score", res.columns)
        self.assertIn("uplift_quadrant", res.columns)
        self.assertIn("targeting_priority", res.columns)
        self.assertEqual(len(res), 4)

    def test_calculate_uplift_efficiency_metrics(self):
        df = pd.DataFrame({
            "uplift_quadrant": ["Persuadables", "Sure Things", "Lost Causes", "Persuadables"],
            "uplift_score": [0.25, 0.05, 0.02, 0.22],
        })
        eff = calculate_uplift_efficiency_metrics(df)
        self.assertEqual(eff["total_customers"], 4)
        self.assertEqual(eff["quadrant_counts"]["Persuadables"], 2)
        self.assertEqual(eff["persuadable_percentage"], 50.0)
        self.assertGreater(eff["targeting_efficiency_multiplier"], 1.0)
        self.assertEqual(eff["budget_waste_prevented_pct"], 50.0)


if __name__ == "__main__":
    unittest.main()
