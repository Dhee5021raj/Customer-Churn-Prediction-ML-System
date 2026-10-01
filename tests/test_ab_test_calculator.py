"""
tests/test_ab_test_calculator.py
--------------------------------
Unit tests for src/ab_test_calculator.py
"""

import unittest
from src.ab_test_calculator import (
    calculate_sample_size_for_retention_test,
    evaluate_ab_test_results,
)


class TestABTestCalculator(unittest.TestCase):

    def test_calculate_sample_size_reasonable_output(self):
        # Baseline 25% churn, 20% relative reduction (down to 20%), alpha=0.05, power=0.80
        res = calculate_sample_size_for_retention_test(
            baseline_churn_rate=0.25,
            expected_reduction_pct=0.20,
            alpha=0.05,
            power=0.80,
        )
        self.assertGreater(res["sample_size_per_variant"], 500)
        self.assertEqual(res["total_sample_size"], res["sample_size_per_variant"] * 2)
        self.assertAlmostEqual(res["baseline_churn_rate"], 0.25)
        self.assertAlmostEqual(res["target_churn_rate"], 0.20)

    def test_evaluate_ab_test_significant_winner(self):
        # Control: 1000 users, 250 churns (25%)
        # Variant: 1000 users, 180 churns (18%)
        res = evaluate_ab_test_results(
            control_size=1000,
            control_churns=250,
            variant_size=1000,
            variant_churns=180,
            alpha=0.05,
        )
        self.assertTrue(res["is_statistically_significant"])
        self.assertGreater(res["z_statistic"], 1.96)
        self.assertLess(res["p_value"], 0.05)
        self.assertIn("Significant Winner", res["recommendation"])

    def test_evaluate_ab_test_inconclusive(self):
        # Control: 50 users, 10 churns (20%)
        # Variant: 50 users, 9 churns (18%) -> small sample size, not significant
        res = evaluate_ab_test_results(
            control_size=50,
            control_churns=10,
            variant_size=50,
            variant_churns=9,
            alpha=0.05,
        )
        self.assertFalse(res["is_statistically_significant"])
        self.assertGreater(res["p_value"], 0.05)
        self.assertIn("Inconclusive", res["recommendation"])

    def test_evaluate_ab_test_negative_effect(self):
        # Variant churns significantly MORE than control (backfired)
        res = evaluate_ab_test_results(
            control_size=1000,
            control_churns=150,
            variant_size=1000,
            variant_churns=250,
            alpha=0.05,
        )
        self.assertFalse(res["is_statistically_significant"])
        self.assertLess(res["z_statistic"], 0)
        self.assertIn("Negative Effect", res["recommendation"])

    def test_zero_size_raises(self):
        with self.assertRaises(ValueError):
            evaluate_ab_test_results(0, 0, 100, 10)


if __name__ == "__main__":
    unittest.main()
