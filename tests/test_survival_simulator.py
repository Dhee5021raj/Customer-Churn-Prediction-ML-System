"""
tests/test_survival_simulator.py
--------------------------------
Unit tests for src/survival_simulator.py
"""

import unittest
from src.survival_simulator import (
    simulate_customer_survival_curve,
    calculate_expected_customer_lifetime,
    compare_contract_survival_curves,
)


class TestSurvivalSimulator(unittest.TestCase):

    def test_survival_curve_monotonic_decreasing(self):
        curve = simulate_customer_survival_curve(0.35, tenure_months=6, contract_type="Month-to-month", periods=12)
        self.assertEqual(len(curve), 12)
        surv_values = [pt["survival_probability"] for pt in curve]
        for i in range(len(surv_values) - 1):
            self.assertGreaterEqual(surv_values[i], surv_values[i + 1])

    def test_contract_type_impact(self):
        # Two year contract should have strictly higher 6-month survival than Month-to-month
        m2m = simulate_customer_survival_curve(0.40, tenure_months=6, contract_type="Month-to-month", periods=12)
        two_yr = simulate_customer_survival_curve(0.40, tenure_months=6, contract_type="Two year", periods=12)
        self.assertGreater(two_yr[5]["survival_probability"], m2m[5]["survival_probability"])

    def test_calculate_expected_customer_lifetime_high_risk(self):
        # High churn probability should reach median <= 24 months
        curve = simulate_customer_survival_curve(0.85, tenure_months=1, contract_type="Month-to-month", periods=24)
        stats = calculate_expected_customer_lifetime(curve)
        self.assertIsInstance(stats["median_survival_months"], int)
        self.assertLessEqual(stats["median_survival_months"], 12)
        self.assertIn("survival_at_12m", stats)
        self.assertIn("survival_at_24m", stats)

    def test_calculate_expected_customer_lifetime_low_risk(self):
        # Very low churn probability should exceed horizon (>24)
        curve = simulate_customer_survival_curve(0.05, tenure_months=24, contract_type="Two year", periods=24)
        stats = calculate_expected_customer_lifetime(curve)
        self.assertEqual(stats["median_survival_months"], ">24")
        self.assertGreater(stats["survival_at_24m"], 0.70)

    def test_compare_contract_survival_curves_shape(self):
        comp = compare_contract_survival_curves(0.30, tenure_months=12, periods=18)
        self.assertEqual(len(comp), 18)
        self.assertIn("Month-to-month", comp.columns)
        self.assertIn("One year", comp.columns)
        self.assertIn("Two year", comp.columns)

    def test_invalid_periods_raises(self):
        with self.assertRaises(ValueError):
            simulate_customer_survival_curve(0.30, periods=0)


if __name__ == "__main__":
    unittest.main()
