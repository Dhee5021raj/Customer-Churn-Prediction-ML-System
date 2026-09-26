"""
tests/test_fairness_checker.py
------------------------------
Unit tests for src/fairness_checker.py
"""

import unittest
import numpy as np
import pandas as pd

from src.fairness_checker import (
    calculate_fairness_metrics,
    calculate_demographic_parity,
    calculate_equal_opportunity,
    run_fairness_report,
)


def _make_fair_data(n=200):
    np.random.seed(7)
    df = pd.DataFrame({
        "gender": np.random.choice(["Male", "Female"], n),
        "senior_citizen": np.random.choice([0, 1], n),
    })
    y_true = np.random.randint(0, 2, n)
    y_pred = (np.random.rand(n) > 0.5).astype(int)
    return df, y_true, y_pred


class TestFairnessChecker(unittest.TestCase):

    def test_calculate_fairness_metrics_structure(self):
        df, y_true, y_pred = _make_fair_data()
        result = calculate_fairness_metrics(y_true, y_pred, "gender", df)
        self.assertIn("sensitive_col", result)
        self.assertIn("groups", result)
        self.assertIn("demographic_parity", result)
        self.assertIn("equal_opportunity", result)
        self.assertIn("Male", result["groups"])
        self.assertIn("Female", result["groups"])

    def test_group_metrics_keys(self):
        df, y_true, y_pred = _make_fair_data()
        result = calculate_fairness_metrics(y_true, y_pred, "gender", df)
        for grp_metrics in result["groups"].values():
            self.assertIn("true_positive_rate", grp_metrics)
            self.assertIn("selection_rate", grp_metrics)
            self.assertIn("count", grp_metrics)

    def test_demographic_parity_perfect_equality(self):
        # Equal selection rates → ratio should be 1.0
        rates = {"A": 0.5, "B": 0.5}
        dp = calculate_demographic_parity(rates)
        self.assertAlmostEqual(dp["ratio"], 1.0)
        self.assertTrue(dp["passes_4_5ths_rule"])

    def test_demographic_parity_disparity(self):
        # min=0.3, max=0.9 → ratio ≈ 0.333 → fails 4/5ths rule
        rates = {"A": 0.3, "B": 0.9}
        dp = calculate_demographic_parity(rates)
        self.assertFalse(dp["passes_4_5ths_rule"])

    def test_equal_opportunity_pass(self):
        rates = {"A": 0.82, "B": 0.90}
        eo = calculate_equal_opportunity(rates)
        self.assertTrue(eo["passes_equal_opportunity"])

    def test_run_fairness_report_returns_all_cols(self):
        df, y_true, y_pred = _make_fair_data()
        report = run_fairness_report(y_true, y_pred, df, ["gender", "senior_citizen"])
        self.assertIn("gender", report)
        self.assertIn("senior_citizen", report)

    def test_missing_column_raises(self):
        df, y_true, y_pred = _make_fair_data()
        with self.assertRaises(ValueError):
            calculate_fairness_metrics(y_true, y_pred, "nonexistent_col", df)


if __name__ == "__main__":
    unittest.main()
