"""
tests/test_risk_classifier.py
------------------------------
Unit tests for src/risk_classifier.py
"""

import unittest
import pandas as pd

from src.risk_classifier import classify_risk_band, annotate_dataframe_with_risk_bands


class TestRiskClassifier(unittest.TestCase):

    # ── classify_risk_band ─────────────────────────────────────────────────

    def test_safe_band(self):
        result = classify_risk_band(0.10)
        self.assertEqual(result["band"], "Safe")
        self.assertEqual(result["confidence"], "High")

    def test_low_band(self):
        result = classify_risk_band(0.30)
        self.assertEqual(result["band"], "Low")

    def test_moderate_band(self):
        result = classify_risk_band(0.50)
        self.assertEqual(result["band"], "Moderate")
        self.assertEqual(result["confidence"], "High")

    def test_high_band(self):
        result = classify_risk_band(0.70)
        self.assertEqual(result["band"], "High")

    def test_critical_band(self):
        result = classify_risk_band(0.90)
        self.assertEqual(result["band"], "Critical")
        self.assertEqual(result["confidence"], "High")

    def test_boundary_low_confidence(self):
        # Right at a band boundary → Low confidence
        result = classify_risk_band(0.20)
        self.assertEqual(result["confidence"], "Low")

    def test_exact_zero(self):
        result = classify_risk_band(0.0)
        self.assertEqual(result["band"], "Safe")

    def test_exact_one(self):
        result = classify_risk_band(1.0)
        self.assertEqual(result["band"], "Critical")

    def test_clamp_above_one(self):
        result = classify_risk_band(1.5)
        self.assertEqual(result["band"], "Critical")

    def test_clamp_below_zero(self):
        result = classify_risk_band(-0.1)
        self.assertEqual(result["band"], "Safe")

    def test_result_keys(self):
        result = classify_risk_band(0.55)
        for key in ["band", "label", "color", "confidence", "probability"]:
            self.assertIn(key, result)

    # ── annotate_dataframe_with_risk_bands ────────────────────────────────

    def test_annotate_adds_columns(self):
        df = pd.DataFrame({"churn_probability": [0.1, 0.35, 0.55, 0.75, 0.92]})
        result = annotate_dataframe_with_risk_bands(df)
        for col in ["risk_band", "risk_label", "risk_color", "confidence"]:
            self.assertIn(col, result.columns)

    def test_annotate_correct_bands(self):
        df = pd.DataFrame({"churn_probability": [0.10, 0.85]})
        result = annotate_dataframe_with_risk_bands(df)
        self.assertEqual(result["risk_band"].iloc[0], "Safe")
        self.assertEqual(result["risk_band"].iloc[1], "Critical")

    def test_annotate_raises_on_missing_col(self):
        df = pd.DataFrame({"score": [0.5]})
        with self.assertRaises(ValueError):
            annotate_dataframe_with_risk_bands(df)

    def test_annotate_does_not_mutate_original(self):
        df = pd.DataFrame({"churn_probability": [0.4, 0.6]})
        original_cols = list(df.columns)
        annotate_dataframe_with_risk_bands(df)
        self.assertEqual(list(df.columns), original_cols)


if __name__ == "__main__":
    unittest.main()
