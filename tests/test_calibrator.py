"""
tests/test_calibrator.py
------------------------
Unit tests for src/calibrator.py
"""

import unittest
import numpy as np

from src.calibrator import (
    assess_calibration_quality,
    compute_calibration_curve,
    calculate_expected_calibration_error,
)


class TestCalibrator(unittest.TestCase):

    def test_assess_calibration_quality_tiers(self):
        self.assertEqual(assess_calibration_quality(0.03), "Well-Calibrated")
        self.assertEqual(assess_calibration_quality(0.08), "Moderately Calibrated")
        self.assertEqual(assess_calibration_quality(0.18), "Poorly Calibrated")

    def test_compute_calibration_curve(self):
        np.random.seed(42)
        y_prob = np.random.uniform(0.1, 0.9, 100)
        y_true = (np.random.uniform(0, 1, 100) < y_prob).astype(int)

        curve = compute_calibration_curve(y_true, y_prob, n_bins=5)
        self.assertIn("prob_true", curve)
        self.assertIn("prob_pred", curve)
        self.assertGreater(len(curve["prob_true"]), 0)
        self.assertEqual(len(curve["prob_true"]), len(curve["prob_pred"]))

    def test_calculate_expected_calibration_error_perfect(self):
        # When probabilities perfectly match empirical frequencies, ECE is minimal
        y_prob = np.array([0.1]*50 + [0.9]*50)
        y_true = np.array([0]*45 + [1]*5 + [0]*5 + [1]*45)

        metrics = calculate_expected_calibration_error(y_true, y_prob, n_bins=10)
        self.assertIn("ece", metrics)
        self.assertIn("mce", metrics)
        self.assertIn("brier_score", metrics)
        self.assertIn("quality", metrics)
        self.assertLessEqual(metrics["ece"], 0.05)
        self.assertEqual(metrics["quality"], "Well-Calibrated")

    def test_calculate_expected_calibration_error_poor(self):
        # Severely overconfident/miscalibrated: predicts 0.95 for all zeros
        y_prob = np.full(50, 0.95)
        y_true = np.zeros(50, dtype=int)

        metrics = calculate_expected_calibration_error(y_true, y_prob, n_bins=5)
        self.assertGreater(metrics["ece"], 0.50)
        self.assertEqual(metrics["quality"], "Poorly Calibrated")

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            calculate_expected_calibration_error([], [])


if __name__ == "__main__":
    unittest.main()
