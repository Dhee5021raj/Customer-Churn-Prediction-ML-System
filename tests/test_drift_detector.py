import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.drift_detector import calculate_feature_drift
from data.generate_data import generate_customer_churn_dataset


class TestDriftDetector(unittest.TestCase):

    def test_calculate_feature_drift_identical(self):
        df = generate_customer_churn_dataset(num_samples=100, random_state=42)
        res = calculate_feature_drift(df, df)
        self.assertIn("drifted_features_count", res)
        self.assertEqual(res["drifted_features_count"], 0)
        self.assertFalse(res["drift_detected"])

    def test_calculate_feature_drift_shifted(self):
        df_base = generate_customer_churn_dataset(num_samples=100, random_state=42)
        df_curr = df_base.copy()
        # Shift numerical charges drastically
        df_curr["monthly_charges"] = df_curr["monthly_charges"] * 5.0
        
        res = calculate_feature_drift(df_base, df_curr)
        self.assertTrue(res["drift_detected"])
        self.assertTrue(res["feature_details"]["monthly_charges"]["is_drifted"])


if __name__ == "__main__":
    unittest.main()
