"""
tests/test_data_profiler.py
---------------------------
Unit tests for src/data_profiler.py
"""

import unittest
import pandas as pd
import numpy as np

from src.data_profiler import profile_dataset, check_data_health


def _make_test_df():
    np.random.seed(0)
    return pd.DataFrame({
        "age": np.random.randint(18, 80, 100),
        "monthly_charges": np.random.uniform(20, 120, 100),
        "contract_type": np.random.choice(["Month-to-Month", "One Year", "Two Year"], 100),
        "constant_col": [42] * 100,
        "skewed_col": np.concatenate([np.zeros(90), np.ones(10) * 1000]),  # highly skewed
    })


class TestDataProfiler(unittest.TestCase):

    def test_profile_dataset_shape(self):
        df = _make_test_df()
        profile = profile_dataset(df)
        self.assertEqual(len(profile), len(df.columns))
        self.assertIn("column", profile.columns)
        self.assertIn("missing_pct", profile.columns)
        self.assertIn("dtype", profile.columns)

    def test_profile_numerical_columns(self):
        df = _make_test_df()
        profile = profile_dataset(df)
        age_row = profile[profile["column"] == "age"].iloc[0]
        self.assertEqual(age_row["column_type"], "numerical")
        self.assertIsNotNone(age_row["mean"])
        self.assertIsNotNone(age_row["skewness"])

    def test_profile_categorical_columns(self):
        df = _make_test_df()
        profile = profile_dataset(df)
        cat_row = profile[profile["column"] == "contract_type"].iloc[0]
        self.assertEqual(cat_row["column_type"], "categorical")
        self.assertIsNotNone(cat_row["top_category"])
        self.assertIsNotNone(cat_row["top_category_freq"])

    def test_check_data_health_missing(self):
        df = _make_test_df()
        df.loc[0:5, "age"] = np.nan
        health = check_data_health(df)
        self.assertTrue(health["has_missing"])
        self.assertIn("age", health["missing_cols"])

    def test_check_data_health_constant_col(self):
        df = _make_test_df()
        health = check_data_health(df)
        self.assertIn("constant_col", health["constant_cols"])

    def test_check_data_health_skewed_col(self):
        df = _make_test_df()
        health = check_data_health(df)
        self.assertIn("skewed_col", health["skewed_cols"])

    def test_check_data_health_returns_overall_rating(self):
        df = _make_test_df()
        health = check_data_health(df)
        self.assertIn(health["overall_health"], ["Good", "Needs Review", "Poor"])

    def test_profile_no_missing_values(self):
        df = _make_test_df()
        profile = profile_dataset(df)
        age_row = profile[profile["column"] == "age"].iloc[0]
        self.assertEqual(age_row["missing_count"], 0)
        self.assertEqual(age_row["missing_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
