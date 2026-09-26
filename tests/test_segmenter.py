"""
tests/test_segmenter.py
-----------------------
Unit tests for src/segmenter.py
"""

import unittest
import pandas as pd
import numpy as np

from src.segmenter import segment_customers, get_segment_summary


def _make_customer_df(n=80):
    np.random.seed(42)
    return pd.DataFrame({
        "monthly_charges": np.random.uniform(20, 120, n),
        "tenure_months": np.random.randint(1, 72, n),
        "num_support_tickets": np.random.randint(0, 10, n),
        "monthly_usage_gb": np.random.uniform(5, 200, n),
        "num_products": np.random.randint(1, 5, n),
        "churn_probability": np.random.uniform(0.1, 0.9, n),
        "clv": np.random.uniform(500, 5000, n),
    })


class TestSegmenter(unittest.TestCase):

    def test_segment_customers_adds_column(self):
        df = _make_customer_df()
        result = segment_customers(df, n_clusters=4)
        self.assertIn("customer_segment", result.columns)
        self.assertEqual(len(result), len(df))

    def test_segment_customers_produces_4_cohorts(self):
        df = _make_customer_df(n=100)
        result = segment_customers(df, n_clusters=4)
        unique_segments = result["customer_segment"].nunique()
        self.assertEqual(unique_segments, 4)

    def test_segment_customers_does_not_mutate_original(self):
        df = _make_customer_df()
        original_cols = list(df.columns)
        segment_customers(df, n_clusters=4)
        self.assertEqual(list(df.columns), original_cols)

    def test_get_segment_summary_shape(self):
        df = _make_customer_df()
        df_seg = segment_customers(df, n_clusters=4)
        summary = get_segment_summary(df_seg)
        self.assertEqual(len(summary), 4)
        self.assertIn("customer_count", summary.columns)
        self.assertIn("avg_churn_probability", summary.columns)

    def test_get_segment_summary_raises_without_segment_col(self):
        df = _make_customer_df()
        with self.assertRaises(ValueError):
            get_segment_summary(df)


if __name__ == "__main__":
    unittest.main()
