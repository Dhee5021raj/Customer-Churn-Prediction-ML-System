import unittest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import (
    load_raw_data,
    get_preprocessor,
    prepare_train_test_data,
    validate_customer_data,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN
)
from data.generate_data import generate_customer_churn_dataset


class TestDataLoader(unittest.TestCase):

    def test_generate_and_validate_data(self):
        df = generate_customer_churn_dataset(num_samples=100, random_state=42)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 100)
        
        is_valid, missing = validate_customer_data(df, require_target=True)
        self.assertTrue(is_valid)
        self.assertEqual(len(missing), 0)

    def test_validate_customer_data_missing_columns(self):
        df = pd.DataFrame({"gender": ["Male"], "tenure": [12]})
        is_valid, missing = validate_customer_data(df, require_target=False)
        self.assertFalse(is_valid)
        self.assertGreater(len(missing), 0)
        self.assertIn("contract", missing)

    def test_prepare_train_test_data(self):
        df = generate_customer_churn_dataset(num_samples=200, random_state=42)
        X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_train_test_data(
            df, test_size=0.2, random_state=42
        )
        
        self.assertEqual(X_train.shape[0], 160)
        self.assertEqual(X_test.shape[0], 40)
        self.assertEqual(len(y_train), 160)
        self.assertEqual(len(y_test), 40)
        self.assertGreater(len(feature_names), 0)
        self.assertTrue(hasattr(preprocessor, "transform"))


if __name__ == "__main__":
    unittest.main()
