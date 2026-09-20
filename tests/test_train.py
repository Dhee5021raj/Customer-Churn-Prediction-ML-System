import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.dummy import DummyClassifier

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import evaluate_model, tune_xgboost_hyperparameters
from data.generate_data import generate_customer_churn_dataset
from src.data_loader import prepare_train_test_data


class TestTrain(unittest.TestCase):

    def test_evaluate_model(self):
        df = generate_customer_churn_dataset(num_samples=100, random_state=42)
        X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_train_test_data(df)
        
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X_train, y_train)
        
        metrics = evaluate_model(model, X_test, y_test)
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertIsInstance(metrics["accuracy"], float)

    def test_tune_xgboost_hyperparameters(self):
        df = generate_customer_churn_dataset(num_samples=100, random_state=42)
        X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_train_test_data(df)
        
        best_model = tune_xgboost_hyperparameters(X_train, y_train, n_iter=2, random_state=42)
        self.assertTrue(hasattr(best_model, "predict"))
        self.assertTrue(hasattr(best_model, "predict_proba"))


if __name__ == "__main__":
    unittest.main()
