import unittest
import os
import sys
import pandas as pd
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.explainability import ChurnExplainer
from data.generate_data import generate_customer_churn_dataset


class TestExplainability(unittest.TestCase):

    def test_churn_explainer(self):
        self.assertTrue(os.path.exists("models/xgboost_model.pkl"))
        self.assertTrue(os.path.exists("models/preprocessor.pkl"))
        
        explainer = ChurnExplainer()
        df = generate_customer_churn_dataset(num_samples=10, random_state=42)
        
        explanation = explainer.explain_sample(df.iloc[:1])
        self.assertIn("churn_probability", explanation)
        self.assertIn("base_value", explanation)
        self.assertIn("top_features", explanation)
        self.assertIsInstance(explanation["churn_probability"], float)
        
        global_imp = explainer.get_global_feature_importance(df)
        self.assertIsInstance(global_imp, pd.DataFrame)
        self.assertIn("Feature", global_imp.columns)
        self.assertIn("Importance", global_imp.columns)
        self.assertGreater(len(global_imp), 0)


if __name__ == "__main__":
    unittest.main()
