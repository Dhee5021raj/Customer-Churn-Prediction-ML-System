import unittest
import os
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.auditor import log_prediction_audit
from src.config import LOGS_DIR
from data.generate_data import generate_customer_churn_dataset


class TestAuditor(unittest.TestCase):

    def test_log_prediction_audit(self):
        df = generate_customer_churn_dataset(num_samples=5, random_state=42)
        probs = np.array([0.1, 0.4, 0.7, 0.2, 0.9])
        
        test_audit_path = os.path.join(LOGS_DIR, "test_predictions_audit.csv")
        if os.path.exists(test_audit_path):
            os.remove(test_audit_path)
            
        audit_file = log_prediction_audit(df, probs, source="test", audit_path=test_audit_path)
        self.assertTrue(os.path.exists(audit_file))
        
        df_log = pd.read_csv(audit_file)
        self.assertEqual(len(df_log), 5)
        self.assertIn("predicted_churn_prob", df_log.columns)
        self.assertIn("prediction_source", df_log.columns)
        
        if os.path.exists(test_audit_path):
            os.remove(test_audit_path)


if __name__ == "__main__":
    unittest.main()
