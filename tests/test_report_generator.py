import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.report_generator import generate_customer_intervention_tasklist
from data.generate_data import generate_customer_churn_dataset
from src.clv_calculator import calculate_clv_risk


class TestReportGenerator(unittest.TestCase):

    def test_generate_customer_intervention_tasklist(self):
        df = generate_customer_churn_dataset(num_samples=20, random_state=42)
        probs = np.random.uniform(0.0, 1.0, size=len(df))
        annotated_df = calculate_clv_risk(df, probs)
        
        tasklist = generate_customer_intervention_tasklist(annotated_df)
        self.assertIn("intervention_priority", tasklist.columns)
        self.assertIn("recommended_playbook", tasklist.columns)
        self.assertEqual(len(tasklist), 20)


if __name__ == "__main__":
    unittest.main()
