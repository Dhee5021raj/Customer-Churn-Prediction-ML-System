import unittest
import os
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.threshold_optimizer import optimize_classification_threshold, get_optimal_threshold
from src.config import MODELS_DIR


class TestThresholdOptimizer(unittest.TestCase):

    def test_optimize_classification_threshold(self):
        y_true = pd.Series([0, 0, 0, 1, 1, 1, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.2, 0.4, 0.65, 0.8, 0.7, 0.9, 0.3, 0.85, 0.15])
        
        test_path = os.path.join(MODELS_DIR, "test_optimal_threshold.json")
        if os.path.exists(test_path):
            os.remove(test_path)
            
        res = optimize_classification_threshold(y_true, y_prob, output_path=test_path)
        self.assertIn("optimal_threshold", res)
        self.assertIn("f1_score", res)
        self.assertTrue(os.path.exists(test_path))
        
        th = get_optimal_threshold(threshold_path=test_path)
        self.assertEqual(th, res["optimal_threshold"])
        
        if os.path.exists(test_path):
            os.remove(test_path)


if __name__ == "__main__":
    unittest.main()
