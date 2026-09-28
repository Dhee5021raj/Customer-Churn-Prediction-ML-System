"""
tests/test_retrain_pipeline.py
------------------------------
Unit tests for src/retrain_pipeline.py
"""

import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np

from src.retrain_pipeline import (
    check_retraining_trigger,
    evaluate_candidate_vs_champion,
    run_retraining_cycle,
)
from data.generate_data import generate_customer_churn_dataset


class TestRetrainPipeline(unittest.TestCase):

    def test_check_retraining_trigger_threshold_exceeded(self):
        drift_report = {
            "drift_detected": True,
            "drifted_features_count": 3,
        }
        res = check_retraining_trigger(drift_report, min_drifted_features=2)
        self.assertTrue(res["should_retrain"])
        self.assertEqual(res["drifted_features"], 3)
        self.assertIn("exceeded", res["reason"])

    def test_check_retraining_trigger_no_drift(self):
        drift_report = {
            "drift_detected": False,
            "drifted_features_count": 0,
        }
        res = check_retraining_trigger(drift_report, min_drifted_features=2)
        self.assertFalse(res["should_retrain"])
        self.assertIn("stable", res["reason"].lower())

    def test_evaluate_candidate_vs_champion_no_existing(self):
        cand = {"roc_auc": 0.85, "f1_score": 0.50}
        res = evaluate_candidate_vs_champion(cand, champion_metrics=None)
        self.assertTrue(res["promoted"])
        self.assertIn("Initial champion", res["decision"])

    def test_evaluate_candidate_vs_champion_promoted(self):
        cand = {"roc_auc": 0.88}
        champ = {"roc_auc": 0.85}
        res = evaluate_candidate_vs_champion(cand, champ, min_improvement=0.01)
        self.assertTrue(res["promoted"])
        self.assertGreater(res["delta"], 0)

    def test_evaluate_candidate_vs_champion_rejected(self):
        cand = {"roc_auc": 0.84}
        champ = {"roc_auc": 0.86}
        res = evaluate_candidate_vs_champion(cand, champ, min_improvement=0.01)
        self.assertFalse(res["promoted"])
        self.assertLess(res["delta"], 0)

    def test_run_retraining_cycle_execution(self):
        # Generate small dataset to test full cycle quickly
        df = generate_customer_churn_dataset(num_samples=150, random_state=42)
        res = run_retraining_cycle(df, version_tag="test_v1", tune=False, save_as_active=False)
        self.assertEqual(res["version"], "test_v1")
        self.assertIn("roc_auc", res["candidate_metrics"])
        self.assertIn("evaluation", res)
        self.assertIn("promoted", res["evaluation"])


if __name__ == "__main__":
    unittest.main()
