"""
tests/test_model_leaderboard.py
--------------------------------
Unit tests for src/model_leaderboard.py
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from src.model_leaderboard import (
    update_leaderboard,
    get_leaderboard,
    get_champion_model,
    compare_models,
    LEADERBOARD_PATH,
)

SAMPLE_METRICS_A = {"accuracy": 0.82, "precision": 0.77, "recall": 0.65, "f1_score": 0.71, "roc_auc": 0.85}
SAMPLE_METRICS_B = {"accuracy": 0.80, "precision": 0.74, "recall": 0.70, "f1_score": 0.72, "roc_auc": 0.87}
SAMPLE_METRICS_C = {"accuracy": 0.78, "precision": 0.68, "recall": 0.60, "f1_score": 0.64, "roc_auc": 0.80}


class TestModelLeaderboard(unittest.TestCase):

    def setUp(self):
        """Use a temporary leaderboard file for every test."""
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        self.tmp.write("[]")
        self.tmp.close()
        self.patcher = patch("src.model_leaderboard.LEADERBOARD_PATH", self.tmp.name)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        os.unlink(self.tmp.name)

    # ── update_leaderboard ────────────────────────────────────────────────

    def test_update_adds_entry(self):
        update_leaderboard("XGBoost", "v1.0.0", SAMPLE_METRICS_A)
        df = get_leaderboard()
        self.assertEqual(len(df), 1)

    def test_update_overwrites_same_version(self):
        update_leaderboard("XGBoost", "v1.0.0", SAMPLE_METRICS_A)
        update_leaderboard("XGBoost", "v1.0.0", SAMPLE_METRICS_B)
        df = get_leaderboard()
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df["roc_auc"].iloc[0], SAMPLE_METRICS_B["roc_auc"])

    def test_update_multiple_models(self):
        update_leaderboard("XGBoost", "v1.0.0", SAMPLE_METRICS_A)
        update_leaderboard("RandomForest", "v1.0.0", SAMPLE_METRICS_C)
        df = get_leaderboard()
        self.assertEqual(len(df), 2)

    # ── get_leaderboard ───────────────────────────────────────────────────

    def test_leaderboard_sorted_by_roc_auc(self):
        update_leaderboard("XGBoost", "v1.0.0", SAMPLE_METRICS_A)
        update_leaderboard("XGBoost", "v1.1.0", SAMPLE_METRICS_B)
        df = get_leaderboard()
        self.assertGreaterEqual(df["roc_auc"].iloc[0], df["roc_auc"].iloc[1])

    def test_leaderboard_empty_returns_empty_df(self):
        df = get_leaderboard()
        self.assertEqual(len(df), 0)

    # ── get_champion_model ────────────────────────────────────────────────

    def test_champion_has_highest_roc_auc(self):
        update_leaderboard("XGBoost", "v1.0.0", SAMPLE_METRICS_A)
        update_leaderboard("XGBoost", "v1.1.0", SAMPLE_METRICS_B)
        update_leaderboard("RF", "v1.0.0", SAMPLE_METRICS_C)
        champion = get_champion_model()
        self.assertAlmostEqual(champion["roc_auc"], SAMPLE_METRICS_B["roc_auc"])
        self.assertEqual(champion["version"], "v1.1.0")

    def test_champion_none_when_empty(self):
        result = get_champion_model()
        self.assertIsNone(result)

    # ── compare_models ────────────────────────────────────────────────────

    def test_compare_returns_correct_keys(self):
        update_leaderboard("XGBoost", "v1.0.0", SAMPLE_METRICS_A)
        update_leaderboard("XGBoost", "v1.1.0", SAMPLE_METRICS_B)
        result = compare_models("v1.0.0", "v1.1.0")
        for key in ["version_a", "version_b", "delta", "winner"]:
            self.assertIn(key, result)

    def test_compare_winner_is_higher_roc(self):
        update_leaderboard("XGBoost", "v1.0.0", SAMPLE_METRICS_A)
        update_leaderboard("XGBoost", "v1.1.0", SAMPLE_METRICS_B)
        result = compare_models("v1.0.0", "v1.1.0")
        self.assertEqual(result["winner"], "v1.1.0")

    def test_compare_missing_version_raises(self):
        update_leaderboard("XGBoost", "v1.0.0", SAMPLE_METRICS_A)
        with self.assertRaises(ValueError):
            compare_models("v1.0.0", "v9.9.9")


if __name__ == "__main__":
    unittest.main()
