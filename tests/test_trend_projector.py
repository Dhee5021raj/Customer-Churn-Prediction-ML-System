"""
tests/test_trend_projector.py
------------------------------
Unit tests for src/trend_projector.py
"""

import unittest

from src.trend_projector import (
    project_churn_trend,
    get_trend_summary,
    project_retention_scenario,
)


class TestTrendProjector(unittest.TestCase):

    # ── project_churn_trend ───────────────────────────────────────────────

    def test_output_length(self):
        result = project_churn_trend(0.5, periods=12)
        self.assertEqual(len(result), 12)

    def test_output_keys(self):
        result = project_churn_trend(0.5, periods=3)
        for row in result:
            self.assertIn("period", row)
            self.assertIn("projected_probability", row)
            self.assertIn("trend_direction", row)

    def test_period_index_sequence(self):
        result = project_churn_trend(0.5, periods=6)
        self.assertEqual([r["period"] for r in result], list(range(1, 7)))

    def test_decay_reduces_probability(self):
        result = project_churn_trend(0.7, periods=6, decay_rate=0.10)
        self.assertLess(result[-1]["projected_probability"], 0.7)

    def test_growth_increases_probability(self):
        result = project_churn_trend(0.3, periods=6, growth_rate=0.10)
        self.assertGreater(result[-1]["projected_probability"], 0.3)

    def test_no_change_stays_stable(self):
        result = project_churn_trend(0.5, periods=5, decay_rate=0.0, growth_rate=0.0)
        for row in result:
            self.assertAlmostEqual(row["projected_probability"], 0.5, places=2)
            self.assertEqual(row["trend_direction"], "Stable")

    def test_probability_clamped_max(self):
        result = project_churn_trend(0.99, periods=5, growth_rate=0.5)
        for row in result:
            self.assertLessEqual(row["projected_probability"], 0.98)

    def test_probability_clamped_min(self):
        result = project_churn_trend(0.01, periods=5, decay_rate=0.9)
        for row in result:
            self.assertGreaterEqual(row["projected_probability"], 0.02)

    def test_invalid_periods_raises(self):
        with self.assertRaises(ValueError):
            project_churn_trend(0.5, periods=0)

    # ── get_trend_summary ─────────────────────────────────────────────────

    def test_summary_keys(self):
        proj = project_churn_trend(0.5, periods=6, growth_rate=0.05)
        summary = get_trend_summary(proj)
        for key in ["start_probability", "end_probability", "overall_direction", "total_change"]:
            self.assertIn(key, summary)

    def test_summary_worsening(self):
        proj = project_churn_trend(0.3, periods=6, growth_rate=0.15)
        summary = get_trend_summary(proj)
        self.assertEqual(summary["overall_direction"], "Worsening")

    def test_summary_improving(self):
        proj = project_churn_trend(0.8, periods=6, decay_rate=0.15)
        summary = get_trend_summary(proj)
        self.assertEqual(summary["overall_direction"], "Improving")

    def test_summary_empty_raises(self):
        with self.assertRaises(ValueError):
            get_trend_summary([])

    # ── project_retention_scenario ────────────────────────────────────────

    def test_scenario_output_length(self):
        result = project_retention_scenario(0.6, periods=12)
        self.assertEqual(len(result), 12)

    def test_scenario_intervention_lower_than_baseline(self):
        result = project_retention_scenario(0.6, periods=12, intervention_decay=0.08)
        for row in result:
            self.assertLessEqual(
                row["intervention_probability"],
                row["baseline_probability"] + 1e-9,
            )

    def test_scenario_keys(self):
        result = project_retention_scenario(0.5, periods=3)
        for row in result:
            for key in ["period", "baseline_probability", "intervention_probability", "probability_saved"]:
                self.assertIn(key, row)


if __name__ == "__main__":
    unittest.main()
