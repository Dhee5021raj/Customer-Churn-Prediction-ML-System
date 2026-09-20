import unittest
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recommender import generate_retention_recommendations


class TestRecommender(unittest.TestCase):

    def test_generate_retention_recommendations_high_risk(self):
        profile = {
            "contract": "Month-to-month",
            "tech_support": "No",
            "num_support_tickets": 4,
            "payment_method": "Electronic check",
            "monthly_charges": 95.0,
            "online_security": "No"
        }
        recs = generate_retention_recommendations(profile)
        self.assertIsInstance(recs, list)
        self.assertGreater(len(recs), 0)
        self.assertTrue(any("Contract" in r for r in recs))
        self.assertTrue(any("Tech Support" in r for r in recs))

    def test_generate_retention_recommendations_low_risk(self):
        profile = {
            "contract": "Two year",
            "tech_support": "Yes",
            "num_support_tickets": 0,
            "payment_method": "Credit card (automatic)",
            "monthly_charges": 45.0,
            "online_security": "Yes"
        }
        recs = generate_retention_recommendations(profile)
        self.assertIsInstance(recs, list)
        self.assertEqual(len(recs), 1)
        self.assertIn("Standard Engagement", recs[0])


if __name__ == "__main__":
    unittest.main()
