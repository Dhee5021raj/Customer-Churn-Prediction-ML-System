import unittest
import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    BASE_DIR, DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, setup_logger
)


class TestConfig(unittest.TestCase):

    def test_config_paths_exist(self):
        self.assertTrue(os.path.exists(DATA_DIR))
        self.assertTrue(os.path.exists(MODELS_DIR))
        self.assertTrue(os.path.exists(REPORTS_DIR))
        self.assertTrue(os.path.exists(LOGS_DIR))

    def test_feature_constants(self):
        self.assertGreater(len(NUMERICAL_FEATURES), 0)
        self.assertGreater(len(CATEGORICAL_FEATURES), 0)

    def test_setup_logger(self):
        logger = setup_logger("test_logger")
        self.assertIsNotNone(logger)
        logger.info("Test log message")
        log_file = os.path.join(LOGS_DIR, "pipeline.log")
        self.assertTrue(os.path.exists(log_file))


if __name__ == "__main__":
    unittest.main()
