import unittest
import os
import json
import sys
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_registry import register_model, get_latest_model_metadata
from src.config import MODELS_DIR


class TestModelRegistry(unittest.TestCase):

    def test_register_model(self):
        model = DummyClassifier(strategy="most_frequent")
        preprocessor = StandardScaler()
        metrics = {"accuracy": 0.85, "roc_auc": 0.90}
        
        test_registry = os.path.join(MODELS_DIR, "test_registry.json")
        if os.path.exists(test_registry):
            os.remove(test_registry)
            
        entry = register_model(model, preprocessor, metrics, version="v1.0.0-test", registry_path=test_registry, save_as_active=False)
        self.assertEqual(entry["version"], "v1.0.0-test")
        self.assertTrue(os.path.exists(test_registry))
        
        metadata = get_latest_model_metadata(registry_path=test_registry)
        self.assertEqual(metadata["version"], "v1.0.0-test")
        
        if os.path.exists(test_registry):
            os.remove(test_registry)


if __name__ == "__main__":
    unittest.main()
