import os
import json
import joblib
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from src.config import MODELS_DIR


def register_model(
    model: Any,
    preprocessor: Any,
    metrics: Dict[str, Any],
    version: str = "v1.0.0",
    registry_path: str = None,
    save_as_active: bool = True
) -> Dict[str, Any]:
    """
    Persist model & preprocessor artifacts and log semantic version metadata 
    to central model registry manifest JSON.
    """
    if registry_path is None:
        registry_path = os.path.join(MODELS_DIR, "registry.json")
        
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    timestamp = datetime.datetime.now().isoformat()
    
    # Artifact paths
    model_filename = f"xgboost_{version}.pkl"
    preprocessor_filename = f"preprocessor_{version}.pkl"
    
    model_path = os.path.join(MODELS_DIR, model_filename)
    preprocessor_path = os.path.join(MODELS_DIR, preprocessor_filename)
    
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    
    # Save active default artifacts if requested
    if save_as_active:
        joblib.dump(model, os.path.join(MODELS_DIR, "xgboost_model.pkl"))
        joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))
    
    entry = {
        "version": version,
        "timestamp": timestamp,
        "model_file": model_filename,
        "preprocessor_file": preprocessor_filename,
        "metrics": metrics,
        "status": "active"
    }
    
    registry = []
    if os.path.exists(registry_path):
        try:
            with open(registry_path, "r") as f:
                registry = json.load(f)
        except Exception:
            registry = []
            
    registry.append(entry)
    
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=4)
        
    return entry


def get_latest_model_metadata(registry_path: str = None) -> Dict[str, Any]:
    """Retrieve metadata entry for the latest active registered model."""
    if registry_path is None:
        registry_path = os.path.join(MODELS_DIR, "registry.json")
        
    if not os.path.exists(registry_path):
        return {}
        
    try:
        with open(registry_path, "r") as f:
            registry = json.load(f)
        if registry:
            return registry[-1]
    except Exception:
        pass
        
    return {}
