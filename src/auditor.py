import os
import datetime
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from src.config import LOGS_DIR


def log_prediction_audit(
    df_input: pd.DataFrame,
    probabilities: np.ndarray,
    source: str = "single",
    audit_path: str = None
) -> str:
    """Log prediction input metadata, risk scores, and timestamps to audit CSV log."""
    if audit_path is None:
        audit_path = os.path.join(LOGS_DIR, "predictions_audit.csv")
        
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    timestamp = datetime.datetime.now().isoformat()
    df_audit = df_input.copy()
    
    df_audit["predicted_churn_prob"] = np.round(probabilities, 4)
    df_audit["prediction_timestamp"] = timestamp
    df_audit["prediction_source"] = source
    
    header = not os.path.exists(audit_path)
    df_audit.to_csv(audit_path, mode="a", index=False, header=header)
    return audit_path
