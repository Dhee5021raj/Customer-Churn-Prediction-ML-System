import os
import logging
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Directories
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "figures")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Dataset File Paths
RAW_DATA_PATH = os.path.join(DATA_DIR, "customer_churn.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
BEST_PARAMS_PATH = os.path.join(MODELS_DIR, "best_params.json")

# Features
NUMERICAL_FEATURES = [
    "senior_citizen", "tenure", "monthly_charges", "total_charges", "num_support_tickets"
]

CATEGORICAL_FEATURES = [
    "gender", "partner", "dependents", "phone_service", "multiple_lines",
    "internet_service", "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies", "contract",
    "paperless_billing", "payment_method"
]

TARGET_COLUMN = "churn"
ID_COLUMN = "customer_id"


def setup_logger(name: str = "pipeline", log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Set up configured Python logger with console and file handlers."""
    if log_file is None:
        log_file = os.path.join(LOGS_DIR, "pipeline.log")
        
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s: %(message)s")
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger
