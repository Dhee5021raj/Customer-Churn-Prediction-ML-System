import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

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


def validate_customer_data(df: pd.DataFrame, require_target: bool = False) -> Tuple[bool, List[str]]:
    """Validate that incoming DataFrame contains all required feature columns."""
    expected_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    if require_target:
        expected_cols.append(TARGET_COLUMN)
    
    missing_cols = [col for col in expected_cols if col not in df.columns]
    is_valid = len(missing_cols) == 0
    return is_valid, missing_cols


def load_raw_data(data_path: str = "data/customer_churn.csv") -> pd.DataFrame:
    """Load raw churn dataset from CSV."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run data/generate_data.py first.")
    return pd.read_csv(data_path)


def get_preprocessor() -> ColumnTransformer:
    """Construct Scikit-Learn ColumnTransformer for numerical & categorical features."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Retrieve processed feature names from fitted ColumnTransformer."""
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    return NUMERICAL_FEATURES + cat_feature_names


def prepare_train_test_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, ColumnTransformer, List[str]]:
    """Clean, split, fit preprocessor, and return scaled feature arrays and target series."""
    df = df.copy()
    
    # Handle missing values if any
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce").fillna(0.0)
    
    X = df.drop(columns=[ID_COLUMN, TARGET_COLUMN], errors="ignore")
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = get_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    feature_names = get_feature_names(preprocessor)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names


def save_preprocessor(preprocessor: ColumnTransformer, filepath: str = "models/preprocessor.pkl") -> None:
    """Save fitted preprocessor artifact to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(preprocessor, filepath)


def load_preprocessor(filepath: str = "models/preprocessor.pkl") -> ColumnTransformer:
    """Load preprocessor artifact from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Preprocessor not found at {filepath}")
    return joblib.load(filepath)
