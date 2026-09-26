"""
src/data_profiler.py
--------------------
Dataset column profiler and preprocessing health checker.

Functions:
    profile_dataset(df) -> pd.DataFrame  — per-column stats
    check_data_health(df) -> dict         — high-level quality flags
"""

import pandas as pd
import numpy as np
from typing import List

from src.config import setup_logger

logger = setup_logger("data_profiler")

SKEWNESS_THRESHOLD = 2.0
HIGH_CARDINALITY_THRESHOLD = 50  # unique values count


def profile_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a per-column profile of the dataset.

    For each column returns:
        - dtype
        - missing_count, missing_pct
        - unique_count, cardinality_flag
        - For numerical: mean, std, min, max, skewness
        - For categorical: top_category, top_category_freq

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Profile table with one row per column.
    """
    records = []

    for col in df.columns:
        series = df[col]
        missing_count = int(series.isna().sum())
        missing_pct = round(missing_count / len(df) * 100, 2) if len(df) > 0 else 0.0
        unique_count = int(series.nunique())
        cardinality_flag = unique_count >= HIGH_CARDINALITY_THRESHOLD

        row: dict = {
            "column": col,
            "dtype": str(series.dtype),
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "unique_count": unique_count,
            "high_cardinality": cardinality_flag,
        }

        if pd.api.types.is_numeric_dtype(series):
            row["column_type"] = "numerical"
            row["mean"] = round(series.mean(), 4) if not series.empty else None
            row["std"] = round(series.std(), 4) if not series.empty else None
            row["min"] = round(series.min(), 4) if not series.empty else None
            row["max"] = round(series.max(), 4) if not series.empty else None
            row["skewness"] = round(float(series.skew()), 4) if series.count() > 2 else None
            row["top_category"] = None
            row["top_category_freq"] = None
        else:
            row["column_type"] = "categorical"
            row["mean"] = None
            row["std"] = None
            row["min"] = None
            row["max"] = None
            row["skewness"] = None
            vc = series.value_counts()
            if not vc.empty:
                row["top_category"] = str(vc.index[0])
                row["top_category_freq"] = round(vc.iloc[0] / len(df) * 100, 2)
            else:
                row["top_category"] = None
                row["top_category_freq"] = None

        records.append(row)

    profile_df = pd.DataFrame(records)
    logger.info(f"Profiled {len(profile_df)} columns from dataset with {len(df)} rows.")
    return profile_df


def check_data_health(df: pd.DataFrame) -> dict:
    """
    Runs a high-level data quality health check on the dataset.

    Returns
    -------
    dict with keys:
        - has_missing: bool
        - missing_cols: list of cols with any missing values
        - high_cardinality_cols: list of cols with unique count >= threshold
        - constant_cols: list of cols with only 1 unique value
        - skewed_cols: list of numerical cols with |skewness| > SKEWNESS_THRESHOLD
        - total_rows: int
        - total_cols: int
        - overall_health: "Good" | "Needs Review" | "Poor"
    """
    missing_cols: List[str] = [c for c in df.columns if df[c].isna().any()]
    constant_cols: List[str] = [c for c in df.columns if df[c].nunique() <= 1]
    high_card_cols: List[str] = [c for c in df.columns if df[c].nunique() >= HIGH_CARDINALITY_THRESHOLD]

    skewed_cols: List[str] = []
    for c in df.select_dtypes(include="number").columns:
        try:
            if abs(df[c].skew()) > SKEWNESS_THRESHOLD:
                skewed_cols.append(c)
        except Exception:
            pass

    issue_count = len(missing_cols) + len(constant_cols) + len(skewed_cols)
    if issue_count == 0:
        overall_health = "Good"
    elif issue_count <= 3:
        overall_health = "Needs Review"
    else:
        overall_health = "Poor"

    result = {
        "has_missing": len(missing_cols) > 0,
        "missing_cols": missing_cols,
        "high_cardinality_cols": high_card_cols,
        "constant_cols": constant_cols,
        "skewed_cols": skewed_cols,
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "overall_health": overall_health,
    }

    logger.info(f"Data health check complete: {overall_health} — {issue_count} issue(s) detected.")
    return result
