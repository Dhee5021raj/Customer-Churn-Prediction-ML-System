"""
src/model_leaderboard.py
------------------------
Model champion-challenger leaderboard tracker.

Maintains a versioned log of registered model performance metrics
and identifies the current champion by ROC-AUC.

Functions:
    update_leaderboard(model_name, version, metrics) -> dict
    get_leaderboard() -> pd.DataFrame
    get_champion_model() -> dict | None
    compare_models(version_a, version_b) -> dict
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from src.config import BASE_DIR, setup_logger

logger = setup_logger("model_leaderboard")

LEADERBOARD_PATH = os.path.join(BASE_DIR, "models", "leaderboard.json")
RANK_METRIC = "roc_auc"
METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]


def _load_leaderboard() -> list:
    if os.path.exists(LEADERBOARD_PATH):
        with open(LEADERBOARD_PATH, "r") as f:
            return json.load(f)
    return []


def _save_leaderboard(entries: list) -> None:
    os.makedirs(os.path.dirname(LEADERBOARD_PATH), exist_ok=True)
    with open(LEADERBOARD_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def update_leaderboard(
    model_name: str,
    version: str,
    metrics: dict,
) -> dict:
    """
    Appends or updates a model entry in the leaderboard.

    If the same (model_name, version) already exists it is overwritten.

    Parameters
    ----------
    model_name : str
        e.g. "XGBoost", "RandomForest", "LogisticRegression"
    version : str
        Semantic version string, e.g. "v1.0.0"
    metrics : dict
        Must contain at minimum 'roc_auc'. Other standard keys optional.

    Returns
    -------
    dict — the entry that was saved.
    """
    entries = _load_leaderboard()

    entry = {
        "model_name": model_name,
        "version": version,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        **{k: round(float(metrics.get(k, 0.0)), 4) for k in METRIC_COLUMNS if k in metrics},
    }

    # Overwrite existing entry with same name + version
    entries = [e for e in entries if not (e["model_name"] == model_name and e["version"] == version)]
    entries.append(entry)

    _save_leaderboard(entries)
    logger.info(f"Leaderboard updated: {model_name} {version} — ROC-AUC={metrics.get(RANK_METRIC, '?')}")
    return entry


def get_leaderboard() -> pd.DataFrame:
    """
    Returns the full leaderboard sorted by ROC-AUC descending.

    Returns
    -------
    pd.DataFrame — empty if no entries recorded yet.
    """
    entries = _load_leaderboard()
    if not entries:
        return pd.DataFrame()

    df = pd.DataFrame(entries)
    if RANK_METRIC in df.columns:
        df = df.sort_values(RANK_METRIC, ascending=False).reset_index(drop=True)
        df.index = df.index + 1          # rank starts at 1
        df.index.name = "rank"

    return df


def get_champion_model() -> Optional[dict]:
    """
    Returns the entry with the highest ROC-AUC score.

    Returns
    -------
    dict or None if leaderboard is empty.
    """
    entries = _load_leaderboard()
    if not entries:
        return None

    valid = [e for e in entries if RANK_METRIC in e]
    if not valid:
        return None

    champion = max(valid, key=lambda e: e[RANK_METRIC])
    logger.info(f"Champion: {champion['model_name']} {champion['version']} — ROC-AUC={champion[RANK_METRIC]}")
    return champion


def compare_models(version_a: str, version_b: str) -> dict:
    """
    Side-by-side metric diff between two versions (any model name).

    Returns
    -------
    dict: {
        "version_a": {...metrics},
        "version_b": {...metrics},
        "delta": {metric: b_value - a_value},
        "winner": version of the higher ROC-AUC
    }
    """
    entries = _load_leaderboard()
    lookup = {e["version"]: e for e in entries}

    if version_a not in lookup:
        raise ValueError(f"Version '{version_a}' not found in leaderboard.")
    if version_b not in lookup:
        raise ValueError(f"Version '{version_b}' not found in leaderboard.")

    a = lookup[version_a]
    b = lookup[version_b]

    delta = {
        m: round(b.get(m, 0.0) - a.get(m, 0.0), 4)
        for m in METRIC_COLUMNS
        if m in a or m in b
    }

    winner = version_b if b.get(RANK_METRIC, 0) >= a.get(RANK_METRIC, 0) else version_a

    return {
        "version_a": a,
        "version_b": b,
        "delta": delta,
        "winner": winner,
    }
