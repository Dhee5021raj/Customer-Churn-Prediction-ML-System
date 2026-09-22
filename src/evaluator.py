import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix


def plot_confusion_matrices(results: Dict[str, Dict[str, Any]], output_dir: str = "reports/figures") -> str:
    """Plot and save confusion matrix heatmaps for benchmark models."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    
    if len(results) == 1:
        axes = [axes]
        
    for ax, (model_name, metrics) in zip(axes, results.items()):
        cm = np.array(metrics["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_title(f"{model_name}\nConfusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        
    plt.tight_layout()
    file_path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(file_path, dpi=300)
    plt.close()
    return file_path


def plot_roc_curves(models: Dict[str, Any], X_test: np.ndarray, y_test: pd.Series, output_dir: str = "reports/figures") -> str:
    """Plot and save ROC Curves comparison for all trained models."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.predict(X_test)
            
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.4f})")
        
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Benchmark Comparison")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    file_path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(file_path, dpi=300)
    plt.close()
    return file_path
