import os
import sys
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from typing import Dict, Any, Tuple, List

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import get_feature_names, load_preprocessor


class ChurnExplainer:
    """Class to manage SHAP (SHapley Additive exPlanations) for Customer Churn XGBoost Model."""
    
    def __init__(self, model_path: str = "models/xgboost_model.pkl", preprocessor_path: str = "models/preprocessor.pkl"):
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            raise FileNotFoundError("Model or preprocessor artifacts not found. Train the model first.")
            
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.feature_names = get_feature_names(self.preprocessor)
        self.explainer = shap.TreeExplainer(self.model)

    def explain_sample(self, customer_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate SHAP feature impact breakdown for a single customer input DataFrame."""
        X_processed = self.preprocessor.transform(customer_df)
        shap_values = self.explainer(X_processed)
        
        # Extract base value and shap values for single instance
        base_value = float(shap_values.base_values[0])
        sample_shap = shap_values.values[0]
        
        # Raw customer values mapped to transformed feature names for clear visualization
        feature_impacts = []
        for name, val, shap_val in zip(self.feature_names, X_processed[0], sample_shap):
            if abs(shap_val) > 0.001:  # Filter negligible features
                feature_impacts.append({
                    "feature": name,
                    "value": round(float(val), 3),
                    "shap_value": round(float(shap_val), 4),
                    "impact": "Increases Churn Risk" if shap_val > 0 else "Decreases Churn Risk"
                })

        # Sort by magnitude of SHAP impact
        feature_impacts.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        
        # Model churn probability
        churn_prob = float(self.model.predict_proba(X_processed)[0][1])

        return {
            "churn_probability": round(churn_prob, 4),
            "base_value": round(base_value, 4),
            "top_features": feature_impacts[:10],
            "raw_shap_values": sample_shap,
            "processed_input": X_processed
        }

    def get_global_feature_importance(self, df_sample: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean absolute SHAP value per feature over a sample dataset."""
        X_processed = self.preprocessor.transform(df_sample)
        shap_values = self.explainer.shap_values(X_processed)
        
        mean_shap = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": mean_shap
        }).sort_values(by="Importance", ascending=False)
        
        return importance_df

    def get_global_importance_figure(self, df_sample: pd.DataFrame, top_n: int = 15):
        """Generate interactive Plotly bar chart for top N global SHAP feature importances."""
        import plotly.express as px
        importance_df = self.get_global_feature_importance(df_sample).head(top_n)
        
        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Global Feature Importance — Top {top_n} Features (SHAP)",
            labels={"Importance": "Mean |SHAP Value| (Impact on Model Outcome)"},
            color="Importance",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        return fig

