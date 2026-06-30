"""
SHAP-based model explainability.
Returns top contributing features for each prediction.
"""

import logging
import os
from typing import Dict, List, Optional

import joblib
import numpy as np
import shap
import xgboost as xgb

from .feature_builder import FeatureBuilder

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "freshness_xgb_v1.json")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "feature_builder_v1.pkl")


class FreshnessExplainer:
    """
    SHAP explainer for the XGBoost freshness model.
    Provides per-prediction feature contributions.
    """

    def __init__(self):
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_builder: Optional[FeatureBuilder] = None
        self.shap_explainer: Optional[shap.TreeExplainer] = None
        self._loaded = False
        self._load()

    def _load(self):
        """Load model and create SHAP explainer."""
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"Model not found at {MODEL_PATH}")
            return

        try:
            self.model = xgb.XGBClassifier()
            self.model.load_model(MODEL_PATH)

            if os.path.exists(ENCODER_PATH):
                self.feature_builder = joblib.load(ENCODER_PATH)

            self.shap_explainer = shap.TreeExplainer(self.model)
            self._loaded = True
            logger.info("SHAP explainer loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load SHAP explainer: {e}")

    def explain(self, input_data: Dict, top_n: int = 3) -> List[Dict]:
        """
        Get top N SHAP feature contributions for a prediction.

        Args:
            input_data: Raw input dictionary (same format as predictor)
            top_n: Number of top features to return

        Returns:
            List of dicts: [{"feature": "storage_time", "impact": -0.18, "direction": "negative"}, ...]
        """
        if not self._loaded:
            return []

        try:
            # Transform input
            X = self.feature_builder.transform_single(input_data)

            # Compute SHAP values
            shap_values = self.shap_explainer.shap_values(X)

            # Get feature names
            feature_names = list(X.columns)

            # For multi-class, shap_values can be:
            # - list of arrays (one per class): [class0_arr, class1_arr, ...]
            # - 3D ndarray: (n_samples, n_features, n_classes)
            pred_idx = int(np.argmax(self.model.predict_proba(X)[0]))

            if isinstance(shap_values, list):
                values = shap_values[pred_idx][0]  # First sample, predicted class
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                values = shap_values[0, :, pred_idx]  # First sample, all features, predicted class
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
                values = shap_values[0]  # Binary classification — first sample
            else:
                values = shap_values[0]

            # Build feature contribution list
            contributions = []
            for i, (name, val) in enumerate(zip(feature_names, values)):
                # Make feature name human-readable
                readable = name.replace("_encoded", "").replace("_", " ").title()
                contributions.append(
                    {
                        "feature": readable,
                        "raw_feature": name,
                        "impact": round(float(val), 4),
                        "direction": "positive" if val > 0 else "negative" if val < 0 else "neutral",
                    }
                )

            # Sort by absolute impact, return top N
            contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
            return contributions[:top_n]

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return []

    def explain_to_text(self, input_data: Dict) -> str:
        """
        Generate human-readable explanation from SHAP values.
        Used for NGO-facing freshness descriptions.
        """
        features = self.explain(input_data, top_n=3)

        if not features:
            return "Explanation unavailable."

        parts = []
        for f in features:
            direction = "favors" if f["direction"] == "positive" else "reduces"
            parts.append(f"{f['feature']} {direction} freshness")

        return "Top factors: " + "; ".join(parts) + "."

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# Singleton
_explainer_instance: Optional[FreshnessExplainer] = None


def get_explainer() -> FreshnessExplainer:
    global _explainer_instance
    if _explainer_instance is None or not _explainer_instance.is_loaded:
        _explainer_instance = FreshnessExplainer()
    return _explainer_instance
