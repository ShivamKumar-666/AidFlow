"""
XGBoost predictor for food freshness.
Loads the trained model and FeatureBuilder for inference.
"""

import logging
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import joblib
import numpy as np
import xgboost as xgb

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "freshness_xgb_v1.json")
ENCODER_PATH = os.path.join(MODEL_DIR, "feature_builder_v1.pkl")
# Fallback paths (old model)
FALLBACK_MODEL_PATH = os.path.join(MODEL_DIR, "freshness_xgb.json")
FALLBACK_ENCODER_PATH = os.path.join(MODEL_DIR, "encoders.pkl")


@dataclass
class PredictionResult:
    """Structured prediction output."""

    freshness_label: str
    freshness_score: int
    confidence: float
    probabilities: Dict[str, float]
    shap_top_features: Optional[List[Dict]] = None

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def to_dict(self):
        return asdict(self)


class FreshnessPredictor:
    """
    Production predictor using XGBoost + FeatureBuilder.
    Loads model once, predicts many times.
    """

    SCORE_MAP = {"Fresh": 100, "Medium": 50, "Spoiled": 0}

    def __init__(self):
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_builder = None
        self._loaded = False
        self._load_model()

    def _load_model(self):
        """Load model and FeatureBuilder from disk."""
        model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_MODEL_PATH
        encoder_path = ENCODER_PATH if os.path.exists(ENCODER_PATH) else FALLBACK_ENCODER_PATH

        if not os.path.exists(model_path):
            logger.warning(f"No model found at {model_path}. Call train_model() first.")
            return

        try:
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")

            if os.path.exists(encoder_path):
                self.feature_builder = joblib.load(encoder_path)
                logger.info(f"FeatureBuilder loaded from {encoder_path}")
            else:
                logger.warning("FeatureBuilder not found — predictions may fail.")

            self._loaded = True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def predict(self, input_data: Dict) -> PredictionResult:
        """
        Predict food freshness from input data.

        Args:
            input_data: Dictionary with feature values:
                - storage_time (float)
                - time_since_cooking (float)
                - storage_condition (str)
                - container_type (str)
                - food_type (str)
                - moisture_type (str)
                - cooking_method (str)
                - texture (str)
                - smell (str)

        Returns:
            PredictionResult with label, score, confidence, probabilities
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Train or load a model first.")

        # Transform input
        X = self.feature_builder.transform_single(input_data)

        # Predict
        probabilities = self.model.predict_proba(X)[0]
        classes = self.feature_builder.classes

        # Get prediction
        pred_idx = np.argmax(probabilities)
        predicted_label = classes[pred_idx]
        confidence = float(probabilities[pred_idx]) * 100

        # Calculate freshness score (weighted sum)
        score = sum(self.SCORE_MAP.get(cls, 0) * prob for cls, prob in zip(classes, probabilities))
        freshness_score = int(round(score))

        # Build probability dict
        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}

        return PredictionResult(
            freshness_label=predicted_label,
            freshness_score=freshness_score,
            confidence=round(confidence, 1),
            probabilities=prob_dict,
        )

    def predict_batch(self, input_list: List[Dict]) -> List[PredictionResult]:
        """Predict on multiple samples."""
        return [self.predict(data) for data in input_list]

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# Singleton for Django views
_predictor_instance: Optional[FreshnessPredictor] = None


def get_predictor() -> FreshnessPredictor:
    """Get or create singleton predictor instance."""
    global _predictor_instance
    if _predictor_instance is None or not _predictor_instance.is_loaded:
        _predictor_instance = FreshnessPredictor()
    return _predictor_instance
