"""
Shared feature engineering for training and prediction.
Eliminates train/serve skew — both trainer and predictor use this exact class.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Central configuration for all features."""
    CATEGORICAL_FEATURES: List[str] = None
    NUMERICAL_FEATURES: List[str] = None
    TARGET_COL: str = 'freshness_level'

    def __post_init__(self):
        if self.CATEGORICAL_FEATURES is None:
            self.CATEGORICAL_FEATURES = [
                'storage_condition', 'container_type', 'food_type',
                'moisture_type', 'cooking_method', 'texture', 'smell'
            ]
        if self.NUMERICAL_FEATURES is None:
            self.NUMERICAL_FEATURES = ['storage_time', 'time_since_cooking']


CONFIG = FeatureConfig()


class FeatureBuilder:
    """
    Transforms raw input data into model-ready feature vectors.
    Used by both trainer (fit + transform) and predictor (transform only).
    """

    def __init__(self):
        self.label_encoders: Dict = {}
        self.target_encoder = None
        self.feature_cols: List[str] = []
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> 'FeatureBuilder':
        """Fit label encoders on training data."""
        from sklearn.preprocessing import LabelEncoder

        for col in CONFIG.CATEGORICAL_FEATURES:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.label_encoders[col] = le

        # Target encoder
        self.target_encoder = LabelEncoder()
        self.target_encoder.fit(df[CONFIG.TARGET_COL].astype(str))

        # Build feature column names
        encoded_cols = [f'{c}_encoded' for c in CONFIG.CATEGORICAL_FEATURES]
        self.feature_cols = CONFIG.NUMERICAL_FEATURES + encoded_cols

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe into encoded feature matrix."""
        if not self._fitted:
            raise RuntimeError("FeatureBuilder must be fitted before transform. Call fit() first.")

        result = pd.DataFrame()

        # Numerical features
        for col in CONFIG.NUMERICAL_FEATURES:
            result[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Categorical features — encode
        for col in CONFIG.CATEGORICAL_FEATURES:
            le = self.label_encoders.get(col)
            if le is None:
                raise RuntimeError(f"Encoder not found for {col}")

            values = df[col].astype(str)
            # Handle unseen values
            known = set(le.classes_)
            encoded = values.apply(lambda x: le.transform([x])[0] if x in known else 0)
            result[f'{col}_encoded'] = encoded

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def transform_single(self, input_data: Dict) -> pd.DataFrame:
        """Transform a single input dictionary for prediction."""
        df = pd.DataFrame([input_data])
        return self.transform(df)

    def encode_target(self, values: pd.Series) -> np.ndarray:
        """Encode target variable."""
        if self.target_encoder is None:
            raise RuntimeError("Target encoder not fitted.")
        return self.target_encoder.transform(values.astype(str))

    @property
    def classes(self) -> np.ndarray:
        """Return target classes."""
        if self.target_encoder is None:
            return np.array(['Fresh', 'Medium', 'Spoiled'])
        return self.target_encoder.classes_
        
    @property
    def feature_names(self) -> List[str]:
        """Alias for feature_cols to maintain backward compatibility."""
        return self.feature_cols
        
    def get_feature_names(self) -> List[str]:
        """Alias for feature_cols to maintain backward compatibility."""
        return self.feature_cols
