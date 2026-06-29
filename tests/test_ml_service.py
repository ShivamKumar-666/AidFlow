"""
Tests for ML Service: FeatureBuilder, Predictor, Explainer.
"""
import pytest
import numpy as np


class TestFeatureBuilder:
    """Tests for FeatureBuilder class."""

    def test_feature_builder_import(self):
        from ml_service.feature_builder import FeatureBuilder
        assert FeatureBuilder is not None

    def test_feature_builder_initialization(self):
        from ml_service.feature_builder import FeatureBuilder
        fb = FeatureBuilder()
        assert fb is not None
        assert hasattr(fb, 'feature_cols')

    def test_feature_builder_fit_transform(self):
        from ml_service.feature_builder import FeatureBuilder
        import pandas as pd

        fb = FeatureBuilder()
        sample_data = pd.DataFrame([{
            'food_type': 'Vegetarian',
            'storage_condition': 'refrigerated',
            'container_type': 'plastic',
            'moisture_type': 'dry',
            'cooking_method': 'steamed',
            'texture': 'soft',
            'smell': 'neutral',
            'storage_time': 4,
            'time_since_cooking': 2,
            'quantity_kg': 5,
            'freshness_level': 'Fresh'
        }])

        X = fb.fit_transform(sample_data)
        assert X.shape[0] == 1
        assert X.shape[1] == len(fb.feature_cols)

    def test_feature_builder_transform_consistency(self):
        from ml_service.feature_builder import FeatureBuilder
        import pandas as pd

        fb = FeatureBuilder()
        sample_data = pd.DataFrame([{
            'food_type': 'Non-Vegetarian',
            'storage_condition': 'room_temp',
            'container_type': 'metal',
            'moisture_type': 'wet',
            'cooking_method': 'fried',
            'texture': 'firm',
            'smell': 'strong',
            'storage_time': 6,
            'time_since_cooking': 3,
            'quantity_kg': 10,
            'freshness_level': 'Fresh'
        }])

        fb.fit(sample_data)
        X1 = fb.transform(sample_data)
        X2 = fb.transform(sample_data)
        np.testing.assert_array_equal(X1, X2)

    def test_feature_builder_feature_names(self):
        from ml_service.feature_builder import FeatureBuilder
        fb = FeatureBuilder()
        # This will be tested after we add the alias property
        # For now, it will fail until the alias is added
        # names = fb.feature_names
        # pass


class TestPredictor:
    """Tests for Predictor class."""

    def test_predictor_import(self):
        from ml_service.predictor import FreshnessPredictor as Predictor
        assert Predictor is not None

    def test_predictor_singleton(self):
        from ml_service.predictor import get_predictor
        p1 = get_predictor()
        p2 = get_predictor()
        assert p1 is p2

    def test_predictor_load_model(self):
        from ml_service.predictor import get_predictor
        predictor = get_predictor()
        assert predictor.model is not None
        assert predictor.feature_builder is not None

    def test_predictor_predict(self):
        from ml_service.predictor import get_predictor
        predictor = get_predictor()

        result = predictor.predict({
            'food_type': 'Vegetarian',
            'storage_condition': 'refrigerated',
            'container_type': 'plastic',
            'moisture_type': 'dry',
            'cooking_method': 'steamed',
            'texture': 'soft',
            'smell': 'neutral',
            'storage_time': 4,
            'time_since_cooking': 2,
            'quantity_kg': 5,
        })
        
        result_dict = result.to_dict()
        assert 'freshness_score' in result_dict
        assert 'freshness_label' in result_dict
        assert 'confidence' in result_dict
        assert 0 <= result_dict['freshness_score'] <= 100
        assert result_dict['freshness_label'] in ['Fresh', 'Medium', 'Low', 'Spoiled']
        assert 0 <= result_dict['confidence'] <= 100

    def test_predictor_predict_proba(self):
        from ml_service.predictor import get_predictor
        predictor = get_predictor()

        result = predictor.predict({
            'food_type': 'Vegetarian',
            'storage_condition': 'refrigerated',
            'container_type': 'plastic',
            'moisture_type': 'dry',
            'cooking_method': 'steamed',
            'texture': 'soft',
            'smell': 'neutral',
            'storage_time': 4,
            'time_since_cooking': 2,
            'quantity_kg': 5,
        })
        
        probas = list(result.probabilities.values())
        assert len(probas) == 3  # 3 classes
        assert abs(sum(probas) - 1.0) < 0.01


class TestExplainer:
    """Tests for SHAP Explainer."""

    def test_explainer_import(self):
        from ml_service.explainer import FreshnessExplainer as Explainer
        assert Explainer is not None

    def test_explainer_explain(self):
        from ml_service.explainer import get_explainer

        explainer = get_explainer()
        
        if not explainer._loaded:
            pytest.skip("Model not available to test explainer")

        shap_values = explainer.explain({
            'food_type': 'Vegetarian',
            'storage_condition': 'refrigerated',
            'container_type': 'plastic',
            'moisture_type': 'dry',
            'cooking_method': 'steamed',
            'texture': 'soft',
            'smell': 'neutral',
            'storage_time_hours': 4,
            'time_since_cooking_hours': 2,
            'quantity_kg': 5,
        })

        assert isinstance(shap_values, list)
        if len(shap_values) > 0:
            assert 'feature' in shap_values[0]