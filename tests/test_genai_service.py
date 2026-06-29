"""
Tests for GenAI Service: Vision, Chat, Explainer.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestGenAIConfig:
    """Tests for GenAI configuration."""

    def test_config_import(self):
        from genai_service.config import get_client, VISION_MODEL, TEXT_MODEL
        assert VISION_MODEL == "meta-llama/llama-4-scout-17b-16e-instruct"
        assert TEXT_MODEL == "llama-3.3-70b-versatile"
        
        # Test getting client with a mock key to avoid ValueError
        with patch('os.getenv', return_value='fake_key'):
            client = get_client()
            assert client is not None


class TestVisionIntake:
    """Tests for Vision Intake service."""

    def test_vision_intake_import(self):
        from genai_service.vision_intake import extract_from_image
        assert extract_from_image is not None

    @patch('genai_service.vision_intake.get_client')
    def test_vision_intake_structure(self, mock_get_client):
        from genai_service.vision_intake import extract_from_image

        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='{"food_name": "Test", "food_type": "Vegetarian", "estimated_quantity_kg": 5, "container_type": "plastic", "moisture_type": "dry", "texture": "soft", "cooking_method": "boiled"}'))]
        mock_client.chat.completions.create.return_value = mock_response

        result = extract_from_image(b"fake_image_data")

        assert 'food_name' in result
        assert 'food_type' in result
        assert 'estimated_quantity_kg' in result


class TestChatIntake:
    """Tests for Chat Intake service."""

    def test_chat_intake_import(self):
        from genai_service.chat_intake import extract_from_text
        assert extract_from_text is not None

    @patch('genai_service.chat_intake.get_client')
    def test_chat_intake_structure(self, mock_get_client):
        from genai_service.chat_intake import extract_from_text

        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='{"food_name": "Test", "food_type": "Vegetarian", "estimated_quantity_kg": 5, "container_type": "plastic", "moisture_type": "dry", "texture": "soft", "cooking_method": "boiled"}'))]
        mock_client.chat.completions.create.return_value = mock_response

        result = extract_from_text("I have 5kg of fresh rice")

        assert 'food_name' in result
        assert 'food_type' in result
        assert 'estimated_quantity_kg' in result


class TestExplainerLLM:
    """Tests for SHAP Explainer LLM."""

    def test_explainer_llm_import(self):
        from genai_service.explainer_llm import explain_freshness
        assert explain_freshness is not None

    @patch('genai_service.explainer_llm.get_client')
    def test_explainer_llm_structure(self, mock_get_client):
        from genai_service.explainer_llm import explain_freshness

        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='The food is predicted fresh because it was stored in refrigerator.'))]
        mock_client.chat.completions.create.return_value = mock_response

        shap_features = [{'feature': 'storage_condition_refrigerated', 'impact': 0.5, 'direction': 'positive'}]
        result = explain_freshness('Fresh', 90, 95.5, shap_features)

        assert isinstance(result, str)
        assert len(result) > 0


class TestGenAIViews:
    """Tests for GenAI API views."""

    def test_vision_intake_view_import(self):
        from genai_service.views import VisionIntakeView
        assert VisionIntakeView is not None

    def test_chat_intake_view_import(self):
        from genai_service.views import ChatIntakeView
        assert ChatIntakeView is not None

    def test_shap_explanation_view_import(self):
        from genai_service.views import SHAPExplanationView
        assert SHAPExplanationView is not None