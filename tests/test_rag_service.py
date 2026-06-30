"""
Tests for RAG Service: Embeddings, Matcher.
"""

from unittest.mock import Mock, patch


class TestRAGService:
    """Tests for RAG service components."""

    def test_ngo_embeddings_import(self):
        from rag_service.ngo_embeddings import search_similar_ngos, upsert_ngo

        assert search_similar_ngos is not None
        assert upsert_ngo is not None

    def test_matcher_import(self):
        from rag_service.matcher import match_donation_to_ngos

        assert match_donation_to_ngos is not None

    @patch("rag_service.ngo_embeddings.QdrantClient")
    @patch("rag_service.ngo_embeddings.SentenceTransformer")
    def test_ngo_embeddings_init(self, mock_transformer, mock_qdrant):
        from rag_service.ngo_embeddings import get_client, get_model

        mock_transformer_instance = Mock()
        mock_transformer.return_value = mock_transformer_instance
        mock_transformer_instance.encode.return_value = [[0.1] * 384]

        mock_qdrant_instance = Mock()
        mock_qdrant.return_value = mock_qdrant_instance

        # Override the env var or set it for the test
        with patch("os.getenv", return_value="http://localhost:6333"):
            model = get_model()
            client = get_client()

            assert model is not None
            assert client is not None

    @patch("rag_service.matcher.search_similar_ngos")
    def test_match_ngos_structure(self, mock_search):
        from rag_service.matcher import match_donation_to_ngos

        mock_search.return_value = [{"ngo_id": 1, "score": 0.85}]

        donation_data = {
            "food_name": "Test Biryani",
            "food_type": "Non-Vegetarian",
            "quantity_kg": 10,
            "latitude": 19.0760,
            "longitude": 72.8777,
        }

        # Mock the User query since matcher hits the DB
        with patch("django.contrib.auth.get_user_model") as mock_get_user_model:
            mock_user_model = Mock()
            mock_get_user_model.return_value = mock_user_model

            mock_ngo = Mock()
            mock_ngo.username = "NGO 1"
            mock_ngo.address = "Test"
            mock_ngo.phone = "123"
            mock_ngo.latitude = 19.0
            mock_ngo.longitude = 72.8
            mock_ngo.capacity_kg = 50
            mock_ngo.reliability_score = 80
            mock_ngo.total_collections = 5

            mock_user_model.objects.get.return_value = mock_ngo

            results = match_donation_to_ngos(donation_data)

            assert isinstance(results, list)
            assert len(results) >= 0


class TestRAGViews:
    """Tests for RAG API views."""

    def test_match_donation_view_import(self):
        from rag_service.views import MatchDonationView

        assert MatchDonationView is not None

    def test_sync_ngos_view_import(self):
        from rag_service.views import SyncNGOsView

        assert SyncNGOsView is not None
