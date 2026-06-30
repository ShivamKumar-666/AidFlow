"""
NGO Embeddings — Embed NGO profiles into Qdrant vector store.
Uses HuggingFace all-MiniLM-L6-v2 for embeddings.
"""

import logging
import os
from typing import Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Config
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ngo_profiles"
VECTOR_DIMENSION = 384  # all-MiniLM-L6-v2 output dimension

_model: Optional[SentenceTransformer] = None
_client: Optional[QdrantClient] = None


def get_model() -> SentenceTransformer:
    """Get or load the embedding model (singleton)."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded")
    return _model


def get_client() -> QdrantClient:
    """Get or create Qdrant client (singleton)."""
    global _client
    if _client is not None:
        return _client

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        _client = QdrantClient(url=qdrant_url, timeout=5)
        # Quick health check
        _client.get_collections()
        logger.info(f"Connected to Qdrant at {qdrant_url}")
    except Exception as e:
        logger.warning(f"Qdrant not available ({e}). Using in-memory store.")
        _client = QdrantClient(":memory:")
    return _client


def ensure_collection():
    """Create the Qdrant collection if it doesn't exist."""
    client = get_client()
    collections = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_DIMENSION,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")


def embed_text(text: str) -> List[float]:
    """Convert text to embedding vector."""
    model = get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def upsert_ngo(ngo_id: int, capability_doc: str, metadata: Dict):
    """
    Embed and store an NGO profile in Qdrant.

    Args:
        ngo_id: Django user ID of the NGO
        capability_doc: Free-text capability description
        metadata: Additional metadata (lat, lng, capacity, etc.)
    """
    client = get_client()
    ensure_collection()

    embedding = embed_text(capability_doc)

    point = PointStruct(
        id=ngo_id,
        vector=embedding,
        payload={
            "ngo_id": ngo_id,
            "capability_document": capability_doc,
            **metadata,
        },
    )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[point],
    )
    logger.info(f"Embedded NGO {ngo_id}: {capability_doc[:80]}...")


def remove_ngo(ngo_id: int):
    """Remove an NGO from the vector store."""
    client = get_client()
    try:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=[ngo_id],
        )
        logger.info(f"Removed NGO {ngo_id} from vector store")
    except Exception as e:
        logger.warning(f"Failed to remove NGO {ngo_id}: {e}")


def search_similar_ngos(
    query_text: str,
    top_k: int = 10,
    filters: Optional[Dict] = None,
) -> List[Dict]:
    """
    Search for NGOs similar to a donation query.

    Args:
        query_text: Donation description (food type, quantity, urgency)
        top_k: Number of results to return
        filters: Optional Qdrant filters (e.g., {"role": "ngo"})

    Returns:
        List of {"ngo_id", "score", "payload"} dicts
    """
    client = get_client()
    ensure_collection()

    query_embedding = embed_text(query_text)

    # Build filter conditions
    query_filter = None
    if filters:
        conditions = []
        for key, value in filters.items():
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        if conditions:
            query_filter = Filter(must=conditions)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        query_filter=query_filter,
    )

    return [
        {
            "ngo_id": hit.id,
            "score": hit.score,
            "payload": hit.payload,
        }
        for hit in results.points
    ]
