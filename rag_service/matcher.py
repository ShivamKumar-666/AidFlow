"""
NGO Matcher — Semantic search + geographic + capacity re-ranking.
Combines RAG retrieval with business logic for optimal matching.
"""

import logging
import math
from typing import Dict, List, Optional
from .ngo_embeddings import search_similar_ngos, upsert_ngo, ensure_collection

logger = logging.getLogger(__name__)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in km using Haversine formula."""
    R = 6371  # Earth's radius in km

    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def build_donation_query(donation: Dict) -> str:
    """
    Build a natural language query from donation metadata.
    This query is embedded and searched against NGO profiles.
    """
    parts = []

    if donation.get("food_type"):
        parts.append(f"Food type: {donation['food_type']}")

    if donation.get("quantity_kg"):
        qty = float(donation["quantity_kg"])
        if qty > 20:
            parts.append("Large quantity donation")
        elif qty > 5:
            parts.append("Medium quantity donation")
        else:
            parts.append("Small quantity donation")

    if donation.get("freshness_score") is not None:
        score = float(donation["freshness_score"])
        if score >= 70:
            parts.append("Fresh food, can be distributed immediately")
        elif score >= 40:
            parts.append("Moderate freshness, should be consumed soon")
        else:
            parts.append("Low freshness, urgent pickup needed")

    if donation.get("storage_condition"):
        parts.append(f"Storage: {donation['storage_condition']}")

    if donation.get("food_name"):
        parts.append(f"Item: {donation['food_name']}")

    return " | ".join(parts) if parts else "Food donation available"


def match_donation_to_ngos(
    donation: Dict,
    top_k: int = 5,
    max_distance_km: float = 50.0,
    donor_lat: Optional[float] = None,
    donor_lng: Optional[float] = None,
) -> List[Dict]:
    """
    Match a donation to the best-fit NGOs using RAG + re-ranking.

    Steps:
    1. Build semantic query from donation metadata
    2. Retrieve top-k similar NGOs from Qdrant
    3. Re-rank by: semantic score + distance + capacity + reliability

    Args:
        donation: Donation metadata dict
        top_k: Number of NGOs to return
        max_distance_km: Maximum pickup distance
        donor_lat: Donor latitude for distance calculation
        donor_lng: Donor longitude for distance calculation

    Returns:
        Ranked list of NGO matches with scores
    """
    # Step 1: Build query
    query = build_donation_query(donation)
    logger.info(f"Matching query: {query}")

    # Step 2: Semantic search
    candidates = search_similar_ngos(
        query_text=query,
        top_k=top_k * 3,  # Get extra for re-ranking
        filters={"role": "ngo"},
    )

    if not candidates:
        logger.info("No NGOs found in vector store")
        return []

    # Step 3: Re-rank
    matched_ngos = []
    for candidate in candidates:
        ngo_id = candidate["ngo_id"]
        semantic_score = candidate["score"]  # 0-1 cosine similarity

        try:
            from django.contrib.auth import get_user_model
            User = get_user_model()
            ngo = User.objects.get(id=ngo_id, role__in=["ngo", "shelter"])
        except Exception: # Catch Exception since User.DoesNotExist might not be defined if get_user_model fails
            continue

        # Distance score
        distance_km = 0.0
        distance_score = 1.0
        if donor_lat and donor_lng and ngo.latitude and ngo.longitude:
            distance_km = haversine_distance(donor_lat, donor_lng, ngo.latitude, ngo.longitude)
            if distance_km > max_distance_km:
                continue  # Skip NGOs too far away
            # Distance score: closer = higher (1.0 at 0km, ~0 at max_distance)
            distance_score = max(0, 1.0 - (distance_km / max_distance_km))

        # Capacity score
        capacity_score = 1.0
        donation_qty = float(donation.get("quantity_kg", 0))
        if hasattr(ngo, 'capacity_kg') and ngo.capacity_kg and donation_qty:
            if donation_qty <= ngo.capacity_kg:
                capacity_score = 1.0  # Fits within capacity
            else:
                capacity_score = max(0.3, ngo.capacity_kg / donation_qty)

        # Reliability score (0-1)
        reliability_score = getattr(ngo, 'reliability_score', 80) / 100.0

        # Combined score (weighted)
        combined_score = (
            0.40 * semantic_score +      # How well NGO capabilities match
            0.25 * distance_score +       # How close the NGO is
            0.15 * capacity_score +       # Whether NGO can handle the quantity
            0.20 * reliability_score      # Past reliability
        )

        matched_ngos.append({
            "ngo_id": ngo_id,
            "ngo_name": ngo.username,
            "ngo_address": ngo.address or "",
            "ngo_phone": ngo.phone or "",
            "latitude": ngo.latitude,
            "longitude": ngo.longitude,
            "capacity_kg": ngo.capacity_kg,
            "reliability_score": ngo.reliability_score,
            "total_collections": ngo.total_collections,
            "semantic_score": round(semantic_score, 4),
            "distance_km": round(distance_km, 1),
            "distance_score": round(distance_score, 4),
            "capacity_score": round(capacity_score, 4),
            "combined_score": round(combined_score, 4),
        })

    # Sort by combined score descending
    matched_ngos.sort(key=lambda x: x["combined_score"], reverse=True)

    return matched_ngos[:top_k]


def sync_ngo_to_vector_store(ngo_id: int) -> bool:
    """
    Sync a single NGO's profile to Qdrant.
    Returns True if successful.
    """
    try:
        ngo = User.objects.get(id=ngo_id, role__in=["ngo", "shelter"])
    except User.DoesNotExist:
        logger.warning(f"NGO {ngo_id} not found")
        return False

    # Generate or use existing capability document
    if not ngo.capability_document:
        ngo.generate_capability_document()
        ngo.save(update_fields=["capability_document"])

    # Build metadata
    metadata = {
        "username": ngo.username,
        "role": ngo.role,
        "latitude": ngo.latitude,
        "longitude": ngo.longitude,
        "capacity_kg": ngo.capacity_kg,
        "dietary_restrictions": ngo.dietary_restrictions,
        "cultural_rules": ngo.cultural_rules,
        "operating_hours": ngo.operating_hours,
    }

    try:
        upsert_ngo(
            ngo_id=ngo.id,
            capability_doc=ngo.capability_document,
            metadata=metadata,
        )
        ngo.is_embedded = True
        ngo.save(update_fields=["is_embedded"])
        return True
    except Exception as e:
        logger.error(f"Failed to embed NGO {ngo_id}: {e}")
        return False


def sync_all_ngos() -> Dict[str, int]:
    """Sync all NGO profiles to Qdrant. Returns counts."""
    ngos = User.objects.filter(role__in=["ngo", "shelter"])

    success = 0
    failed = 0

    for ngo in ngos:
        if sync_ngo_to_vector_store(ngo.id):
            success += 1
        else:
            failed += 1

    logger.info(f"Synced {success} NGOs, {failed} failed")
    return {"success": success, "failed": failed}
