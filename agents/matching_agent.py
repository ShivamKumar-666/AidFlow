"""
Matching Agent — Finds best-fit NGOs using RAG.
Third step in the pipeline.
"""

import logging
from datetime import datetime
from typing import Dict
from .state import DonationState

logger = logging.getLogger(__name__)


def matching_agent(state: DonationState) -> DonationState:
    """
    Matching Agent: Find best-fit NGOs for this donation.
    - Uses RAG semantic matching (Phase 3)
    - Re-ranks by distance, capacity, reliability
    - Returns ranked list of NGO candidates
    """
    logger.info("MATCH: Finding best-fit NGOs")

    trail_entry = {
        "agent": "matching",
        "action": "search_ngos",
        "timestamp": datetime.now().isoformat(),
    }

    try:
        from rag_service.matcher import match_donation_to_ngos

        donation = {
            "food_name": state.get("food_name", ""),
            "food_type": state.get("food_type", "Vegetarian"),
            "quantity_kg": state.get("quantity_kg", 1),
            "freshness_score": state.get("freshness_score", 50),
            "storage_condition": state.get("storage_condition", "room_temperature"),
        }

        matches = match_donation_to_ngos(
            donation=donation,
            top_k=5,
            donor_lat=state.get("latitude"),
            donor_lng=state.get("longitude"),
        )

        state["matched_ngos"] = matches
        state["current_ngo_index"] = 0

        if matches:
            state["assigned_ngo_id"] = matches[0]["ngo_id"]
            state["assigned_ngo_name"] = matches[0]["ngo_name"]
            trail_entry["result"] = {
                "found": len(matches),
                "top_ngo": matches[0]["ngo_name"],
                "top_score": matches[0]["combined_score"],
            }
            logger.info(f"MATCH: Found {len(matches)} NGOs, top={matches[0]['ngo_name']}")
        else:
            state["assigned_ngo_id"] = None
            state["assigned_ngo_name"] = ""
            trail_entry["result"] = {"found": 0}
            logger.warning("MATCH: No NGOs found")

    except Exception as e:
        logger.error(f"MATCH: RAG matching failed: {e}")
        state["matched_ngos"] = []
        state["assigned_ngo_id"] = None
        state["assigned_ngo_name"] = ""
        trail_entry["error"] = str(e)

    state["decision_trail"] = state.get("decision_trail", []) + [trail_entry]
    state["status"] = "match_complete"

    return state
