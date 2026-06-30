"""
Orchestrator — LangGraph state machine wiring all agents.
Handles conditional routing and autonomous escalation.
"""

import logging
from typing import Literal

from django.utils import timezone
from langgraph.graph import END, StateGraph

from .intake_agent import intake_agent
from .logistics_agent import logistics_agent
from .matching_agent import matching_agent
from .state import DonationState
from .verification_agent import verification_agent

logger = logging.getLogger(__name__)


def route_after_verify(state: DonationState) -> Literal["match", "end"]:
    """Route: if valid → match, if invalid → end."""
    if state.get("is_valid", False):
        return "match"
    return "end"


def route_after_logistics(state: DonationState) -> Literal["logistics", "match", "end"]:
    """
    Route: if needs_escalation → logistics (re-try with next NGO)
           if complete → end
           if failed → end
    """
    if state.get("needs_escalation", False):
        return "logistics"
    return "end"


def build_graph() -> StateGraph:
    """
    Build the LangGraph agent pipeline:

    intake → verify → [match → logistics] → end
                     ↗                          ↑
                     (invalid)            (escalate loops back)
    """
    graph = StateGraph(DonationState)

    # Add nodes
    graph.add_node("intake", intake_agent)
    graph.add_node("verify", verification_agent)
    graph.add_node("match", matching_agent)
    graph.add_node("logistics", logistics_agent)

    # Set entry point
    graph.set_entry_point("intake")

    # Wire edges
    graph.add_edge("intake", "verify")

    # Conditional: verify → match or end
    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "match": "match",
            "end": END,
        },
    )

    # match → logistics (always)
    graph.add_edge("match", "logistics")

    # Conditional: logistics → end or escalate back to logistics
    graph.add_conditional_edges(
        "logistics",
        route_after_logistics,
        {
            "logistics": "logistics",  # Self-loop for escalation
            "end": END,
        },
    )

    return graph


# Compile the graph (singleton)
_compiled_graph = None


def get_graph():
    """Get or compile the LangGraph pipeline."""
    global _compiled_graph
    if _compiled_graph is None:
        graph = build_graph()
        _compiled_graph = graph.compile()
        logger.info("LangGraph pipeline compiled")
    return _compiled_graph


def run_pipeline(donation_data: dict) -> dict:
    """
    Run the full agent pipeline on a donation.

    Args:
        donation_data: Dict with all donation fields

    Returns:
        Final state dict with results and decision trail
    """
    # Build initial state
    initial_state: DonationState = {
        # Input
        "donation_id": donation_data.get("donation_id"),
        "donor_id": donation_data.get("donor_id"),
        "donor_username": donation_data.get("donor_username", ""),
        "food_name": donation_data.get("food_name", ""),
        "description": donation_data.get("description", ""),
        "quantity_kg": float(donation_data.get("quantity_kg", 0)),
        "pickup_address": donation_data.get("pickup_address", ""),
        "latitude": donation_data.get("latitude"),
        "longitude": donation_data.get("longitude"),
        # ML inputs
        "storage_time_hours": float(donation_data.get("storage_time_hours", 0)),
        "time_since_cooking_hours": float(donation_data.get("time_since_cooking_hours", 0)),
        "storage_condition": donation_data.get("storage_condition", "room_temperature"),
        "food_type": donation_data.get("food_type", "Vegetarian"),
        "container_type": donation_data.get("container_type", "closed"),
        "moisture_type": donation_data.get("moisture_type", "dry"),
        "cooking_method": donation_data.get("cooking_method", "boiled"),
        "texture": donation_data.get("texture", "firm"),
        "smell": donation_data.get("smell", "neutral"),
        # ML results (set by intake agent)
        "freshness_score": 0,
        "freshness_label": "",
        "ml_confidence": 0,
        "shap_explanation": [],
        # Verification (set by verification agent)
        "is_valid": False,
        "verification_notes": [],
        "anomalies": [],
        # Matching (set by matching agent)
        "matched_ngos": [],
        "current_ngo_index": 0,
        "assigned_ngo_id": None,
        "assigned_ngo_name": "",
        # Logistics
        "claim_sent": False,
        "claim_accepted": False,
        "escalation_count": 0,
        "max_escalations": 3,
        "needs_escalation": False,
        # Meta
        "status": "starting",
        "error_message": "",
        "decision_trail": [],
        "started_at": timezone.now().isoformat(),
        "completed_at": "",
    }

    # Run pipeline
    logger.info(f"PIPELINE: Starting for {initial_state['food_name']}")
    graph = get_graph()
    final_state = graph.invoke(initial_state)

    # Set completion time
    final_state["completed_at"] = timezone.now().isoformat()

    logger.info(f"PIPELINE: Completed — status={final_state['status']}")
    return final_state
