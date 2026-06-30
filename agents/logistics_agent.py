"""
Logistics Agent — Manages claim offers, timeouts, and autonomous escalation.
Fourth step — the key autonomy behavior.
"""

import logging

from django.utils import timezone

from .state import DonationState

logger = logging.getLogger(__name__)

# Simulated timeout for demo (in production, use Celery + Redis)
CLAIM_TIMEOUT_MINUTES = 10


def logistics_agent(state: DonationState) -> DonationState:
    """
    Logistics Agent: Send claim offer and handle escalation.
    - If NGO available: send claim offer
    - If no response (simulated): escalate to next NGO
    - This is the autonomous re-routing behavior
    """
    logger.info("LOGISTICS: Processing claim offer")

    matched_ngos = state.get("matched_ngos", [])
    current_index = state.get("current_ngo_index", 0)
    escalation_count = state.get("escalation_count", 0)
    max_escalations = state.get("max_escalations", 3)

    trail_entry = {
        "agent": "logistics",
        "action": "claim_process",
        "timestamp": timezone.now().isoformat(),
    }

    # Check if we have NGOs to try
    if not matched_ngos:
        state["claim_sent"] = False
        state["claim_accepted"] = False
        state["status"] = "failed"
        state["error_message"] = "No NGOs available for this donation"
        trail_entry["result"] = "no_ngos_available"
        state["decision_trail"] = state.get("decision_trail", []) + [trail_entry]
        return state

    if current_index >= len(matched_ngos):
        state["claim_sent"] = False
        state["claim_accepted"] = False
        state["status"] = "failed"
        state["error_message"] = "All NGOs exhausted — no claims accepted"
        trail_entry["result"] = "all_ngos_exhausted"
        state["decision_trail"] = state.get("decision_trail", []) + [trail_entry]
        return state

    # Current NGO candidate
    ngo = matched_ngos[current_index]
    ngo_id = ngo["ngo_id"]
    ngo_name = ngo["ngo_name"]

    # --- Simulate claim offer ---
    # In production: send notification via Celery task, wait for response
    # For demo: simulate acceptance based on reliability score
    # TODO: Replace with real NGO notification logic
    import random

    reliability = ngo.get("reliability_score", 80) / 100.0
    acceptance_probability = reliability * 0.8  # 80% base chance

    # Simulate: if escalation count > 0, NGO is less likely to accept (already tried others)
    if escalation_count > 0:
        acceptance_probability *= 0.9

    accepted = random.random() < acceptance_probability

    trail_entry["ngo_id"] = ngo_id
    trail_entry["ngo_name"] = ngo_name
    trail_entry["escalation_count"] = escalation_count
    trail_entry["simulated_acceptance"] = accepted

    if accepted:
        # Claim accepted!
        state["claim_sent"] = True
        state["claim_accepted"] = True
        state["assigned_ngo_id"] = ngo_id
        state["assigned_ngo_name"] = ngo_name
        state["needs_escalation"] = False
        state["status"] = "complete"
        trail_entry["action"] = "claim_accepted"
        logger.info(f"LOGISTICS: Claim accepted by {ngo_name}")

    else:
        # Claim not accepted — escalate
        state["claim_sent"] = True
        state["claim_accepted"] = False
        state["escalation_count"] = escalation_count + 1

        if state["escalation_count"] >= max_escalations:
            state["needs_escalation"] = False
            state["status"] = "failed"
            state["error_message"] = f"Max escalations ({max_escalations}) reached"
            trail_entry["action"] = "max_escalations_reached"
            logger.warning(f"LOGISTICS: Max escalations reached")
        else:
            state["needs_escalation"] = True
            state["current_ngo_index"] = current_index + 1
            next_ngo = matched_ngos[current_index + 1] if current_index + 1 < len(matched_ngos) else None
            state["assigned_ngo_id"] = next_ngo["ngo_id"] if next_ngo else None
            state["assigned_ngo_name"] = next_ngo["ngo_name"] if next_ngo else ""
            state["status"] = "escalating"
            trail_entry["action"] = "escalating"
            trail_entry["escalated_to"] = next_ngo["ngo_name"] if next_ngo else "none"
            logger.info(f"LOGISTICS: Escalating to next NGO ({state['assigned_ngo_name']})")

    state["decision_trail"] = state.get("decision_trail", []) + [trail_entry]
    return state
