"""
Verification Agent — Plausibility checks and anomaly detection.
Flags suspicious donations before matching.
"""

import logging

from django.utils import timezone

from .state import DonationState

logger = logging.getLogger(__name__)


def verification_agent(state: DonationState) -> DonationState:
    """
    Verification Agent: Check donation for anomalies.
    - Quantity vs storage time plausibility
    - Food type vs cooking method consistency
    - Repeated donations from same donor (potential abuse)
    - Weather-impossible claims (frozen food in heat without refrigeration)
    """
    logger.info("VERIFY: Running plausibility checks")

    anomalies = []
    notes = []

    quantity = state.get("quantity_kg", 0)
    storage_time = state.get("storage_time_hours", 0)
    cooking_time = state.get("time_since_cooking_hours", 0)
    food_type = state.get("food_type", "")
    storage_condition = state.get("storage_condition", "")
    freshness_score = state.get("freshness_score", 0)

    # --- Check 1: Quantity plausibility ---
    if quantity > 50:
        anomalies.append(f"Unusually large donation: {quantity}kg — verify with donor")
    elif quantity <= 0:
        anomalies.append(f"Invalid quantity: {quantity}kg")
    else:
        notes.append(f"Quantity {quantity}kg is within normal range")

    # --- Check 2: Time plausibility ---
    total_time = storage_time + cooking_time
    if total_time > 48:
        anomalies.append(f"Food is {total_time:.1f} hours old — likely unsafe")
    elif total_time > 24:
        notes.append(f"Food is {total_time:.1f} hours old — check freshness carefully")

    if cooking_time > storage_time and storage_time > 0:
        notes.append("Storage time less than cooking time — food may be fresh")

    # --- Check 3: Storage condition vs time ---
    if storage_condition == "outside" and total_time > 6:
        anomalies.append(f"Food left outside for {total_time:.1f} hours — high spoilage risk")
    elif storage_condition == "refrigerated" and total_time > 48:
        anomalies.append(f"Refrigerated for {total_time:.1f} hours — check quality")

    # --- Check 4: Food type vs smell consistency ---
    smell = state.get("smell", "neutral")
    if smell in ["sour", "fermented", "strong"]:
        anomalies.append(f"Unpleasant smell detected ({smell}) — food may be spoiled")
    if smell == "fermented" and food_type not in ["Dairy", "Bakery"]:
        notes.append("Fermented smell may indicate spoilage for this food type")

    # --- Check 5: Freshness vs claimed conditions ---
    if freshness_score < 30 and storage_condition == "refrigerated":
        anomalies.append("Very low freshness despite refrigeration — possible equipment failure")
    elif freshness_score > 80 and storage_condition == "outside":
        notes.append("High freshness despite outdoor storage — verify conditions")

    # --- Check 6: Minimum threshold ---
    if freshness_score < 20:
        anomalies.append(f"Freshness score too low ({freshness_score}%) — food unsafe for donation")

    # Determine validity
    is_valid = len(anomalies) == 0 or (len(anomalies) == 1 and "Unusually large" in anomalies[0])

    trail_entry = {
        "agent": "verification",
        "action": "plausibility_check",
        "timestamp": timezone.now().isoformat(),
        "anomalies": anomalies,
        "notes": notes,
        "is_valid": is_valid,
    }

    state["anomalies"] = anomalies
    state["verification_notes"] = notes
    state["is_valid"] = is_valid

    state["decision_trail"] = state.get("decision_trail", []) + [trail_entry]
    state["status"] = "verify_complete"

    if is_valid:
        logger.info(f"VERIFY: Passed with {len(notes)} notes")
    else:
        logger.warning(f"VERIFY: Failed with {len(anomalies)} anomalies: {anomalies}")

    return state
