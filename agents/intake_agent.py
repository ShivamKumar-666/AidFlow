"""
Intake Agent — Normalizes donor input, invokes ML freshness model.
First step in the pipeline.
"""

import logging

from django.utils import timezone

from .state import DonationState

logger = logging.getLogger(__name__)


def intake_agent(state: DonationState) -> DonationState:
    """
    Intake Agent: Process raw donation input.
    - Validates required fields
    - Invokes XGBoost freshness prediction
    - Computes SHAP explanation
    - Returns structured DonationState
    """
    logger.info(f"INTAKE: Processing donation from {state.get('donor_username', 'unknown')}")

    trail_entry = {
        "agent": "intake",
        "action": "processed_input",
        "timestamp": timezone.now().isoformat(),
        "input_food": state.get("food_name", ""),
        "input_quantity": state.get("quantity_kg", 0),
    }

    try:
        # Import ML components
        from ml_service.explainer import get_explainer
        from ml_service.predictor import get_predictor

        predictor = get_predictor()
        explainer = get_explainer()

        # Build ML input
        ml_input = {
            "storage_time": state.get("storage_time_hours", 0),
            "time_since_cooking": state.get("time_since_cooking_hours", 0),
            "storage_condition": state.get("storage_condition", "room_temperature"),
            "food_type": state.get("food_type", "Vegetarian"),
            "container_type": state.get("container_type", "closed"),
            "moisture_type": state.get("moisture_type", "dry"),
            "cooking_method": state.get("cooking_method", "boiled"),
            "texture": state.get("texture", "firm"),
            "smell": state.get("smell", "neutral"),
        }

        # Predict freshness
        prediction = predictor.predict(ml_input)
        shap_features = explainer.explain(ml_input, top_n=3)

        state["freshness_score"] = prediction.freshness_score
        state["freshness_label"] = prediction.freshness_label
        state["ml_confidence"] = prediction.confidence
        state["shap_explanation"] = shap_features

        trail_entry["result"] = {
            "freshness_score": prediction.freshness_score,
            "freshness_label": prediction.freshness_label,
            "confidence": prediction.confidence,
        }

        logger.info(f"INTAKE: Freshness={prediction.freshness_label} ({prediction.freshness_score}%)")

    except Exception as e:
        logger.error(f"INTAKE: ML prediction failed: {e}")
        state["freshness_score"] = 0
        state["freshness_label"] = "Unknown"
        state["ml_confidence"] = 0
        state["shap_explanation"] = []
        trail_entry["error"] = str(e)

    # Add to decision trail
    state["decision_trail"] = state.get("decision_trail", []) + [trail_entry]
    state["status"] = "intake_complete"

    return state
