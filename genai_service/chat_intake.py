"""
Chat Intake — Extract donation details from natural language.
Uses Groq's Llama 3.3 70B with JSON mode for structured extraction.
"""

import json
import logging

from .config import TEXT_MODEL, get_client

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are a food donation intake assistant. Extract structured donation data from the user's free-text description.

Return ONLY a valid JSON object with these fields:
{
    "food_name": "string - name of the food",
    "food_type": "one of: Vegetarian, Non-Vegetarian, Seafood, Dairy, Bakery",
    "description": "string - brief description of the food",
    "quantity_kg": number (estimate in kg),
    "container_type": "one of: closed, plastic, metal, open",
    "storage_condition": "one of: refrigerated, room_temperature, heated, outside",
    "time_since_cooking_hours": number (hours since prepared),
    "storage_time_hours": number (hours in storage, can be 0 if just cooked),
    "moisture_type": "one of: dry, semi-wet, wet",
    "cooking_method": "one of: boiled, fried, steamed, baked",
    "texture": "one of: firm, soft, crispy, moist, dry, soggy",
    "smell": "one of: neutral, slight, sour, strong, fermented",
    "confidence": number between 0.0 and 1.0
}

Rules:
- Infer as much as possible from the user's description
- If the user says "biryani" → food_type=Non-Vegetarian, cooking_method=fried/steamed
- If "steel containers" → container_type=metal
- If "fridge" or "refrigerator" → storage_condition=refrigerated
- If no time mentioned, estimate conservatively (assume cooked 2+ hours ago)
- Return ONLY the JSON, no explanation text"""


def extract_from_text(user_input: str) -> dict:
    """
    Extract donation details from free-text input.

    Args:
        user_input: Natural language description of the food donation

    Returns:
        dict with extracted fields matching Donation model
    """
    client = get_client()

    try:
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": user_input},
            ],
            temperature=0.1,
            max_tokens=512,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        return _validate_result(result)

    except Exception as e:
        logger.error(f"Chat extraction failed: {e}")
        return {"success": False, "error": str(e), "confidence": 0.0}


def extract_with_fallback(user_input: str) -> dict:
    """
    Try extraction, return result with low-confidence flag if uncertain.
    """
    result = extract_from_text(user_input)

    if result.get("success") and result.get("confidence", 0) >= 0.6:
        return result

    # Low confidence — flag for manual review
    result["needs_manual_review"] = True
    return result


def _validate_result(result: dict) -> dict:
    """Validate and normalize extracted fields."""
    valid_food_types = ["Vegetarian", "Non-Vegetarian", "Seafood", "Dairy", "Bakery"]
    valid_containers = ["closed", "plastic", "metal", "open"]
    valid_storage = ["refrigerated", "room_temperature", "heated", "outside"]
    valid_moisture = ["dry", "semi-wet", "wet"]
    valid_texture = ["firm", "soft", "crispy", "moist", "dry", "soggy"]
    valid_cooking = ["boiled", "fried", "steamed", "baked"]
    valid_smell = ["neutral", "slight", "sour", "strong", "fermented"]

    # Clamp numeric values
    result["quantity_kg"] = max(0.5, min(100, float(result.get("quantity_kg", 1.0))))
    result["time_since_cooking_hours"] = max(0, min(168, float(result.get("time_since_cooking_hours", 2.0))))
    result["storage_time_hours"] = max(0, min(168, float(result.get("storage_time_hours", 1.0))))
    result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.5))))

    # Validate enums with fallbacks
    _validate_enum(result, "food_type", valid_food_types, "Vegetarian")
    _validate_enum(result, "container_type", valid_containers, "closed")
    _validate_enum(result, "storage_condition", valid_storage, "room_temperature")
    _validate_enum(result, "moisture_type", valid_moisture, "dry")
    _validate_enum(result, "texture", valid_texture, "firm")
    _validate_enum(result, "cooking_method", valid_cooking, "boiled")
    _validate_enum(result, "smell", valid_smell, "neutral")

    result["success"] = True
    return result


def _validate_enum(result: dict, key: str, valid_values: list, default: str):
    """Validate a single enum field."""
    if result.get(key) not in valid_values:
        result[key] = default
