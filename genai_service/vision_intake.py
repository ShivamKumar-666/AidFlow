"""
Vision Intake — Extract donation details from a food photo.
Uses Groq's Llama 4 Scout vision model.
"""

import base64
import logging

from .config import VISION_MODEL, get_client

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Analyze this food photo and extract donation details as JSON.

Return ONLY a valid JSON object with these fields:
{
    "food_name": "string - name of the food item (e.g. 'Chicken Biryani')",
    "food_type": "one of: Vegetarian, Non-Vegetarian, Seafood, Dairy, Bakery",
    "container_type": "one of: closed, plastic, metal, open",
    "estimated_quantity_kg": number (estimate in kg, 0.5 to 100),
    "moisture_type": "one of: dry, semi-wet, wet",
    "texture": "one of: firm, soft, crispy, moist, dry, soggy",
    "visual_freshness": "one of: fresh, moderate, spoiled",
    "cooking_method": "one of: boiled, fried, steamed, baked",
    "confidence": number between 0.0 and 1.0
}

Rules:
- Estimate quantity based on visible portion size relative to the container
- If unsure about any field, use the most likely value and lower confidence
- Focus on what you CAN see — don't guess wildly
- Return ONLY the JSON object, no explanation text"""


def extract_from_image(image_bytes: bytes, filename: str = "") -> dict:
    """
    Extract donation details from a food photo.

    Args:
        image_bytes: Raw image bytes (JPEG, PNG, WebP)
        filename: Original filename (used for MIME type detection)

    Returns:
        dict with extracted fields + confidence score
    """
    client = get_client()

    # Determine MIME type
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else "jpeg"
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
    mime_type = mime_map.get(ext, "image/jpeg")

    # Encode to base64
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64_image}"

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": EXTRACTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=512,
            response_format={"type": "json_object"},
        )

        import json

        result = json.loads(response.choices[0].message.content)

        # Validate and normalize
        return _validate_result(result)

    except Exception as e:
        logger.error(f"Vision extraction failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "confidence": 0.0,
        }


def extract_from_url(image_url: str) -> dict:
    """
    Extract donation details from an image URL.
    Groq can fetch images from public URLs directly.
    """
    client = get_client()

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": EXTRACTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=512,
            response_format={"type": "json_object"},
        )

        import json

        result = json.loads(response.choices[0].message.content)
        return _validate_result(result)

    except Exception as e:
        logger.error(f"Vision extraction from URL failed: {e}")
        return {"success": False, "error": str(e), "confidence": 0.0}


def _validate_result(result: dict) -> dict:
    """Validate and normalize extracted fields."""
    valid_food_types = ["Vegetarian", "Non-Vegetarian", "Seafood", "Dairy", "Bakery"]
    valid_containers = ["closed", "plastic", "metal", "open"]
    valid_moisture = ["dry", "semi-wet", "wet"]
    valid_texture = ["firm", "soft", "crispy", "moist", "dry", "soggy"]
    valid_cooking = ["boiled", "fried", "steamed", "baked"]

    # Clamp values
    result["estimated_quantity_kg"] = max(0.5, min(100, float(result.get("estimated_quantity_kg", 1.0))))
    result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.5))))

    # Validate enums
    if result.get("food_type") not in valid_food_types:
        result["food_type"] = "Vegetarian"
    if result.get("container_type") not in valid_containers:
        result["container_type"] = "closed"
    if result.get("moisture_type") not in valid_moisture:
        result["moisture_type"] = "dry"
    if result.get("texture") not in valid_texture:
        result["texture"] = "firm"
    if result.get("cooking_method") not in valid_cooking:
        result["cooking_method"] = "boiled"

    result["success"] = True
    return result
