"""
GenAI API views — Vision intake, Chat intake, SHAP explanation.
"""

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import permissions, status
from rest_framework.parsers import MultiPartParser, FormParser

from .vision_intake import extract_from_image
from .chat_intake import extract_from_text
from ml_service.explainer import get_explainer


class VisionIntakeView(APIView):
    """POST /api/genai/vision/ — Upload food photo, get structured data."""
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        image_file = request.FILES.get("image")
        if not image_file:
            return Response(
                {"success": False, "error": "No image file provided"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Basic security checks
        MAX_FILE_SIZE = 10 * 1024 * 1024 # 10 MB
        if image_file.size > MAX_FILE_SIZE:
             return Response(
                {"success": False, "error": "File size exceeds 10MB limit"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        
        valid_mime_types = ['image/jpeg', 'image/png', 'image/webp']
        if image_file.content_type not in valid_mime_types:
             return Response(
                {"success": False, "error": f"Invalid file type. Allowed: {', '.join(valid_mime_types)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            image_bytes = image_file.read()
            result = extract_from_image(image_bytes, image_file.name)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"success": False, "error": "An error occurred while processing the image."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ChatIntakeView(APIView):
    """POST /api/genai/chat/ — Free text → structured donation data."""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        text = request.data.get("text", "").strip()
        if not text:
            return Response(
                {"success": False, "error": "No text provided"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            result = extract_from_text(text)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"success": False, "error": "An error occurred while processing the text."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class SHAPExplanationView(APIView):
    """POST /api/genai/explain/ — Get LLM-powered SHAP explanation."""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        data = request.data
        freshness_label = data.get("freshness_label", "Unknown")
        freshness_score = int(data.get("freshness_score", 0))
        confidence = float(data.get("confidence", 0))

        # Get SHAP features from the ML explainer
        ml_input = {
            "storage_time": float(data.get("storage_time", 5)),
            "time_since_cooking": float(data.get("time_since_cooking", 3)),
            "storage_condition": data.get("storage_condition", "room_temperature"),
            "container_type": data.get("container_type", "closed"),
            "food_type": data.get("food_type", "Vegetarian"),
            "moisture_type": data.get("moisture_type", "dry"),
            "cooking_method": data.get("cooking_method", "boiled"),
            "texture": data.get("texture", "firm"),
            "smell": data.get("smell", "neutral"),
        }

        explainer = get_explainer()
        shap_features = explainer.explain(ml_input, top_n=3)

        try:
            from .explainer_llm import explain_freshness
            explanation = explain_freshness(
                freshness_label=freshness_label,
                freshness_score=freshness_score,
                confidence=confidence,
                shap_features=shap_features,
            )
            return Response({
                "success": True,
                "explanation": explanation,
                "shap_features": shap_features,
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"success": False, "error": "An error occurred while generating the explanation."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
