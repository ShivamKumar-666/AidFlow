"""
RAG Matching API views.
"""

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import permissions, status

from .matcher import match_donation_to_ngos, sync_all_ngos, sync_ngo_to_vector_store


class MatchDonationView(APIView):
    """POST /api/rag/match/ — Find best-fit NGOs for a donation."""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        data = request.data

        donation = {
            "food_name": data.get("food_name", ""),
            "food_type": data.get("food_type", "Vegetarian"),
            "quantity_kg": float(data.get("quantity_kg", 1.0)),
            "freshness_score": float(data.get("freshness_score", 50)),
            "storage_condition": data.get("storage_condition", "room_temperature"),
        }

        donor_lat = data.get("latitude")
        donor_lng = data.get("longitude")
        top_k = int(data.get("top_k", 5))

        try:
            matches = match_donation_to_ngos(
                donation=donation,
                top_k=top_k,
                donor_lat=float(donor_lat) if donor_lat else None,
                donor_lng=float(donor_lng) if donor_lng else None,
            )

            return Response({
                "success": True,
                "matches": matches,
                "count": len(matches),
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {"success": False, "error": "An error occurred during matching."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class SyncNGOsView(APIView):
    """POST /api/rag/sync/ — Sync all NGO profiles to Qdrant."""
    permission_classes = [permissions.IsAdminUser]

    def post(self, request):
        try:
            result = sync_all_ngos()
            return Response({
                "success": True,
                "result": result,
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"success": False, "error": "An error occurred while syncing NGOs."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class SyncSingleNGOView(APIView):
    """POST /api/rag/sync/<ngo_id>/ — Sync a single NGO to Qdrant."""
    permission_classes = [permissions.IsAdminUser]

    def post(self, request, ngo_id):
        try:
            success = sync_ngo_to_vector_store(ngo_id)
            return Response({
                "success": success,
                "ngo_id": ngo_id,
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"success": False, "error": "An error occurred while syncing the NGO."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
