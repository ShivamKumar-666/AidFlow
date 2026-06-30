"""
Agent API views — Trigger and monitor the agentic pipeline.
"""

import logging
from datetime import datetime

from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.shortcuts import render
from django.utils import timezone
from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from donations.models import AgentRun, Donation

from .orchestrator import run_pipeline

logger = logging.getLogger(__name__)


@login_required
def agent_dashboard_view(request):
    """Render the agent observability dashboard."""
    return render(request, "agent_dashboard.html")


class RunPipelineView(APIView):
    """POST /api/agents/run/ — Run the full agent pipeline on a donation."""

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        data = request.data
        donation_id = data.get("donation_id")

        # If donation_id provided, load from DB
        if donation_id:
            try:
                donation = Donation.objects.get(id=donation_id)
                donation_data = {
                    "donation_id": donation.id,
                    "donor_id": donation.donor_id,
                    "donor_username": donation.donor.username,
                    "food_name": donation.food_name,
                    "description": donation.description,
                    "quantity_kg": donation.quantity_kg,
                    "pickup_address": donation.pickup_address,
                    "latitude": getattr(donation.donor, "latitude", None),
                    "longitude": getattr(donation.donor, "longitude", None),
                    "storage_time_hours": donation.storage_time_hours,
                    "time_since_cooking_hours": donation.time_since_cooking_hours,
                    "storage_condition": donation.storage_condition,
                    "food_type": donation.food_type,
                    "container_type": donation.container_type,
                    "moisture_type": donation.moisture_type,
                    "cooking_method": donation.cooking_method,
                    "texture": donation.texture,
                    "smell": donation.smell,
                }
            except Donation.DoesNotExist:
                return Response(
                    {"success": False, "error": f"Donation {donation_id} not found"},
                    status=status.HTTP_404_NOT_FOUND,
                )
        else:
            # Use inline data (for testing)
            donation_data = data

        # Run pipeline
        try:
            final_state = run_pipeline(donation_data)

            # Persist to AgentRun
            started = datetime.fromisoformat(final_state["started_at"]) if final_state.get("started_at") else None
            completed = datetime.fromisoformat(final_state["completed_at"]) if final_state.get("completed_at") else None
            duration = None
            if started and completed:
                duration = (completed - started).total_seconds()

            agent_run = AgentRun.objects.create(
                donation_id=donation_id,
                status=final_state.get("status", "unknown"),
                final_ngo_id=final_state.get("assigned_ngo_id"),
                final_ngo_name=final_state.get("assigned_ngo_name", ""),
                decision_trail=final_state.get("decision_trail", []),
                ml_freshness_score=final_state.get("freshness_score"),
                ml_confidence=final_state.get("ml_confidence"),
                anomalies_found=len(final_state.get("anomalies", [])),
                escalations=final_state.get("escalation_count", 0),
                matched_ngos_count=len(final_state.get("matched_ngos", [])),
                started_at=started,
                completed_at=completed,
                duration_seconds=duration,
            )

            # Update donation status if accepted
            if final_state.get("claim_accepted") and donation_id:
                try:
                    with transaction.atomic():
                        donation = Donation.objects.select_for_update().get(id=donation_id, status="pending")
                        donation.status = "claimed"
                        donation.recipient_id = final_state.get("assigned_ngo_id")
                        donation.claimed_at = timezone.now()
                        donation.freshness_score = final_state.get("freshness_score", donation.freshness_score)
                        donation.freshness_label = final_state.get("freshness_label", donation.freshness_label)
                        donation.confidence = final_state.get("ml_confidence", donation.confidence)
                        donation.save()
                except Donation.DoesNotExist:
                    logger.warning(f"Donation {donation_id} already claimed or missing.")
                except Exception as e:
                    logger.error(f"Failed to update donation: {e}")

            return Response(
                {
                    "success": True,
                    "pipeline_status": final_state.get("status"),
                    "freshness_score": final_state.get("freshness_score"),
                    "freshness_label": final_state.get("freshness_label"),
                    "is_valid": final_state.get("is_valid"),
                    "anomalies": final_state.get("anomalies", []),
                    "assigned_ngo": final_state.get("assigned_ngo_name"),
                    "escalations": final_state.get("escalation_count", 0),
                    "decision_trail": final_state.get("decision_trail", []),
                    "agent_run_id": agent_run.id,
                },
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return Response(
                {"success": False, "error": "An internal error occurred while running the pipeline."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class AgentRunsView(APIView):
    """GET /api/agents/runs/ — List recent agent runs for observability."""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        runs = AgentRun.objects.select_related("donation").all()[:20]

        data = []
        for run in runs:
            data.append(
                {
                    "id": run.id,
                    "donation_id": run.donation_id,
                    "donation_name": run.donation.food_name if run.donation else "",
                    "status": run.status,
                    "final_ngo": run.final_ngo_name,
                    "ml_score": run.ml_freshness_score,
                    "anomalies": run.anomalies_found,
                    "escalations": run.escalations,
                    "matched_ngos": run.matched_ngos_count,
                    "duration_seconds": run.duration_seconds,
                    "started_at": run.started_at.isoformat() if run.started_at else "",
                    "decision_trail": run.decision_trail,
                }
            )

        return Response({"runs": data, "count": len(data)})
