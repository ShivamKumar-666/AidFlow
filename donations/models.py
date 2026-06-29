from django.db import models
from django.conf import settings
from django.utils import timezone

class Donation(models.Model):
    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        CLAIMED = 'claimed', 'Claimed'
        PICKED_UP = 'picked_up', 'Picked Up'
        DELIVERED = 'delivered', 'Delivered'
        EXPIRED = 'expired', 'Expired'

    # Relationships
    donor = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="donations")
    recipient = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True, related_name="claimed_donations")

    # Basic Info
    food_name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    quantity_kg = models.FloatField()
    pickup_address = models.TextField()
    
    # ML Inputs (Critical for AI Prediction)
    storage_time_hours = models.FloatField(help_text="How long has it been stored?")
    time_since_cooking_hours = models.FloatField(help_text="Hours since preparation")
    storage_condition = models.CharField(max_length=50, default="room_temp")
    food_type = models.CharField(max_length=50, default="cooked_meal")
    
    # Sensory ML Inputs (now persisted for retraining)
    container_type = models.CharField(max_length=50, blank=True, default="closed")
    moisture_type = models.CharField(max_length=50, blank=True, default="moist")
    cooking_method = models.CharField(max_length=50, blank=True, default="boiled")
    texture = models.CharField(max_length=50, blank=True, default="firm")
    smell = models.CharField(max_length=50, blank=True, default="neutral")
    
    # AI Results (Auto-filled)
    freshness_score = models.FloatField(null=True, blank=True)
    freshness_label = models.CharField(max_length=50, blank=True)
    confidence = models.FloatField(null=True, blank=True, help_text="ML model confidence %")
    
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    claimed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status']),
            models.Index(fields=['donor']),
        ]

    def __str__(self):
        score_display = f"{self.freshness_score}%" if self.freshness_score is not None else "N/A"
        return f"{self.food_name} - {self.freshness_label} ({score_display})"


class AgentRun(models.Model):
    """
    Persists the full agent decision trail for auditability.
    One donation can have multiple agent runs (escalations).
    """
    donation = models.ForeignKey(Donation, on_delete=models.CASCADE, related_name="agent_runs", null=True, blank=True)

    # Pipeline results
    status = models.CharField(max_length=20, help_text="intake, verify, match, logistics, complete, failed")
    final_ngo_id = models.IntegerField(null=True, blank=True)
    final_ngo_name = models.CharField(max_length=255, blank=True, default="")

    # Decision trail (JSON array of agent actions)
    decision_trail = models.JSONField(default=list, blank=True)

    # Performance metrics
    ml_freshness_score = models.FloatField(null=True, blank=True)
    ml_confidence = models.FloatField(null=True, blank=True)
    anomalies_found = models.IntegerField(default=0)
    escalations = models.IntegerField(default=0)
    matched_ngos_count = models.IntegerField(default=0)

    # Timestamps
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    duration_seconds = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['donation']),
            models.Index(fields=['status']),
        ]

    def __str__(self):
        food_name = self.donation.food_name if self.donation else "N/A"
        return f"Run for {food_name} — {self.status}"