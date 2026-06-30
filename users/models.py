from django.contrib.auth.models import AbstractUser
from django.db import models


class CustomUser(AbstractUser):
    class UserRole(models.TextChoices):
        DONOR = "donor", "Donor"
        NGO = "ngo", "NGO"
        SHELTER = "shelter", "Shelter"
        RESTAURANT = "restaurant", "Restaurant"

    role = models.CharField(
        max_length=20,
        choices=UserRole.choices,
        default=UserRole.DONOR,
        help_text="The primary role of this user in the ecosystem.",
    )

    # Contact & Profile Details
    phone = models.CharField(max_length=15, blank=True, null=True)
    address = models.TextField(blank=True, null=True)

    # For NGOs/Shelters: How much food can they accept?
    capacity_kg = models.IntegerField(blank=True, null=True, help_text="Max storage capacity in KG (for Shelters/NGOs)")

    # Geo-Location
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)

    # --- NGO Capability Fields (for RAG matching) ---

    # Dietary restrictions they serve (e.g. "vegetarian only, halal, no pork")
    dietary_restrictions = models.CharField(
        max_length=500,
        blank=True,
        default="",
        help_text="Comma-separated dietary rules: vegetarian, halal, jain, no-pork, etc.",
    )

    # Cultural/religious rules
    cultural_rules = models.CharField(
        max_length=500,
        blank=True,
        default="",
        help_text="Cultural/religious constraints: no-beef, halal-only, kosher, etc.",
    )

    # Operating hours (JSON string or simple text)
    operating_hours = models.CharField(
        max_length=200, blank=True, default="8:00-20:00", help_text="Operating hours: e.g. 8:00-20:00 or 24/7"
    )

    # Past reliability score (0-100, updated by system)
    reliability_score = models.FloatField(
        default=80.0, help_text="Past pickup reliability (0-100). Updated by logistics agent."
    )

    # Total donations successfully collected
    total_collections = models.IntegerField(default=0)

    # Free-text capability description (for embedding)
    capability_document = models.TextField(
        blank=True,
        default="",
        help_text="Free-text NGO profile for semantic matching. Auto-generated from fields above or written manually.",
    )

    # Whether the NGO profile has been embedded in Qdrant
    is_embedded = models.BooleanField(default=False)

    class Meta:
        indexes = [
            models.Index(fields=["role"]),
            models.Index(fields=["is_embedded"]),
        ]

    def __str__(self):
        return f"{self.username} ({self.get_role_display()})"

    def generate_capability_document(self) -> str:
        """
        Auto-generate a free-text capability document from structured fields.
        Used for embedding into Qdrant.
        """
        parts = [
            f"NGO: {self.username}",
            f"Location: {self.address or 'Unknown'}",
            f"Capacity: {self.capacity_kg or 'unspecified'} kg",
            f"Operating hours: {self.operating_hours}",
            f"Reliability: {self.reliability_score}/100",
        ]

        if self.dietary_restrictions:
            parts.append(f"Dietary: {self.dietary_restrictions}")
        if self.cultural_rules:
            parts.append(f"Cultural: {self.cultural_rules}")
        if self.total_collections > 0:
            parts.append(f"Total collections: {self.total_collections}")

        doc = " | ".join(parts)
        self.capability_document = doc
        self.save(update_fields=["capability_document"])
        return doc
