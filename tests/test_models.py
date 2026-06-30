"""
Tests for Donation and AgentRun models.
"""

from django.contrib.auth import get_user_model


class TestUserModel:
    def test_user_model_exists(self):
        User = get_user_model()
        assert User is not None

    def test_user_has_role_field(self):
        User = get_user_model()
        fields = [f.name for f in User._meta.get_fields()]
        assert "role" in fields

    def test_user_has_ngo_capability_fields(self):
        User = get_user_model()
        field_names = {f.name: f for f in User._meta.get_fields()}
        ngo_fields = [
            "total_collections",
            "dietary_restrictions",
            "cultural_rules",
            "operating_hours",
            "reliability_score",
            "capability_document",
        ]
        for f in ngo_fields:
            assert f in field_names, f"Missing field: {f}"

    def test_user_roles(self):
        User = get_user_model()
        role_field = User._meta.get_field("role")
        role_labels = [choice[0] for choice in role_field.choices]
        assert "donor" in role_labels
        assert "ngo" in role_labels
        assert "shelter" in role_labels
        assert "restaurant" in role_labels


class TestDonationModel:
    def test_donation_model_exists(self):
        from donations.models import Donation

        assert Donation is not None

    def test_donation_has_required_fields(self):
        from donations.models import Donation

        field_names = {f.name for f in Donation._meta.get_fields()}
        required = ["food_name", "quantity_kg", "status", "donor"]
        for f in required:
            assert f in field_names

    def test_donation_status_choices(self):
        from donations.models import Donation

        assert hasattr(Donation, "Status")
        statuses = [s[0] for s in Donation.Status.choices]
        assert "pending" in statuses
        assert "claimed" in statuses
        assert "picked_up" in statuses
        assert "delivered" in statuses
        assert "expired" in statuses

    def test_donation_freshness_fields(self):
        from donations.models import Donation

        field_names = {f.name for f in Donation._meta.get_fields()}
        assert "freshness_score" in field_names
        assert "freshness_label" in field_names
        assert "confidence" in field_names

    def test_donation_str(self):
        from donations.models import Donation

        d = Donation(food_name="Test Rice", freshness_label="Fresh", freshness_score=85)
        result = str(d)
        assert "Test Rice" in result
        assert "Fresh" in result
        assert "85" in result


class TestAgentRunModel:
    def test_agentrun_model_exists(self):
        from donations.models import AgentRun

        assert AgentRun is not None

    def test_agentrun_has_trail_field(self):
        from donations.models import AgentRun

        field_names = {f.name for f in AgentRun._meta.get_fields()}
        assert "decision_trail" in field_names
        assert "status" in field_names
        assert "ml_freshness_score" in field_names
        assert "escalations" in field_names
        assert "duration_seconds" in field_names

    def test_agentrun_donation_nullable(self):
        from donations.models import AgentRun

        field = AgentRun._meta.get_field("donation")
        assert field.null is True
        assert field.blank is True

    def test_agentrun_str(self):
        from donations.models import AgentRun

        run = AgentRun(status="complete")
        result = str(run)
        assert "complete" in result or "Run" in result


class TestModelRelationships:
    def test_donation_agent_runs_related_name(self):
        from donations.models import AgentRun, Donation

        donation_field = Donation._meta.get_field("agent_runs")
        assert donation_field.related_model == AgentRun


class TestTimestampMixin:
    def test_donation_has_timestamps(self):
        from donations.models import Donation

        field_names = {f.name for f in Donation._meta.get_fields()}
        assert "created_at" in field_names
        # updated_at will be tested after we add it

    def test_agentrun_has_timestamps(self):
        from donations.models import AgentRun

        field_names = {f.name for f in AgentRun._meta.get_fields()}
        assert "started_at" in field_names
        assert "completed_at" in field_names
        assert "created_at" in field_names
