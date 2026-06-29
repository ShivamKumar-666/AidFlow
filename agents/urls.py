from django.urls import path
from .views import RunPipelineView, AgentRunsView, agent_dashboard_view

urlpatterns = [
    path("run/", RunPipelineView.as_view(), name="agent-run-pipeline"),
    path("runs/", AgentRunsView.as_view(), name="agent-runs-list"),
    path("dashboard/", agent_dashboard_view, name="agent-dashboard"),
]
