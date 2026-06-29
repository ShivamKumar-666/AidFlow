from django.urls import path
from .views import MatchDonationView, SyncNGOsView, SyncSingleNGOView

urlpatterns = [
    path("match/", MatchDonationView.as_view(), name="rag-match"),
    path("sync/", SyncNGOsView.as_view(), name="rag-sync-all"),
    path("sync/<int:ngo_id>/", SyncSingleNGOView.as_view(), name="rag-sync-single"),
]
