from django.urls import path
from .views import VisionIntakeView, ChatIntakeView, SHAPExplanationView

urlpatterns = [
    path("vision/", VisionIntakeView.as_view(), name="genai-vision"),
    path("chat/", ChatIntakeView.as_view(), name="genai-chat"),
    path("explain/", SHAPExplanationView.as_view(), name="genai-explain"),
]
