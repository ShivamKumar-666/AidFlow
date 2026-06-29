from django.urls import path
from .views import PredictFreshnessView

urlpatterns = [
    path('predict/', PredictFreshnessView.as_view(), name='predict-freshness'),
]