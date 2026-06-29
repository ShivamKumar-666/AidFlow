from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/users/', include('users.urls')),
    path('api/ml/', include('ml_service.urls')),
    path('api/donations/', include('donations.urls')),
    path('api/genai/', include('genai_service.urls')),
    path('api/rag/', include('rag_service.urls')),
    path('api/agents/', include('agents.urls')),
    
    path('', include('users.urls')),
]