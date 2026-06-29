from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/users/', include('users.urls')),
    path('api/ml/', include('ml_service.urls')),
    path('api/donations/', include('donations.urls')),
    
    path('', include('users.urls')),
]