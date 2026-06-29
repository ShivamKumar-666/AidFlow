from django.urls import path
from django.shortcuts import redirect # <--- Add this import
from django.contrib.auth.decorators import login_required
from .views import (
    CreateDonationView, 
    ListDonationsView, 
    DonationUpdateView,
    donor_dashboard_view, 
    ngo_dashboard_view, 
    map_dashboard_view
)

# 🟢 Helper function to catch the "Ghost" URL
@login_required
def fix_dashboard_redirect(request):
    if request.user.role == 'ngo':
        return redirect('ngo-dashboard-ui')
    return redirect('donor-dashboard-ui')

urlpatterns = [
    # API Endpoints
    path('create/', CreateDonationView.as_view(), name='create-donation'),
    path('list/', ListDonationsView.as_view(), name='list-donations'),
    path('update/<int:pk>/', DonationUpdateView.as_view(), name='update-donation'),

    # HTML Views
    path('donor-dashboard/', donor_dashboard_view, name='donor-dashboard-ui'),
    path('ngo-dashboard/', ngo_dashboard_view, name='ngo-dashboard-ui'),
    path('map/', map_dashboard_view, name='map-dashboard'),

    # 🟢 SAFETY NET: Catch the "dashboard/" error and fix it automatically
    path('dashboard/', fix_dashboard_redirect), 
]