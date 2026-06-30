from django.urls import path

from .views import RegisterUserView, UserProfileUpdateView, landing_view, login_view, logout_view, register_view

urlpatterns = [
    # API ENDPOINT (For the Register Form to talk to)
    path("api/register/", RegisterUserView.as_view(), name="register-api"),
    # FRONTEND PAGES (For Humans)
    path("", landing_view, name="landing"),  # <--- This loads landing.html
    path("login/", login_view, name="login-ui"),
    path("signup/", register_view, name="register-ui"),  # <--- This loads register.html
    path("logout/", logout_view, name="logout-ui"),
    path("profile/update/", UserProfileUpdateView.as_view(), name="profile-update"),
]
