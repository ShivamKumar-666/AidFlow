from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import redirect, render
from rest_framework import generics, permissions, status
from rest_framework.permissions import IsAuthenticated  # Added this import
from rest_framework.response import Response
from rest_framework.views import APIView  # Added this import

from .models import CustomUser
from .serializers import UserSerializer

# ==========================================
# 1. API VIEWS (For Computer Code/Forms)
# ==========================================


class RegisterUserView(generics.CreateAPIView):
    queryset = CustomUser.objects.all()
    serializer_class = UserSerializer
    permission_classes = [permissions.AllowAny]

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        password = request.data.get("password")
        if not password:
            return Response({"error": "Password is required"}, status=status.HTTP_400_BAD_REQUEST)

        user = serializer.save()
        user.set_password(password)
        user.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class UserProfileUpdateView(APIView):
    permission_classes = [IsAuthenticated]

    def patch(self, request):
        user = request.user
        data = request.data

        # Update fields if provided with validation
        if "latitude" in data:
            try:
                lat = float(data["latitude"])
                if -90 <= lat <= 90:
                    user.latitude = lat
            except ValueError:
                pass
        if "longitude" in data:
            try:
                lng = float(data["longitude"])
                if -180 <= lng <= 180:
                    user.longitude = lng
            except ValueError:
                pass
        if "address" in data:
            addr = str(data["address"]).strip()
            if len(addr) <= 1000:
                user.address = addr

        user.save()
        return Response({"message": "Profile Updated Successfully!"}, status=status.HTTP_200_OK)


# ==========================================
# 2. FRONTEND VIEWS (For Humans/HTML Pages)
# ==========================================


def landing_view(request):
    """Renders the Home/Landing Page"""
    return render(request, "landing.html")


def login_view(request):
    """Handles User Login"""
    if request.method == "POST":
        u = request.POST.get("username")
        p = request.POST.get("password")
        user = authenticate(request, username=u, password=p)
        if user:
            login(request, user)

            # 🟢 REDIRECT TO THE CORRECT DASHBOARD
            if user.role == "ngo":
                return redirect("ngo-dashboard-ui")  # Goes to /api/donations/ngo-dashboard/
            elif user.role == "donor":
                return redirect("donor-dashboard-ui")  # Goes to /api/donations/donor-dashboard/

            return redirect("donor-dashboard-ui")  # Fallback
        else:
            messages.error(request, "Invalid Credentials")

    return render(request, "auth.html")


def register_view(request):
    """Renders the combined Auth page for registration"""
    return render(request, "auth.html")


def logout_view(request):
    """Logs out the user and sends them Home"""
    logout(request)
    return redirect("landing")
