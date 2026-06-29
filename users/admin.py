from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser

class CustomUserAdmin(UserAdmin):
    model = CustomUser
    # Display these fields in the admin list view
    list_display = ['username', 'email', 'role', 'phone', 'is_staff']
    
    # Add our custom fields to the "Edit User" page
    fieldsets = UserAdmin.fieldsets + (
        ('Extra Profile Info', {'fields': ('role', 'phone', 'address', 'capacity_kg')}),
        ('Location Data', {'fields': ('latitude', 'longitude')}),
    )

admin.site.register(CustomUser, CustomUserAdmin)