from django.contrib import admin
from .models import Donation

@admin.register(Donation)
class DonationAdmin(admin.ModelAdmin):
    list_display = ['food_name', 'donor', 'quantity_kg', 'freshness_label', 'freshness_score', 'status', 'created_at']
    list_filter = ['status', 'freshness_label', 'food_type', 'storage_condition', 'created_at']
    search_fields = ['food_name', 'donor__username', 'pickup_address', 'description']
    readonly_fields = ['freshness_score', 'freshness_label', 'confidence', 'created_at', 'claimed_at']
    list_per_page = 25
    ordering = ['-created_at']
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('donor', 'recipient', 'food_name', 'description', 'quantity_kg', 'pickup_address')
        }),
        ('ML Inputs', {
            'fields': ('storage_time_hours', 'time_since_cooking_hours', 'storage_condition', 'food_type',
                       'container_type', 'moisture_type', 'cooking_method', 'texture', 'smell'),
            'classes': ('collapse',),
        }),
        ('AI Results (Auto-filled)', {
            'fields': ('freshness_score', 'freshness_label', 'confidence'),
        }),
        ('Status & Timestamps', {
            'fields': ('status', 'created_at', 'claimed_at'),
        }),
    )
