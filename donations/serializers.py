from rest_framework import serializers
from .models import Donation
from users.serializers import UserSerializer

class DonationSerializer(serializers.ModelSerializer):
    donor_name = serializers.ReadOnlyField(source='donor.username')
    lat = serializers.ReadOnlyField(source='donor.latitude')
    lng = serializers.ReadOnlyField(source='donor.longitude')
    
    # NGO Details (using 'recipient')
    recipient_details = UserSerializer(source='recipient', read_only=True)
    
    class Meta:
        model = Donation
        fields = [
            'id', 'donor_name', 'food_name', 'description', 'quantity_kg',
            'pickup_address', 'storage_time_hours', 'time_since_cooking_hours',
            'storage_condition', 'food_type',
            'container_type', 'moisture_type', 'cooking_method', 'texture', 'smell',
            'freshness_score', 'freshness_label', 'confidence',
            'status', 'created_at', 'claimed_at',
            'lat', 'lng', 
            'recipient_details'
        ]
        
        read_only_fields = ['id', 'donor', 'freshness_score', 'freshness_label', 'confidence', 'created_at', 'claimed_at', 'recipient']