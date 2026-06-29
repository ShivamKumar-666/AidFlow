"""
Agent State — TypedDict schema for the LangGraph pipeline.
Each agent reads and writes to this shared state.
"""

from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime


class DonationState(TypedDict):
    """State that flows through the agent pipeline."""

    # --- Input ---
    donation_id: Optional[int]
    donor_id: Optional[int]
    donor_username: str
    food_name: str
    description: str
    quantity_kg: float
    pickup_address: str
    latitude: Optional[float]
    longitude: Optional[float]

    # --- ML Inputs ---
    storage_time_hours: float
    time_since_cooking_hours: float
    storage_condition: str
    food_type: str
    container_type: str
    moisture_type: str
    cooking_method: str
    texture: str
    smell: str

    # --- ML Results ---
    freshness_score: float
    freshness_label: str
    ml_confidence: float
    shap_explanation: List[Dict[str, Any]]

    # --- Verification ---
    is_valid: bool
    verification_notes: List[str]
    anomalies: List[str]

    # --- Matching ---
    matched_ngos: List[Dict[str, Any]]
    current_ngo_index: int
    assigned_ngo_id: Optional[int]
    assigned_ngo_name: str

    # --- Logistics ---
    claim_sent: bool
    claim_accepted: bool
    escalation_count: int
    max_escalations: int
    needs_escalation: bool

    # --- Meta ---
    status: str  # intake, verify, match, logistics, complete, failed
    error_message: str
    decision_trail: List[Dict[str, Any]]
    started_at: str
    completed_at: str
