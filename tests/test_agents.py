"""
Tests for Agent Pipeline: State, Intake, Verification, Matching, Logistics, Orchestrator.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestDonationState:
    def test_state_import(self):
        from agents.state import DonationState
        assert DonationState is not None

    def test_state_keys(self):
        from agents.state import DonationState
        hints = DonationState.__annotations__
        required_keys = ['food_name', 'quantity_kg', 'status', 'freshness_score', 'is_valid']
        for k in required_keys:
            assert k in hints


class TestIntakeAgent:
    def test_intake_agent_import(self):
        from agents.intake_agent import intake_agent
        assert intake_agent is not None

    @patch('ml_service.predictor.get_predictor')
    @patch('ml_service.explainer.get_explainer')
    def test_intake_agent_success(self, mock_get_explainer, mock_get_predictor):
        from agents.intake_agent import intake_agent

        mock_pred = MagicMock()
        mock_pred.freshness_score = 85
        mock_pred.freshness_label = 'Fresh'
        mock_pred.confidence = 92.0
        mock_get_predictor.return_value.predict.return_value = mock_pred
        mock_get_explainer.return_value.explain.return_value = [{'feature': 'storage_condition_refrigerated', 'impact': 0.5, 'direction': 'positive'}]

        state = {
            'donation_id': None, 'donor_id': None, 'donor_username': 'test',
            'food_name': 'Rice', 'description': '', 'quantity_kg': 5.0,
            'pickup_address': '', 'latitude': None, 'longitude': None,
            'storage_time_hours': 4.0, 'time_since_cooking_hours': 2.0,
            'storage_condition': 'refrigerated', 'food_type': 'Vegetarian',
            'container_type': 'closed', 'moisture_type': 'dry',
            'cooking_method': 'boiled', 'texture': 'firm', 'smell': 'neutral',
            'freshness_score': 0, 'freshness_label': '', 'ml_confidence': 0,
            'shap_explanation': [], 'is_valid': False,
            'verification_notes': [], 'anomalies': [],
            'matched_ngos': [], 'current_ngo_index': 0,
            'assigned_ngo_id': None, 'assigned_ngo_name': '',
            'claim_sent': False, 'claim_accepted': False,
            'escalation_count': 0, 'max_escalations': 3, 'needs_escalation': False,
            'status': 'starting', 'error_message': '', 'decision_trail': [],
            'started_at': '', 'completed_at': '',
        }

        result = intake_agent(state)

        assert result['freshness_score'] == 85
        assert result['freshness_label'] == 'Fresh'
        assert result['ml_confidence'] == 92.0
        assert len(result['shap_explanation']) == 1
        assert result['status'] == 'intake_complete'
        assert len(result['decision_trail']) == 1
        assert result['decision_trail'][0]['agent'] == 'intake'

    @patch('ml_service.predictor.get_predictor')
    @patch('ml_service.explainer.get_explainer')
    def test_intake_agent_error(self, mock_get_explainer, mock_get_predictor):
        from agents.intake_agent import intake_agent

        mock_get_predictor.side_effect = Exception('Model load failed')

        state = {
            'donation_id': None, 'donor_id': None, 'donor_username': 'test',
            'food_name': 'Rice', 'description': '', 'quantity_kg': 5.0,
            'pickup_address': '', 'latitude': None, 'longitude': None,
            'storage_time_hours': 4.0, 'time_since_cooking_hours': 2.0,
            'storage_condition': 'refrigerated', 'food_type': 'Vegetarian',
            'container_type': 'closed', 'moisture_type': 'dry',
            'cooking_method': 'boiled', 'texture': 'firm', 'smell': 'neutral',
            'freshness_score': 0, 'freshness_label': '', 'ml_confidence': 0,
            'shap_explanation': [], 'is_valid': False,
            'verification_notes': [], 'anomalies': [],
            'matched_ngos': [], 'current_ngo_index': 0,
            'assigned_ngo_id': None, 'assigned_ngo_name': '',
            'claim_sent': False, 'claim_accepted': False,
            'escalation_count': 0, 'max_escalations': 3, 'needs_escalation': False,
            'status': 'starting', 'error_message': '', 'decision_trail': [],
            'started_at': '', 'completed_at': '',
        }

        result = intake_agent(state)

        assert result['freshness_score'] == 0
        assert result['freshness_label'] == 'Unknown'
        assert 'error' in result['decision_trail'][0]


class TestVerificationAgent:
    def test_verification_agent_import(self):
        from agents.verification_agent import verification_agent
        assert verification_agent is not None

    def test_verify_passes_good_donation(self):
        from agents.verification_agent import verification_agent

        state = {'quantity_kg': 10, 'storage_time_hours': 3,
                 'time_since_cooking_hours': 2, 'food_type': 'Vegetarian',
                 'storage_condition': 'refrigerated', 'freshness_score': 85,
                 'smell': 'neutral', 'anomalies': [], 'verification_notes': [],
                 'is_valid': False, 'status': '', 'decision_trail': []}

        result = verification_agent(state)

        assert result['is_valid'] is True
        assert len(result['anomalies']) == 0

    def test_verify_fails_large_quantity(self):
        from agents.verification_agent import verification_agent

        state = {'quantity_kg': 100, 'storage_time_hours': 3,
                 'time_since_cooking_hours': 2, 'food_type': 'Vegetarian',
                 'storage_condition': 'refrigerated', 'freshness_score': 85,
                 'smell': 'neutral', 'anomalies': [], 'verification_notes': [],
                 'is_valid': False, 'status': '', 'decision_trail': []}

        result = verification_agent(state)

        assert len(result['anomalies']) == 1
        assert 'Unusually large' in result['anomalies'][0]

    def test_verify_fails_invalid_quantity(self):
        from agents.verification_agent import verification_agent

        state = {'quantity_kg': -1, 'storage_time_hours': 3,
                 'time_since_cooking_hours': 2, 'food_type': 'Vegetarian',
                 'storage_condition': 'refrigerated', 'freshness_score': 85,
                 'smell': 'neutral', 'anomalies': [], 'verification_notes': [],
                 'is_valid': False, 'status': '', 'decision_trail': []}

        result = verification_agent(state)

        assert result['is_valid'] is False
        assert any('Invalid quantity' in a for a in result['anomalies'])

    def test_verify_fails_old_food(self):
        from agents.verification_agent import verification_agent

        state = {'quantity_kg': 5, 'storage_time_hours': 30,
                 'time_since_cooking_hours': 20, 'food_type': 'Vegetarian',
                 'storage_condition': 'refrigerated', 'freshness_score': 30,
                 'smell': 'neutral', 'anomalies': [], 'verification_notes': [],
                 'is_valid': False, 'status': '', 'decision_trail': []}

        result = verification_agent(state)

        assert 'likely unsafe' in result['anomalies'][0] or 'check quality' in result['anomalies'][0]

    def test_verify_fails_low_freshness(self):
        from agents.verification_agent import verification_agent

        state = {'quantity_kg': 5, 'storage_time_hours': 4,
                 'time_since_cooking_hours': 2, 'food_type': 'Vegetarian',
                 'storage_condition': 'refrigerated', 'freshness_score': 15,
                 'smell': 'neutral', 'anomalies': [], 'verification_notes': [],
                 'is_valid': False, 'status': '', 'decision_trail': []}

        result = verification_agent(state)

        assert result['is_valid'] is False
        assert any('unsafe' in a for a in result['anomalies'])

    def test_verify_detects_bad_smell(self):
        from agents.verification_agent import verification_agent

        state = {'quantity_kg': 5, 'storage_time_hours': 4,
                 'time_since_cooking_hours': 2, 'food_type': 'Vegetarian',
                 'storage_condition': 'refrigerated', 'freshness_score': 70,
                 'smell': 'sour', 'anomalies': [], 'verification_notes': [],
                 'is_valid': False, 'status': '', 'decision_trail': []}

        result = verification_agent(state)

        assert any('spoiled' in a for a in result['anomalies'])

    def test_verify_trail_entry(self):
        from agents.verification_agent import verification_agent

        state = {'quantity_kg': 5, 'storage_time_hours': 4,
                 'time_since_cooking_hours': 2, 'food_type': 'Vegetarian',
                 'storage_condition': 'refrigerated', 'freshness_score': 85,
                 'smell': 'neutral', 'anomalies': [], 'verification_notes': [],
                 'is_valid': False, 'status': '', 'decision_trail': []}

        result = verification_agent(state)

        assert len(result['decision_trail']) == 1
        assert result['decision_trail'][0]['agent'] == 'verification'
        assert 'anomalies' in result['decision_trail'][0]
        assert 'is_valid' in result['decision_trail'][0]


class TestMatchingAgent:
    def test_matching_agent_import(self):
        from agents.matching_agent import matching_agent
        assert matching_agent is not None

    @patch('rag_service.matcher.match_donation_to_ngos')
    def test_matching_finds_ngos(self, mock_match):
        from agents.matching_agent import matching_agent

        mock_match.return_value = [
            {'ngo_id': 1, 'ngo_name': 'NGO One', 'combined_score': 0.85},
            {'ngo_id': 2, 'ngo_name': 'NGO Two', 'combined_score': 0.72},
        ]

        state = {'food_name': 'Rice', 'food_type': 'Vegetarian', 'quantity_kg': 5.0,
                 'freshness_score': 85, 'storage_condition': 'refrigerated',
                 'latitude': 19.076, 'longitude': 72.877,
                 'matched_ngos': [], 'current_ngo_index': 0,
                 'assigned_ngo_id': None, 'assigned_ngo_name': '',
                 'status': '', 'decision_trail': []}

        result = matching_agent(state)

        assert len(result['matched_ngos']) == 2
        assert result['assigned_ngo_name'] == 'NGO One'
        assert result['current_ngo_index'] == 0

    @patch('rag_service.matcher.match_donation_to_ngos')
    def test_matching_finds_none(self, mock_match):
        from agents.matching_agent import matching_agent

        mock_match.return_value = []

        state = {'food_name': 'Rice', 'food_type': 'Vegetarian', 'quantity_kg': 5.0,
                 'freshness_score': 85, 'storage_condition': 'refrigerated',
                 'latitude': 19.076, 'longitude': 72.877,
                 'matched_ngos': [], 'current_ngo_index': 0,
                 'assigned_ngo_id': None, 'assigned_ngo_name': '',
                 'status': '', 'decision_trail': []}

        result = matching_agent(state)

        assert len(result['matched_ngos']) == 0
        assert result['assigned_ngo_id'] is None

    @patch('rag_service.matcher.match_donation_to_ngos')
    def test_matching_handles_error(self, mock_match):
        from agents.matching_agent import matching_agent

        mock_match.side_effect = Exception('RAG service down')

        state = {'food_name': 'Rice', 'food_type': 'Vegetarian', 'quantity_kg': 5.0,
                 'freshness_score': 85, 'storage_condition': 'refrigerated',
                 'latitude': 19.076, 'longitude': 72.877,
                 'matched_ngos': [], 'current_ngo_index': 0,
                 'assigned_ngo_id': None, 'assigned_ngo_name': '',
                 'status': '', 'decision_trail': []}

        result = matching_agent(state)

        assert len(result['matched_ngos']) == 0
        assert 'error' in result['decision_trail'][0]


class TestLogisticsAgent:
    def test_logistics_agent_import(self):
        from agents.logistics_agent import logistics_agent
        assert logistics_agent is not None

    def test_logistics_no_ngos(self):
        from agents.logistics_agent import logistics_agent

        state = {'matched_ngos': [], 'current_ngo_index': 0,
                 'escalation_count': 0, 'max_escalations': 3,
                 'claim_sent': False, 'claim_accepted': False,
                 'needs_escalation': False, 'assigned_ngo_id': None,
                 'assigned_ngo_name': '', 'status': '', 'error_message': '',
                 'decision_trail': []}

        result = logistics_agent(state)

        assert result['status'] == 'failed'
        assert 'No NGOs available' in result['error_message']

    def test_logistics_all_ngos_exhausted(self):
        from agents.logistics_agent import logistics_agent

        state = {'matched_ngos': [{'ngo_id': 1, 'ngo_name': 'NGO One', 'reliability_score': 80}],
                 'current_ngo_index': 5, 'escalation_count': 3, 'max_escalations': 3,
                 'claim_sent': False, 'claim_accepted': False,
                 'needs_escalation': False, 'assigned_ngo_id': None,
                 'assigned_ngo_name': '', 'status': '', 'error_message': '',
                 'decision_trail': []}

        result = logistics_agent(state)

        assert result['status'] == 'failed'
        assert 'exhausted' in result['error_message']

    @patch('random.random')
    def test_logistics_accepts_claim(self, mock_random):
        from agents.logistics_agent import logistics_agent

        mock_random.return_value = 0.1  # Below threshold → accepted

        state = {'matched_ngos': [{'ngo_id': 1, 'ngo_name': 'NGO One', 'reliability_score': 90}],
                 'current_ngo_index': 0, 'escalation_count': 0, 'max_escalations': 3,
                 'claim_sent': False, 'claim_accepted': False,
                 'needs_escalation': False, 'assigned_ngo_id': None,
                 'assigned_ngo_name': '', 'status': '', 'error_message': '',
                 'decision_trail': []}

        result = logistics_agent(state)

        assert result['claim_accepted'] is True
        assert result['assigned_ngo_name'] == 'NGO One'
        assert result['status'] == 'complete'

    @patch('random.random')
    def test_logistics_escalates_on_rejection(self, mock_random):
        from agents.logistics_agent import logistics_agent

        mock_random.return_value = 1.0  # Above threshold → rejected

        state = {'matched_ngos': [
            {'ngo_id': 1, 'ngo_name': 'NGO One', 'reliability_score': 50},
            {'ngo_id': 2, 'ngo_name': 'NGO Two', 'reliability_score': 80},
        ], 'current_ngo_index': 0, 'escalation_count': 0, 'max_escalations': 3,
                 'claim_sent': False, 'claim_accepted': False,
                 'needs_escalation': False, 'assigned_ngo_id': None,
                 'assigned_ngo_name': '', 'status': '', 'error_message': '',
                 'decision_trail': []}

        result = logistics_agent(state)

        assert result['claim_accepted'] is False
        assert result['needs_escalation'] is True
        assert result['escalation_count'] == 1
        assert result['current_ngo_index'] == 1
        assert result['status'] == 'escalating'

    @patch('random.random')
    def test_logistics_max_escalations(self, mock_random):
        from agents.logistics_agent import logistics_agent

        mock_random.return_value = 1.0  # Always reject

        state = {'matched_ngos': [{'ngo_id': 1, 'ngo_name': 'NGO One', 'reliability_score': 50}],
                 'current_ngo_index': 0, 'escalation_count': 2, 'max_escalations': 3,
                 'claim_sent': False, 'claim_accepted': False,
                 'needs_escalation': False, 'assigned_ngo_id': None,
                 'assigned_ngo_name': '', 'status': '', 'error_message': '',
                 'decision_trail': []}

        result = logistics_agent(state)

        assert result['claim_accepted'] is False
        assert result['escalation_count'] == 3
        assert result['needs_escalation'] is False
        assert result['status'] == 'failed'


class TestOrchestrator:
    def test_routing_verify_valid(self):
        from agents.orchestrator import route_after_verify
        result = route_after_verify({'is_valid': True})
        assert result == 'match'

    def test_routing_verify_invalid(self):
        from agents.orchestrator import route_after_verify
        result = route_after_verify({'is_valid': False})
        assert result == 'end'

    def test_routing_logistics_complete(self):
        from agents.orchestrator import route_after_logistics
        result = route_after_logistics({'needs_escalation': False})
        assert result == 'end'

    def test_routing_logistics_escalate(self):
        from agents.orchestrator import route_after_logistics
        result = route_after_logistics({'needs_escalation': True})
        assert result == 'logistics'

    def test_build_graph(self):
        from agents.orchestrator import build_graph
        graph = build_graph()
        assert graph is not None

    def test_get_graph_singleton(self):
        from agents.orchestrator import get_graph
        g1 = get_graph()
        g2 = get_graph()
        assert g1 is g2