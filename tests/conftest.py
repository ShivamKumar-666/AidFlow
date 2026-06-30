"""
Pytest configuration and fixtures for AidFlow tests.

Heavy ML/AI libraries (sentence-transformers, shap, qdrant-client, groq) are
stubbed out via sys.modules BEFORE Django apps are imported. This means CI
only needs the lightweight Django/DRF stack — the actual ML calls are mocked
in each individual test via @patch decorators.
"""
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub out heavy libraries that are imported at module level inside the app.
# These must be set before Django does its app registry import sweep.
# ---------------------------------------------------------------------------

# shap — imported at top of ml_service/explainer.py
_shap_mock = MagicMock()
sys.modules['shap'] = _shap_mock

# sentence_transformers — imported at top of rag_service/ngo_embeddings.py
_st_mock = MagicMock()
sys.modules['sentence_transformers'] = _st_mock

# qdrant_client — imported at top of rag_service/ngo_embeddings.py
_qdrant_mock = MagicMock()
_qdrant_models_mock = MagicMock()
sys.modules['qdrant_client'] = _qdrant_mock
sys.modules['qdrant_client.models'] = _qdrant_models_mock

# groq — imported at top of genai_service/config.py
_groq_mock = MagicMock()
sys.modules['groq'] = _groq_mock

# langgraph submodules that may be imported at module level in agents/
_lg_mock = MagicMock()
sys.modules['langgraph'] = _lg_mock
sys.modules['langgraph.graph'] = _lg_mock
sys.modules['langgraph.graph.state'] = _lg_mock
sys.modules['langgraph.checkpoint'] = _lg_mock
sys.modules['langgraph.checkpoint.memory'] = _lg_mock

# langchain_core — may be imported by langgraph or agents
_lc_mock = MagicMock()
sys.modules['langchain_core'] = _lc_mock
sys.modules['langchain_core.runnables'] = _lc_mock
sys.modules['langchain_core.messages'] = _lc_mock

# ---------------------------------------------------------------------------

import pytest  # noqa: E402
from django.conf import settings  # noqa: E402


def pytest_configure():
    """Configure Django settings for pytest."""
    if not settings.configured:
        settings.configure(
            DEBUG=True,
            DATABASES={
                'default': {
                    'ENGINE': 'django.db.backends.sqlite3',
                    'NAME': ':memory:',
                }
            },
            INSTALLED_APPS=[
                'django.contrib.admin',
                'django.contrib.auth',
                'django.contrib.contenttypes',
                'django.contrib.sessions',
                'django.contrib.messages',
                'django.contrib.staticfiles',
                'rest_framework',
                'corsheaders',
                'users',
                'donations',
                'ml_service',
                'genai_service',
                'rag_service',
                'agents',
            ],
            MIDDLEWARE=[
                'django.middleware.security.SecurityMiddleware',
                'django.contrib.sessions.middleware.SessionMiddleware',
                'corsheaders.middleware.CorsMiddleware',
                'django.middleware.common.CommonMiddleware',
                'django.middleware.csrf.CsrfViewMiddleware',
                'django.contrib.auth.middleware.AuthenticationMiddleware',
                'django.contrib.messages.middleware.MessageMiddleware',
                'django.middleware.clickjacking.XFrameOptionsMiddleware',
            ],
            ROOT_URLCONF='config.urls',
            TEMPLATES=[{
                'BACKEND': 'django.template.backends.django.DjangoTemplates',
                'DIRS': [],
                'APP_DIRS': True,
                'OPTIONS': {
                    'context_processors': [
                        'django.template.context_processors.request',
                        'django.contrib.auth.context_processors.auth',
                        'django.contrib.messages.context_processors.messages',
                    ],
                },
            }],
            AUTH_USER_MODEL='users.CustomUser',
            SECRET_KEY='test-secret-key',
            USE_TZ=True,
            TIME_ZONE='Asia/Kolkata',
            STATIC_URL='/static/',
            DEFAULT_AUTO_FIELD='django.db.models.BigAutoField',
            REST_FRAMEWORK={
                'DEFAULT_PERMISSION_CLASSES': ['rest_framework.permissions.AllowAny'],
                'DEFAULT_AUTHENTICATION_CLASSES': [],
            },
            CORS_ALLOW_ALL_ORIGINS=True,
            QDRANT_URL='http://localhost:6333',
            GROQ_API_KEY='test-key',
            ML_MODEL_DIR='/tmp/ml_models',
        )


@pytest.fixture(scope='session')
def django_db_setup():
    """Set up database for testing."""
    import django
    from django.db import connection
    django.setup()
    with connection.cursor() as cursor:
        cursor.execute("PRAGMA foreign_keys = ON")


@pytest.fixture
def api_client():
    """DRF API client for testing."""
    from rest_framework.test import APIClient
    return APIClient()