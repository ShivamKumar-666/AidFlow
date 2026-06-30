"""
Pytest configuration and fixtures for AidFlow tests.

ALL heavy ML/AI/data libraries are stubbed via sys.modules BEFORE Django
imports any app. This lets CI run on a minimal pure-Python install with zero
C-extension compilation requirements. Each test already mocks the actual
function calls with @patch, so the stubs are completely transparent.
"""
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# ML / numeric stack — imported at module level in ml_service/
# ---------------------------------------------------------------------------
_np = MagicMock()
_np.argmax = MagicMock(return_value=0)
sys.modules['numpy'] = _np

_pd = MagicMock()
sys.modules['pandas'] = _pd
sys.modules['pandas.core'] = MagicMock()
sys.modules['pandas.core.frame'] = MagicMock()

sys.modules['joblib'] = MagicMock()

_xgb = MagicMock()
sys.modules['xgboost'] = _xgb

sys.modules['shap'] = MagicMock()

_sklearn = MagicMock()
sys.modules['sklearn'] = _sklearn
sys.modules['sklearn.model_selection'] = MagicMock()
sys.modules['sklearn.metrics'] = MagicMock()
sys.modules['sklearn.preprocessing'] = MagicMock()

# ---------------------------------------------------------------------------
# RAG / embedding stack — imported at module level in rag_service/
# ---------------------------------------------------------------------------
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['qdrant_client'] = MagicMock()
sys.modules['qdrant_client.models'] = MagicMock()

# ---------------------------------------------------------------------------
# GenAI / LLM clients — imported at module level in genai_service/
# ---------------------------------------------------------------------------
sys.modules['groq'] = MagicMock()

# ---------------------------------------------------------------------------
# Agentic orchestration — imported at module level in agents/
# ---------------------------------------------------------------------------
_lg = MagicMock()
sys.modules['langgraph'] = _lg
sys.modules['langgraph.graph'] = _lg
sys.modules['langgraph.graph.state'] = _lg
sys.modules['langgraph.checkpoint'] = _lg
sys.modules['langgraph.checkpoint.memory'] = _lg

_lc = MagicMock()
sys.modules['langchain_core'] = _lc
sys.modules['langchain_core.runnables'] = _lc
sys.modules['langchain_core.messages'] = _lc

# ---------------------------------------------------------------------------
# Other optional deps that may be transitively imported
# ---------------------------------------------------------------------------
sys.modules['celery'] = MagicMock()
sys.modules['redis'] = MagicMock()
sys.modules['geopy'] = MagicMock()
sys.modules['geopy.distance'] = MagicMock()

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