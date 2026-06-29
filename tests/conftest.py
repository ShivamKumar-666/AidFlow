"""
Pytest configuration and fixtures for EcoFeast tests.
"""
import pytest
from django.conf import settings


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