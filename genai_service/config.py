"""
Groq API client configuration.
Centralized model selection and API setup.
"""

import os

from groq import Groq

# Model IDs (update these as Groq deprecates models)
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TEXT_MODEL = "llama-3.3-70b-versatile"

_client: Groq | None = None


def get_client() -> Groq:
    """Get or create singleton Groq client."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment. Add it to .env")
        _client = Groq(api_key=api_key)
    return _client
