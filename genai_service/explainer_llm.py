"""
SHAP-to-Language Explainer — Generate human-readable freshness explanations.
Combines SHAP feature contributions with LLM for natural language output.
"""

import logging
from typing import Dict, List
from .config import get_client, TEXT_MODEL

logger = logging.getLogger(__name__)

EXPLANATION_PROMPT = """You are a food safety assistant for NGOs. Given the following freshness analysis, write ONE concise, human-readable sentence explaining why the food is rated this way.

Freshness Level: {freshness_label} ({freshness_score}%)
Confidence: {confidence}%

Top Contributing Factors:
{shap_features}

Rules:
- Be direct and actionable
- Mention the most impactful factor first
- If fresh: briefly note what makes it safe
- If moderate: note the concern and estimated remaining window
- If spoiled: state clearly it should NOT be consumed
- Keep it under 25 words
- No markdown, no bullets, just one clean sentence"""


def explain_freshness(
    freshness_label: str,
    freshness_score: int,
    confidence: float,
    shap_features: List[Dict],
) -> str:
    """
    Generate a human-readable freshness explanation.

    Args:
        freshness_label: "Fresh", "Medium", or "Spoiled"
        freshness_score: 0-100 score
        confidence: Model confidence %
        shap_features: List of {"feature", "impact", "direction"} dicts

    Returns:
        One-line explanation string
    """
    if not shap_features:
        return _simple_explanation(freshness_label, freshness_score)

    # Format SHAP features for prompt
    shap_text = "\n".join(
        f"- {f['feature']}: {f['impact']:+.4f} ({f['direction']})"
        for f in shap_features[:3]
    )

    prompt = EXPLANATION_PROMPT.format(
        freshness_label=freshness_label,
        freshness_score=freshness_score,
        confidence=confidence,
        shap_features=shap_text,
    )

    try:
        client = get_client()
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": "You are a concise food safety assistant. Output only the explanation sentence, nothing else."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=100,
        )

        explanation = response.choices[0].message.content.strip()

        # Clean up any markdown or extra formatting
        explanation = explanation.strip('"').strip("'")
        if explanation.startswith("- "):
            explanation = explanation[2:]

        return explanation

    except Exception as e:
        logger.error(f"LLM explanation failed: {e}")
        return _simple_explanation(freshness_label, freshness_score)


def _simple_explanation(label: str, score: int) -> str:
    """Fallback explanation without LLM."""
    if label == "Fresh":
        return f"Fresh food — safe for distribution ({score}% freshness score)."
    elif label == "Medium":
        return f"Moderate freshness — should be consumed within 1-2 hours ({score}% score)."
    else:
        return f"Food is past safe consumption — do not distribute ({score}% score)."
