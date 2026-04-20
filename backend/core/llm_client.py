"""
Shared LLM client supporting Google Gemini and Anthropic Claude.

Usage:
    from llm_client import call_llm
    response = call_llm("Your prompt here", model="claude-sonnet-4-20250514")
"""

import os
import json
import re


def call_llm(prompt: str, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 1024) -> str:
    if model.startswith("claude"):
        return _call_claude(prompt, model, max_tokens)
    elif model.startswith("gemini"):
        return _call_gemini(prompt, model, max_tokens)
    else:
        raise ValueError(f"Unsupported model '{model}'.")


def call_llm_json(prompt: str, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 1024) -> any:
    raw = call_llm(prompt, model=model, max_tokens=max_tokens)
    clean = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    clean = re.sub(r"\s*```$", "", clean)
    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON.\nRaw response:\n{raw}\nError: {e}")


# ──────────────────────────────────────────────
# Claude backend
# ──────────────────────────────────────────────

def _call_claude(prompt: str, model: str, max_tokens: int) -> str:
    try:
        import anthropic
    except ImportError as e:
        raise ImportError(
            "anthropic package not found. Install it with:\n"
            "  pip install anthropic"
        ) from e

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Add it to your .env file."
        )

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ──────────────────────────────────────────────
# Gemini backend
# ──────────────────────────────────────────────

def _call_gemini(prompt: str, model: str, max_tokens: int) -> str:
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise ImportError(
            "google-genai package not found. Install it with:\n"
            "  pip install google-genai"
        ) from e

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=max_tokens),
    )
    return response.text