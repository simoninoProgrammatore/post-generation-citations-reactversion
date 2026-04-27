"""
Shared LLM client supporting Ollama (local), Anthropic Claude, and Google Gemini.

Usage:
    from llm_client import call_llm
    response = call_llm("Your prompt here", model="claude-haiku-4-5-20251001")
"""

import os
import json
import re

# ──────────────────────────────────────────────
# Ollama configuration
# ──────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Models that route to Ollama (add more as needed)
OLLAMA_MODELS = {
    "gemma3:1b", "gemma3:4b", "gemma3:12b",
    "phi4-mini", "phi4-mini:3.8b",
    "llama3.2:1b", "llama3.2:3b",
    "mistral", "mistral:7b",
    "qwen2.5:7b", "qwen2.5:3b",
}


def _is_ollama_model(model: str) -> bool:
    """Check if a model should be routed to Ollama."""
    return model in OLLAMA_MODELS or model.startswith("ollama/")


def call_llm(prompt: str, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 1024) -> str:
    if _is_ollama_model(model):
        clean_name = model.replace("ollama/", "")
        return _call_ollama(prompt, clean_name, max_tokens)
    elif model.startswith("claude"):
        return _call_claude(prompt, model, max_tokens)
    elif model.startswith("gemini"):
        return _call_gemini(prompt, model, max_tokens)
    else:
        # Default: try Ollama for unknown models
        return _call_ollama(prompt, model, max_tokens)


def call_llm_json(prompt: str, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 1024) -> any:
    import logging
    logger = logging.getLogger(__name__)
    
    raw = call_llm(prompt, model=model, max_tokens=max_tokens)
    logger.info(f"[call_llm_json] Raw LLM response ({len(raw)} chars):\n{raw[:500]}")
    
    clean = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    clean = re.sub(r"\s*```$", "", clean)

    # Try to extract JSON from response if model added extra text
    if not clean.startswith("[") and not clean.startswith("{"):
        # Look for JSON array or object in the response
        json_match = re.search(r'(\[.*\]|\{.*\})', clean, re.DOTALL)
        if json_match:
            clean = json_match.group(1)

    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        logger.error(f"[call_llm_json] JSON parse FAILED.\nRaw: {raw}\nCleaned: {clean}\nError: {e}")
        raise ValueError(f"LLM did not return valid JSON.\nRaw response:\n{raw}\nError: {e}")


# ──────────────────────────────────────────────
# Ollama backend (LOCAL)
# ──────────────────────────────────────────────

def _call_ollama(prompt: str, model: str, max_tokens: int) -> str:
    import urllib.request
    import urllib.error
    import logging
    logger = logging.getLogger(__name__)

    url = f"{OLLAMA_BASE_URL}/api/chat"
    logger.info(f"[ollama] Calling {model} at {url}")
    
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.1,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["message"]["content"]
            logger.info(f"[ollama] Response OK ({len(content)} chars)")
            return content
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            f"Make sure Ollama is running ('ollama serve').\n"
            f"Error: {e}"
        )
    except KeyError:
        raise ValueError(f"Unexpected Ollama response format: {data}")


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