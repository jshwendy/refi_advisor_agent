"""
agent/llm.py

Anthropic API client and JSON response utilities shared across all agent nodes.
"""

import os
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

load_dotenv()


# ── API constants ─────────────────────────────────────────────────────────────

BASE_URL           = os.getenv("ANTHROPIC_BASE_URL",      "https://api.anthropic.com")
MESSAGES_EP        = os.getenv("ANTHROPIC_MESSAGES_EP",   "/v1/messages")
ANTHROPIC_VERSION  = os.getenv("ANTHROPIC_VERSION",       "2023-06-01")
MODEL              = os.getenv("ANTHROPIC_MODEL",         "claude-haiku-4-5-20251001")
DEFAULT_MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "1024"))


# ── Public helpers ────────────────────────────────────────────────────────────

def claude_chat(
    messages: List[Dict[str, str]],
    model: str = MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    system: str = None,
    **options: Any,
) -> str:
    """
    Send a chat request to the Anthropic /v1/messages endpoint.

    Args:
        messages:   List of {"role": "user"/"assistant", "content": "..."} dicts.
        model:      Claude model ID.
        max_tokens: Maximum tokens in the reply.
        system:     Optional system prompt string.
        **options:  Extra API params (e.g. temperature=0.7, top_p=0.9).

    Returns:
        The assistant's reply as a plain string.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Add it to your environment or .env file."
        )

    headers = {
        "x-api-key":         api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type":      "application/json",
    }

    payload = {
        "model":      model,
        "max_tokens": max_tokens,
        "messages":   messages,
        **options,
    }
    if system:
        payload["system"] = system

    try:
        r = requests.post(
            f"{BASE_URL}{MESSAGES_EP}",
            headers=headers,
            json=payload,
            timeout=180,
        )
    except requests.exceptions.ConnectionError:
        raise SystemExit("Cannot reach Anthropic API. Check your internet connection.")

    if r.status_code != 200:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise RuntimeError(f"/v1/messages error {r.status_code}: {detail}")

    content_blocks = r.json().get("content", [])
    return "".join(b["text"] for b in content_blocks if b.get("type") == "text")


def strip_json_fences(text: str) -> str:
    """Strip markdown code fences from a JSON response string."""
    return text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
