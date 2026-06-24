"""Provider-agnostic LLM completion client for JSON hypothesis proposals.

The SDK imports are intentionally lazy: deterministic generators and tests should
not require OpenAI or Anthropic packages unless the opt-in ``llm`` family is used.
"""

from __future__ import annotations

import os
from typing import Any


class LlmClientError(RuntimeError):
    """Raised when the configured LLM provider cannot complete a request."""


def complete_json(system: str, user: str) -> str:
    """Return raw JSON text from the configured LLM provider.

    Environment:
      - ``LLM_PROVIDER``: ``anthropic`` or ``openai``
      - ``LLM_MODEL``: provider model name
      - provider key: ``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY``
    """
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    model = os.environ.get("LLM_MODEL", "").strip()
    if provider not in {"anthropic", "openai"}:
        raise LlmClientError("LLM_PROVIDER must be 'anthropic' or 'openai'")
    if not model:
        raise LlmClientError("LLM_MODEL must be set")
    if provider == "anthropic":
        return _complete_anthropic(model, system, user)
    return _complete_openai(model, system, user)


def _complete_anthropic(model: str, system: str, user: str) -> str:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise LlmClientError("ANTHROPIC_API_KEY must be set for LLM_PROVIDER=anthropic")
    try:
        import anthropic
    except ImportError as exc:
        raise LlmClientError(
            "install the 'anthropic' package to use LLM_PROVIDER=anthropic"
        ) from exc

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0.2,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return _content_to_text(response.content)


def _complete_openai(model: str, system: str, user: str) -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        raise LlmClientError("OPENAI_API_KEY must be set for LLM_PROVIDER=openai")
    try:
        import openai
    except ImportError as exc:
        raise LlmClientError("install the 'openai' package to use LLM_PROVIDER=openai") from exc

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise LlmClientError("OpenAI returned an empty JSON completion")
    return content


def _content_to_text(content: Any) -> str:
    """Extract text from Anthropic content blocks without depending on their type."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
    result = "".join(parts).strip()
    if not result:
        raise LlmClientError("Anthropic returned an empty JSON completion")
    return result
