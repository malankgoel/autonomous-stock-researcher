"""Provider-agnostic LLM completion client for JSON hypothesis proposals.

The SDK imports are intentionally lazy: deterministic generators and tests should
not require OpenAI or Anthropic packages unless the opt-in ``llm`` family is used.
"""

from __future__ import annotations

import os
from typing import Any

# All tunables live here, read from the environment so nothing is hardcoded.
DEFAULT_MAX_OUTPUT_TOKENS = 4096
_VALID_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


class LlmClientError(RuntimeError):
    """Raised when the configured LLM provider cannot complete a request."""


def _max_output_tokens() -> int:
    """Hard cap on generated tokens, from ``LLM_MAX_OUTPUT_TOKENS`` (default 4096)."""
    raw = os.environ.get("LLM_MAX_OUTPUT_TOKENS", "").strip()
    if not raw:
        return DEFAULT_MAX_OUTPUT_TOKENS
    try:
        value = int(raw)
    except ValueError as exc:
        raise LlmClientError("LLM_MAX_OUTPUT_TOKENS must be an integer") from exc
    if value <= 0:
        raise LlmClientError("LLM_MAX_OUTPUT_TOKENS must be positive")
    return value


def _reasoning_effort() -> str | None:
    """Reasoning level from ``LLM_REASONING_EFFORT`` (e.g. ``medium``), or None."""
    raw = os.environ.get("LLM_REASONING_EFFORT", "").strip().lower()
    if not raw:
        return None
    if raw not in _VALID_EFFORTS:
        raise LlmClientError(
            f"LLM_REASONING_EFFORT must be one of {sorted(_VALID_EFFORTS)} or unset"
        )
    return raw


def _temperature() -> float | None:
    """Optional sampling temperature from ``LLM_TEMPERATURE``.

    Reasoning models (GPT-5 family) reject a non-default temperature, so it is only
    sent when explicitly set. Leave unset for those models.
    """
    raw = os.environ.get("LLM_TEMPERATURE", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise LlmClientError("LLM_TEMPERATURE must be a number") from exc


def complete_json(system: str, user: str) -> str:
    """Return raw JSON text from the configured LLM provider.

    Environment:
      - ``LLM_PROVIDER``: ``anthropic`` or ``openai``
      - ``LLM_MODEL``: provider model name (e.g. ``gpt-5.4``, ``claude-opus-4-8``)
      - provider key: ``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY``
      - ``LLM_REASONING_EFFORT`` (optional): none|minimal|low|medium|high|xhigh
      - ``LLM_MAX_OUTPUT_TOKENS`` (optional, default 4096): hard output cap
      - ``LLM_TEMPERATURE`` (optional): only sent when set (omit for reasoning models)
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
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": _max_output_tokens(),
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    temperature = _temperature()
    if temperature is not None:
        kwargs["temperature"] = temperature
    response = client.messages.create(**kwargs)
    return _content_to_text(response.content)


def _complete_openai(model: str, system: str, user: str) -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        raise LlmClientError("OPENAI_API_KEY must be set for LLM_PROVIDER=openai")
    try:
        import openai
    except ImportError as exc:
        raise LlmClientError("install the 'openai' package to use LLM_PROVIDER=openai") from exc

    client = openai.OpenAI()
    kwargs: dict[str, Any] = {
        "model": model,
        "response_format": {"type": "json_object"},
        "max_completion_tokens": _max_output_tokens(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    effort = _reasoning_effort()
    if effort is not None:
        kwargs["reasoning_effort"] = effort
    temperature = _temperature()
    if temperature is not None:
        kwargs["temperature"] = temperature

    try:
        response = client.chat.completions.create(**kwargs)
    except TypeError as exc:
        # Older SDKs predate ``max_completion_tokens``/``reasoning_effort``. If the
        # caller asked for reasoning, fail loudly rather than silently downgrading;
        # otherwise retry once with the legacy ``max_tokens`` parameter.
        if effort is not None:
            raise LlmClientError(
                "installed openai SDK does not support reasoning_effort; upgrade the "
                "'openai' package or unset LLM_REASONING_EFFORT"
            ) from exc
        legacy = dict(kwargs)
        legacy["max_tokens"] = legacy.pop("max_completion_tokens")
        try:
            response = client.chat.completions.create(**legacy)
        except Exception as inner:  # noqa: BLE001
            raise LlmClientError(f"OpenAI request failed: {inner}") from inner
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        finish = getattr(response.choices[0], "finish_reason", None)
        usage = getattr(response, "usage", None)
        hint = ""
        if finish == "length":
            hint = (
                " — finish_reason='length': the model hit LLM_MAX_OUTPUT_TOKENS "
                "before emitting any text. For reasoning models this cap covers "
                "reasoning + output, so raise LLM_MAX_OUTPUT_TOKENS (try 16000-32000) "
                "and/or lower LLM_REASONING_EFFORT."
            )
        raise LlmClientError(
            f"OpenAI returned an empty completion (finish_reason={finish}, usage={usage}){hint}"
        )
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
