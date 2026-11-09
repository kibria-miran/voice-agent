"""NVIDIA NIM client (OpenAI-compatible API)."""

import time

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from . import config

_client = None


def _get_client():
    global _client
    if _client is None:
        if not config.NVIDIA_API_KEY:
            raise RuntimeError("NVIDIA_API_KEY missing from .env")
        _client = OpenAI(
            base_url=config.NIM_BASE_URL,
            api_key=config.NVIDIA_API_KEY,
            timeout=60.0,
            max_retries=0,  # retries are handled here so backoff is explicit
        )
    return _client


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500
    return False


def chat(messages, tools=None):
    """Send one non-streaming chat completion and return the assistant message."""
    kwargs = dict(
        model=config.NIM_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=config.MAX_TOKENS,
    )
    if tools:
        kwargs.update(tools=tools, tool_choice="auto")

    last_exc = None
    for attempt in range(config.LLM_RETRIES + 1):
        try:
            response = _get_client().chat.completions.create(**kwargs)
            return response.choices[0].message
        except Exception as exc:  # noqa: BLE001 - re-raised below
            last_exc = exc
            if attempt >= config.LLM_RETRIES or not _retryable(exc):
                raise
            time.sleep(config.LLM_RETRY_BASE_S * (2**attempt))
    raise last_exc  # pragma: no cover - loop always returns or raises


def chat_stream(messages, tools=None):
    """Yield content deltas (Phase 3.5 helper).

    Never use this for a turn that may carry tool calls: tool rounds stay
    non-streaming.  The caller is responsible for that decision.
    """
    kwargs = dict(
        model=config.NIM_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=config.MAX_TOKENS,
        stream=True,
    )
    if tools:
        kwargs.update(tools=tools, tool_choice="auto")
    stream = _get_client().chat.completions.create(**kwargs)
    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            yield delta.content


def self_test_tools(schemas):
    """Fail-soft check that the configured model actually calls tools.

    This is a real request, so it also pays the TLS handshake; no separate
    warm-up call is needed.
    """
    try:
        msg = chat(
            [{"role": "user", "content": "What time is it? Use a tool."}],
            schemas,
        )
        if not getattr(msg, "tool_calls", None):
            print(
                "WARNING: model did not call a tool; "
                f"pick a tool-capable NIM_MODEL (current: {config.NIM_MODEL})"
            )
    except Exception as exc:  # noqa: BLE001 - fail-soft by design
        print(f"WARNING: NIM self-test failed: {type(exc).__name__}: {exc}")
