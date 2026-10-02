"""Request-level public API policies that must run before session mutation."""
from __future__ import annotations

import json
from typing import Any


def user_stop_present(body: bytes | bytearray | str | dict[str, Any]) -> bool:
    """Return True when a request explicitly asks for protocol stop strings.

    The stateful session API rejects these before recipe conversion/session lookup
    because arbitrary detokenized stop-string truncation is not cache-consistent
    with the committed GenerationBatch token/KV history.

    Missing ``stop`` and JSON ``null`` are accepted. Empty strings/lists are
    treated as absent to match common OpenAI-compatible client defaults.
    """

    if isinstance(body, dict):
        payload = body
    else:
        raw = body.decode() if isinstance(body, (bytes, bytearray)) else str(body)
        if not raw.strip():
            return False
        payload = json.loads(raw)
    if not isinstance(payload, dict) or "stop" not in payload:
        return False
    stop = payload.get("stop")
    if stop is None:
        return False
    if stop == "":
        return False
    if isinstance(stop, (list, tuple)) and len(stop) == 0:
        return False
    return True


def validate_stateful_chat_request_policy(body: bytes | bytearray | str | dict[str, Any]) -> None:
    """Raise ValueError if the stateful Chat Completions policy rejects a request."""

    if user_stop_present(body):
        raise ValueError(
            "stateful Chat Completions sessions do not support request stop strings; "
            "omit 'stop' and rely on DeepSeek EOS/length termination, or use a stateless endpoint"
        )
