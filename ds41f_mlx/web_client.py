"""HTTP client and tool loop for the local ds41f browser client."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import urllib.error
import urllib.request
from urllib.parse import quote
from typing import Any

from ds41f_mlx.web_tools import ToolRegistry, ToolError


@dataclass
class ChatStep:
    kind: str
    data: dict[str, Any]


@dataclass
class ChatLoopResult:
    session_id: str
    messages: list[dict[str, Any]]
    steps: list[ChatStep] = field(default_factory=list)
    final_response: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {"session_id": self.session_id, "messages": self.messages, "steps": [{"kind": s.kind, "data": s.data} for s in self.steps], "final_response": self.final_response}


class RuntimeHTTPError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"runtime HTTP {status}: {body[:500]}")
        self.status = status; self.body = body


class RuntimeClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def request(self, method: str, path: str, body: dict[str, Any] | None = None, *, timeout: float = 1800) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base_url + path, data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            raise RuntimeHTTPError(exc.code, exc.read().decode("utf-8", "replace")) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise RuntimeHTTPError(503, str(exc)) from exc
        return json.loads(raw) if raw else {}

    def internal_fenced_request(self, session_id: str, body: bytes, sequence: int):
        """Exact-byte fenced transport; reused by the explicit local MTP helper.

        Not used by the public browser loop.

        The caller owns/ closes the returned response even on interrupted reads.
        No serialization, retry, or sequence allocation occurs here.
        """
        import http.client
        from urllib.parse import urlsplit, quote
        url = urlsplit(self.base_url)
        if url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost') or url.path:
            raise ValueError('internal recovery requires loopback HTTP without a base path')
        conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=1800)
        try:
            conn.request('POST', f'/v1/sessions/{quote(session_id, safe="")}/chat/completions',
                         body, {'Content-Type': 'application/json',
                                'X-DS41F-Request-Sequence': str(sequence)})
            response = conn.getresponse()
            if response.status != 200:
                raise RuntimeHTTPError(response.status, response.read().decode('utf-8'))
            return conn, response
        except BaseException:
            conn.close()
            raise

    def health(self) -> dict[str, Any]:
        return self.request("GET", "/health", timeout=5)

    def create_session(self, session_id: str | None = None) -> dict[str, Any]:
        return self.request("POST", "/v1/sessions", {} if session_id is None else {"id": session_id}, timeout=30)

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self.request("GET", f"/v1/sessions/{quote(session_id, safe='')}", timeout=10)

    def close_session(self, session_id: str) -> dict[str, Any]:
        return self.request("DELETE", f"/v1/sessions/{quote(session_id, safe='')}", timeout=120)

    def chat(self, session_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", f"/v1/sessions/{quote(session_id, safe='')}/chat/completions", body, timeout=1800)

    def persist(self, session_id: str, artifact_root: str | None = None) -> dict[str, Any]:
        return self.request("POST", f"/v1/sessions/{quote(session_id, safe='')}/persist", {} if artifact_root is None else {"artifact_root": artifact_root}, timeout=300)

    def restore(self, artifact_path: str, session_id: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"artifact_path": artifact_path}
        if session_id: body["id"] = session_id
        return self.request("POST", "/v1/sessions/restore", body, timeout=1800)


def _message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise RuntimeError("runtime response did not contain choices[0]")
    return choices[0].get("message") or {}


def _finish_reason(response: dict[str, Any]) -> str | None:
    choices = response.get("choices") or []
    return choices[0].get("finish_reason") if choices and isinstance(choices[0], dict) else None


def _assistant_protocol_message(message: dict[str, Any]) -> dict[str, Any]:
    out = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("tool_calls"):
        out["tool_calls"] = message["tool_calls"]
    if message.get("reasoning_content"):
        out["reasoning_content"] = message["reasoning_content"]
    return out


class StatefulToolChatClient:
    def __init__(self, runtime: RuntimeClient, registry: ToolRegistry) -> None:
        self.runtime = runtime; self.registry = registry

    def run_turn(self, *, session_id: str, transcript: list[dict[str, Any]], user_message: str, max_tokens: int = 512, temperature: float = 0.0, tools_enabled: bool = True, max_tool_rounds: int = 4) -> ChatLoopResult:
        handshake = getattr(self.runtime, 'health', None)
        if handshake is not None and handshake().get('profile') == 'mtp-singleton-v1':
            raise ToolError('StatefulToolChatClient does not support the explicit local MTP profile')
        messages = list(transcript) + [{"role": "user", "content": user_message}]
        result = ChatLoopResult(session_id=session_id, messages=messages)
        for round_idx in range(max_tool_rounds + 1):
            body: dict[str, Any] = {"model": "deepseek-v4.1-flash", "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": False}
            if tools_enabled:
                body["tools"] = self.registry.declarations
                body["tool_choice"] = "auto"
            response = self.runtime.chat(session_id, body)
            msg = _message(response)
            finish = _finish_reason(response)
            result.steps.append(ChatStep("assistant", {"finish_reason": finish, "message": msg}))
            messages.append(_assistant_protocol_message(msg))
            if finish != "tool_calls":
                result.messages = messages; result.final_response = response
                return result
            calls = msg.get("tool_calls") or []
            if not isinstance(calls, list) or not calls:
                raise ToolError("finish_reason=tool_calls but no tool_calls were present")
            if round_idx >= max_tool_rounds:
                raise ToolError("maximum client tool rounds exceeded")
            tool_messages, displays = self.registry.execute_calls(calls, session_id=session_id)
            messages.extend(tool_messages)
            result.steps.append(ChatStep("tools", {"calls": calls, "results": displays}))
        raise ToolError("unreachable tool loop exit")
