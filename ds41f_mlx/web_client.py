"""Shared HTTP transport for local MTP clients and acceptance tooling.

The historical module name is retained for active non-Web consumers. There is
no application server, conversation manager or tool executor here.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from urllib.parse import quote
from typing import Any

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

    def internal_outcome_request(self, session_id: str, body: bytes, sequence: int):
        return self.internal_fenced_request(session_id, body, sequence, outcome_projection=True)

    def internal_fenced_request(self, session_id: str, body: bytes, sequence: int, *, outcome_projection=False):
        """Exact-byte fenced transport; reused by the explicit local MTP helper.

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
                                'X-DS41F-Request-Sequence': str(sequence),
                                'X-DS41F-Outcome-Projection': 'json' if outcome_projection else 'delivery'})
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
