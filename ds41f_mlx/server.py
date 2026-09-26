"""Minimal M0 developer API server.

This standard-library server is a contract harness, not the final Rust/Axum
server.  It keeps API behavior explicit while the oMLX runtime bridge is being
connected.
"""

from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .backend import MODEL_ID, make_backend

BACKEND = make_backend()


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("content-type", "application/json; charset=utf-8")
    handler.send_header("content-length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _error(status: int, message: str, code: str) -> tuple[int, dict[str, Any]]:
    return status, {"error": {"message": message, "type": code, "code": code}}


def validate_chat_request(data: dict[str, Any]) -> tuple[int, dict[str, Any]] | None:
    model = data.get("model")
    if model != MODEL_ID:
        return _error(404, "model not found", "model_not_found")
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        return _error(400, "messages must be a non-empty array", "invalid_request")
    max_tokens = data.get("max_tokens")
    if max_tokens is not None:
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            return _error(400, "max_tokens must be a positive integer", "invalid_request")
        if max_tokens > 262144:
            return _error(400, "max_tokens exceeds current admission guard", "invalid_request")
    temperature = data.get("temperature")
    if temperature is not None:
        if not isinstance(temperature, (int, float)) or temperature < 0 or temperature != temperature:
            return _error(400, "temperature must be finite and non-negative", "invalid_request")
    return None


class Handler(BaseHTTPRequestHandler):
    server_version = "ds41f-m0/0"

    def log_message(self, fmt: str, *args: Any) -> None:  # keep smoke logs quiet
        if os.environ.get("DS41F_HTTP_LOG"):
            super().log_message(fmt, *args)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            _json_response(
                self,
                200,
                {
                    "status": "ok",
                    "runtime": BACKEND.name,
                    "model": MODEL_ID,
                    "release": False,
                    "time": time.time(),
                },
            )
            return
        if self.path == "/v1/models":
            _json_response(
                self,
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": MODEL_ID,
                            "object": "model",
                            "created": 0,
                            "owned_by": "deepseek-ai",
                        }
                    ],
                },
            )
            return
        status, payload = _error(404, "not found", "not_found")
        _json_response(self, status, payload)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            status, payload = _error(404, "not found", "not_found")
            _json_response(self, status, payload)
            return
        try:
            n = int(self.headers.get("content-length", "0"))
            data = json.loads(self.rfile.read(n).decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("request body must be an object")
        except Exception as exc:
            status, payload = _error(400, f"invalid JSON: {exc}", "invalid_json")
            _json_response(self, status, payload)
            return
        invalid = validate_chat_request(data)
        if invalid is not None:
            status, payload = invalid
            _json_response(self, status, payload)
            return
        result = BACKEND.complete(data)
        _json_response(self, result.status, result.payload)


def main() -> None:
    bind = os.environ.get("DS41F_BIND", "127.0.0.1:8080")
    host, port_s = bind.rsplit(":", 1)
    httpd = ThreadingHTTPServer((host, int(port_s)), Handler)
    print(f"ds41f M0 server listening on http://{bind} model={MODEL_ID} backend={BACKEND.name}", flush=True)
    try:
        httpd.serve_forever()
    finally:
        BACKEND.close()


if __name__ == "__main__":
    main()
