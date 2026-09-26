"""Runtime backends for the M0 developer API server."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = os.environ.get("DS41F_MODEL", "DeepSeek-V4.1-Flash")
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BackendResult:
    status: int
    payload: dict[str, Any]


def _audit_backend_call(backend: str, request: dict[str, Any]) -> None:
    audit = os.environ.get("DS41F_BACKEND_AUDIT_LOG")
    if not audit:
        return
    rec = {
        "time": time.time(),
        "backend": backend,
        "model": request.get("model"),
        "message_count": len(request.get("messages", [])) if isinstance(request.get("messages"), list) else None,
        "max_tokens": request.get("max_tokens"),
    }
    path = Path(audit)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")


class RuntimeBackend:
    name = "unconnected"

    def complete(self, request: dict[str, Any]) -> BackendResult:
        _audit_backend_call(self.name, request)
        return BackendResult(
            501,
            {
                "error": {
                    "message": "runtime backend is not connected",
                    "type": "runtime_unavailable",
                    "code": "runtime_unavailable",
                }
            },
        )

    def close(self) -> None:
        pass


class OmlxWorkerBackend(RuntimeBackend):
    name = "omlx-worker"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        log_dir = Path(os.environ.get("DS41F_WORKER_LOG_DIR", "artifacts/m0/omlx-worker"))
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = log_dir / f"worker-{int(time.time())}.log"
        self._log = self._log_path.open("a", buffering=1)
        default_venv_python = Path.home() / ".venvs" / "omlx-0.7.0.dev2" / "bin" / "python"
        self._python = os.environ.get("DS41F_WORKER_PYTHON") or os.environ.get("DS41F_PYTHON") or (str(default_venv_python) if default_venv_python.exists() else sys.executable)

    def _start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        self._proc = subprocess.Popen(
            [self._python, "-m", "ds41f_mlx.runtime.omlx_worker"],
            cwd=ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
            bufsize=1,
        )

    def complete(self, request: dict[str, Any]) -> BackendResult:
        _audit_backend_call(self.name, request)
        with self._lock:
            self._start()
            assert self._proc is not None and self._proc.stdin is not None and self._proc.stdout is not None
            payload = dict(request)
            payload["op"] = "chat.completions"
            self._proc.stdin.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
            if not line:
                code = self._proc.poll()
                return BackendResult(
                    502,
                    {"error": {"message": f"oMLX worker exited before response: {code}; log={self._log_path}", "type": "backend_error", "code": "backend_error"}},
                )
            try:
                resp = json.loads(line)
            except Exception as exc:
                return BackendResult(502, {"error": {"message": f"invalid worker response: {exc}", "type": "backend_error", "code": "backend_error"}})
            if not resp.get("ok"):
                return BackendResult(
                    500,
                    {"error": {"message": resp.get("error", "oMLX worker error"), "type": resp.get("error_type", "backend_error"), "code": "backend_error"}},
                )
            return BackendResult(200, self._openai_response(resp))

    def _openai_response(self, resp: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex[:24],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": resp.get("text", "")},
                    "finish_reason": resp.get("finish_reason", "stop"),
                }
            ],
            "usage": resp.get("usage"),
            "ds41f_runtime": {
                "backend": self.name,
                "prompt_ids_sha256": hashlib.sha256(json.dumps(resp.get("prompt_ids", []), separators=(",", ":")).encode()).hexdigest(),
                "timing": resp.get("timing"),
                "memory": resp.get("memory"),
                "generated_ids": resp.get("generated_ids"),
            },
        }

    def close(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                assert proc.stdin is not None
                proc.stdin.write('{"op":"shutdown"}\n')
                proc.stdin.flush()
                proc.wait(timeout=5)
            except Exception:
                proc.terminate()
        self._log.close()


def make_backend() -> RuntimeBackend:
    kind = os.environ.get("DS41F_BACKEND", "unconnected")
    if kind in ("", "unconnected"):
        return RuntimeBackend()
    if kind in ("omlx", "omlx-worker"):
        return OmlxWorkerBackend()
    raise ValueError(f"unsupported DS41F_BACKEND={kind!r}")
