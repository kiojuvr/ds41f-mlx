"""Canonical server acceptance for the ds41f operator launcher."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

from ds41f_mlx.config import load_runtime_config
from ds41f_mlx.provenance import inspect_runtime

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def http_json(method: str, url: str, data: dict[str, Any] | None = None, *, timeout: float = 1800) -> tuple[int, dict[str, Any]]:
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"raw": raw}
        return e.code, parsed


def wait_health(base: str, *, deadline_s: float = 180) -> dict[str, Any]:
    deadline = time.time() + deadline_s
    last: Any = None
    while time.time() < deadline:
        try:
            code, data = http_json("GET", base + "/health", timeout=5)
            last = {"code": code, "data": data}
            if code == 200:
                return last
        except Exception as exc:
            last = repr(exc)
        time.sleep(0.5)
    raise RuntimeError(f"health timeout: {last}")


def chat_body(messages: list[dict[str, Any]], *, max_tokens: int = 32) -> dict[str, Any]:
    return {"model": "deepseek-v4.1-flash", "messages": messages, "temperature": 0, "reasoning_effort": "none", "max_tokens": max_tokens}


def assistant_content(resp: dict[str, Any]) -> str:
    return resp["choices"][0]["message"].get("content") or ""


def run_acceptance(output: Path | None = None) -> dict[str, Any]:
    cfg = load_runtime_config()
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    out_path = (output or (ROOT / "artifacts" / "m18" / f"canonical-server-acceptance-{time.strftime('%Y%m%d-%H%M%S')}.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = out_path.with_suffix(".server.stdout.log")
    stderr_path = out_path.with_suffix(".server.stderr.log")
    started = time.time()
    proc: subprocess.Popen[str] | None = None
    result: dict[str, Any] = {"schema": "ds41f.m18.canonical-server-acceptance.v1", "created_at": started, "base_url": base, "status": "FAILED", "steps": []}
    try:
        env = dict(**__import__("os").environ)
        env["DS41F_HOST"] = "127.0.0.1"; env["DS41F_PORT"] = str(port)
        env.setdefault("DS41F_ENABLE_DIAGNOSTIC_ENDPOINTS", "0")
        with stdout_path.open("w") as so, stderr_path.open("w") as se:
            proc = subprocess.Popen([sys.executable, "-m", "ds41f_mlx.serve", "--host", "127.0.0.1", "--port", str(port)], cwd=ROOT, env=env, text=True, stdout=so, stderr=se)
            health_before = wait_health(base)
            result["steps"].append({"name": "startup_health", "status": "PASS", **health_before})

            code, models = http_json("GET", base + "/v1/models", timeout=30)
            result["steps"].append({"name": "models", "status": "PASS" if code == 200 else "FAIL", "code": code, "data": models})

            code, stateless = http_json("POST", base + "/v1/chat/completions", chat_body([{"role": "user", "content": "Answer with exactly one short sentence: 2+2?"}], max_tokens=32))
            result["steps"].append({"name": "stateless_chat", "status": "PASS" if code == 200 else "FAIL", "code": code, "finish_reason": (stateless.get("choices") or [{}])[0].get("finish_reason"), "content_prefix": assistant_content(stateless)[:120] if code == 200 else None})

            code, ready_health = http_json("GET", base + "/health", timeout=30)
            result["steps"].append({"name": "health_after_model_request", "status": "PASS" if code == 200 and ready_health.get("model_ready") else "FAIL", "code": code, "data": ready_health})

            code, session = http_json("POST", base + "/v1/sessions", {})
            sid = session.get("id")
            result["steps"].append({"name": "create_session", "status": "PASS" if code == 200 and sid else "FAIL", "code": code, "session_id": sid})

            first_messages = [{"role": "user", "content": "Say a short greeting."}]
            code, first = http_json("POST", f"{base}/v1/sessions/{sid}/chat/completions", chat_body(first_messages, max_tokens=32))
            first_text = assistant_content(first) if code == 200 else ""
            result["steps"].append({"name": "stateful_first_turn", "status": "PASS" if code == 200 else "FAIL", "code": code, "finish_reason": (first.get("choices") or [{}])[0].get("finish_reason"), "content_prefix": first_text[:120]})

            second_messages = first_messages + [{"role": "assistant", "content": first_text}, {"role": "user", "content": "Now say goodbye briefly."}]
            code, second = http_json("POST", f"{base}/v1/sessions/{sid}/chat/completions", chat_body(second_messages, max_tokens=32))
            result["steps"].append({"name": "stateful_continuation", "status": "PASS" if code == 200 else "FAIL", "code": code, "finish_reason": (second.get("choices") or [{}])[0].get("finish_reason"), "content_prefix": assistant_content(second)[:120] if code == 200 else None})

            code, closed = http_json("DELETE", f"{base}/v1/sessions/{sid}", timeout=120)
            result["steps"].append({"name": "close_session", "status": "PASS" if code == 200 else "FAIL", "code": code, "state": closed.get("state")})

            proc.terminate()
            try:
                rc = proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill(); rc = proc.wait(timeout=60)
            result["steps"].append({"name": "graceful_shutdown", "status": "PASS" if rc in (0, -15) else "WARNING", "returncode": rc})
            proc = None
        result["status"] = "PASS" if all(s["status"] in {"PASS", "WARNING"} for s in result["steps"]) else "FAILED"
    except Exception as exc:
        result["error"] = repr(exc)
        if proc is not None:
            proc.terminate()
            try: proc.wait(timeout=30)
            except Exception: proc.kill()
    result["duration_s"] = time.time() - started
    result["provenance"] = inspect_runtime(cfg)
    result["server_logs"] = {"stdout": str(stdout_path), "stderr": str(stderr_path)}
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    result["artifact"] = str(out_path)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = run_acceptance(args.output)
    print(json.dumps({"status": result["status"], "artifact": result["artifact"], "duration_s": result["duration_s"]}, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
