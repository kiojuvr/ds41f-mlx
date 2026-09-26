#!/usr/bin/env python3
"""Run the bounded M1 invalid-request atomicity gate against the thin API server."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("DS41F_M1_ARTIFACTS", "artifacts/m1")) / "api-atomicity" / time.strftime("run-%Y%m%d-%H%M%S")
PORT = int(os.environ.get("DS41F_M1_ATOMICITY_PORT", "18081"))
BASE = f"http://127.0.0.1:{PORT}"
MODEL = os.environ.get("DS41F_MODEL", "DeepSeek-V4.1-Flash")


def request(method: str, path: str, payload: dict | None = None, timeout: float = 10.0) -> tuple[int, dict]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data, method=method)
    if payload is not None:
        req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        try:
            body = json.loads(raw)
        except Exception:
            body = {"raw": raw}
        return e.code, body


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = OUT / "backend-audit.jsonl"
    env = os.environ.copy()
    env["DS41F_BIND"] = f"127.0.0.1:{PORT}"
    env["DS41F_BACKEND"] = "unconnected"
    env["DS41F_BACKEND_AUDIT_LOG"] = str(audit)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    log = (OUT / "server.log").open("w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "ds41f_mlx.server"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        for _ in range(100):
            try:
                status, _ = request("GET", "/health", timeout=1)
                if status == 200:
                    break
            except Exception:
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError("server did not become ready")

        cases = [
            ("valid_unconnected", {"model": MODEL, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}, 501, True),
            ("unknown_model", {"model": "unknown", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}, 404, False),
            ("empty_messages", {"model": MODEL, "messages": [], "max_tokens": 1}, 400, False),
            ("negative_temperature", {"model": MODEL, "messages": [{"role": "user", "content": "ping"}], "temperature": -1}, 400, False),
            ("zero_max_tokens", {"model": MODEL, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 0}, 400, False),
            ("max_tokens_over_guard", {"model": MODEL, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 262145}, 400, False),
        ]
        results = []
        for name, payload, expected, should_invoke in cases:
            before = audit.read_text().count("\n") if audit.exists() else 0
            status, body = request("POST", "/v1/chat/completions", payload)
            after = audit.read_text().count("\n") if audit.exists() else 0
            invoked = after > before
            rec = {"name": name, "status": status, "expected_status": expected, "backend_invoked": invoked, "expected_backend_invoked": should_invoke, "passed": status == expected and invoked == should_invoke}
            (OUT / f"{name}.json").write_text(json.dumps({"request": payload, "response": {"status": status, "body": body}, "atomicity": rec}, indent=2, sort_keys=True) + "\n")
            results.append(rec)
        audit_records = []
        if audit.exists():
            audit_records = [json.loads(line) for line in audit.read_text().splitlines() if line.strip()]
        result = {
            "schema": "ds41f.m1.api-invalid-request-atomicity.v1",
            "passed": all(r["passed"] for r in results),
            "server_backend": "unconnected",
            "audit_log": str(audit),
            "backend_audit_records": audit_records,
            "results": results,
            "scope": "Invalid chat requests must fail before backend invocation; valid request reaches explicit runtime_unavailable backend.",
        }
        (OUT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(OUT)
        return 0 if result["passed"] else 1
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
            proc.wait()
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
