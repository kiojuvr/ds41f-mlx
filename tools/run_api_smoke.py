#!/usr/bin/env python3
"""Run M0 API smoke checks against the developer server."""

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
OUT = Path(os.environ.get("DS41F_API_SMOKE_OUT", "artifacts/m0/api-smoke")) / time.strftime("run-%Y%m%d-%H%M%S")
PORT = int(os.environ.get("DS41F_SMOKE_PORT", "18080"))
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
        body = e.read().decode("utf-8")
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = {"raw": body}
        return e.code, parsed


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["DS41F_BIND"] = f"127.0.0.1:{PORT}"
    env["DS41F_BACKEND"] = os.environ.get("DS41F_SMOKE_BACKEND", "unconnected")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    log = (OUT / "server.log").open("w")
    proc = subprocess.Popen([sys.executable, "-m", "ds41f_mlx.server"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        for _ in range(100):
            try:
                status, _ = request("GET", "/health", timeout=2)
                if status == 200:
                    break
            except Exception:
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError("server did not become ready")

        smoke_backend = env["DS41F_BACKEND"]
        valid_expected = 501 if smoke_backend == "unconnected" else 200
        valid_name = "valid_unconnected" if smoke_backend == "unconnected" else "valid_connected"
        request_timeout = float(os.environ.get("DS41F_SMOKE_REQUEST_TIMEOUT", "600" if smoke_backend != "unconnected" else "10"))
        cases = [
            ("health", "GET", "/health", None, 200),
            ("models", "GET", "/v1/models", None, 200),
            (valid_name, "POST", "/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}, valid_expected),
            ("unknown_model", "POST", "/v1/chat/completions", {"model": "unknown", "messages": [{"role": "user", "content": "ping"}]}, 404),
            ("empty_messages", "POST", "/v1/chat/completions", {"model": MODEL, "messages": []}, 400),
            ("negative_temperature", "POST", "/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": "ping"}], "temperature": -1}, 400),
            ("zero_max_tokens", "POST", "/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 0}, 400),
        ]
        results = []
        ok = True
        for name, method, path, payload, expected in cases:
            timeout = request_timeout if name.startswith("valid_") else 10
            status, body = request(method, path, payload, timeout=timeout)
            (OUT / f"{name}.json").write_text(json.dumps({"status": status, "body": body}, indent=2, sort_keys=True) + "\n")
            passed = status == expected
            ok = ok and passed
            results.append({"name": name, "status": status, "expected": expected, "passed": passed})
        (OUT / "result.json").write_text(json.dumps({"passed": ok, "results": results}, indent=2, sort_keys=True) + "\n")
        print(OUT)
        return 0 if ok else 1
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
