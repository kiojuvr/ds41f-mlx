#!/usr/bin/env python3
"""M0.5 short performance runner through the oMLX production server.

This starts `omlx serve` from the local known-good venv, sends one streaming
OpenAI-compatible chat request, and records token arrival timing plus memory and
swap telemetry.  It uses the production server/generation topology instead of
the diagnostic direct `generate_step` path.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_short_perf_omlx import system_memory_snapshot, delta


def request_json(method: str, url: str, payload: dict | None = None, timeout: float = 10) -> tuple[int, dict]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    if payload is not None:
        req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = {"raw": body}
        return exc.code, parsed


def stream_chat(url: str, payload: dict, timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST")
    req.add_header("content-type", "application/json")
    req.add_header("accept", "text/event-stream")
    started = time.perf_counter()
    token_events: list[dict[str, Any]] = []
    usage = None
    finish_reason = None
    text_parts: list[str] = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except Exception:
                continue
            if chunk.get("usage"):
                usage = chunk.get("usage")
            for choice in chunk.get("choices", []) or []:
                if choice.get("finish_reason"):
                    finish_reason = choice.get("finish_reason")
                delta_obj = choice.get("delta") or {}
                content = delta_obj.get("content")
                if content:
                    now = time.perf_counter()
                    text_parts.append(content)
                    token_events.append({"index": len(token_events) + 1, "seconds_from_start": now - started, "content": content})
    ended = time.perf_counter()
    return {
        "seconds": ended - started,
        "token_events": token_events,
        "usage": usage,
        "finish_reason": finish_reason,
        "text": "".join(text_parts),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="ping")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash")
    ap.add_argument("--model-dir", default="/Volumes/KIOXIA-PRO-1/models/deepseek-ai")
    ap.add_argument("--port", type=int, default=18081)
    ap.add_argument("--timeout", type=float, default=1200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--omlx-bin", default=str(Path.home() / ".venvs" / "omlx-0.7.0.dev2" / "bin" / "omlx"))
    ap.add_argument("--base-path", default="artifacts/m0_5/omlx-server-base")
    ap.add_argument(
        "--model-settings-source",
        default=None,
        help="Optional existing model_settings.json to copy into --base-path instead of writing the minimal M0.5 settings",
    )
    args = ap.parse_args()

    out = Path(args.out or f"artifacts/m0_5/short-perf-omlx-server/run-{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    base_path = Path(args.base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    model_settings_path = base_path / "model_settings.json"
    if args.model_settings_source:
        shutil.copyfile(args.model_settings_source, model_settings_path)
    else:
        model_settings_path.write_text(json.dumps({
            "version": 1,
            "models": {
                args.model: {
                    "deepseek_v41_engram_ssd_offload": True,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": max(args.max_tokens, 128),
                }
            },
        }, indent=2, sort_keys=True) + "\n")
    log_path = out.with_suffix(out.suffix + ".server.log")
    prompt = "\n".join([args.prompt] * max(1, args.repeat))
    base_url = f"http://127.0.0.1:{args.port}"

    system_before = system_memory_snapshot()
    log = log_path.open("w", buffering=1)
    cmd = [
        args.omlx_bin,
        "serve",
        "--model-dir",
        args.model_dir,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--base-path",
        str(base_path),
        "--memory-guard",
        "off",
        "--no-cache",
        "--log-level",
        "info",
    ]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    record: dict[str, Any] = {
        "schema": "ds41f.m0_5.short-perf-omlx-server.v1",
        "command": cmd,
        "model": args.model,
        "model_dir": args.model_dir,
        "base_path": str(base_path),
        "model_settings_path": str(model_settings_path),
        "deepseek_v41_engram_ssd_offload": True,
        "model_settings_source": args.model_settings_source,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "prompt_repeat": args.repeat,
        "server_log": str(log_path),
    }
    try:
        ready_start = time.perf_counter()
        for _ in range(600):
            try:
                status, _ = request_json("GET", base_url + "/health", timeout=2)
                if status == 200:
                    break
            except Exception:
                pass
            if proc.poll() is not None:
                raise RuntimeError(f"omlx server exited early with {proc.poll()}; see {log_path}")
            time.sleep(0.5)
        else:
            raise RuntimeError("omlx server did not become ready")
        record["server_ready_seconds"] = time.perf_counter() - ready_start

        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        stream = stream_chat(base_url + "/v1/chat/completions", payload, args.timeout)
        usage = stream.get("usage") or {}
        events = stream["token_events"]
        first = events[0]["seconds_from_start"] if events else None
        last = events[-1]["seconds_from_start"] if events else None
        completion_tokens = int(usage.get("completion_tokens") or len(events))
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        decode_seconds = (last - first) if first is not None and last is not None and len(events) > 1 else None
        usage_generation_tps = usage.get("generation_tokens_per_second") if isinstance(usage, dict) else None
        usage_generation_duration = usage.get("generation_duration") if isinstance(usage, dict) else None
        usage_prompt_tps = usage.get("prompt_tokens_per_second") if isinstance(usage, dict) else None
        usage_prompt_duration = usage.get("prompt_eval_duration") if isinstance(usage, dict) else None
        usage_ttft = usage.get("time_to_first_token") if isinstance(usage, dict) else None
        if isinstance(usage_generation_duration, (int, float)) and usage_generation_duration > 0:
            decode_seconds = float(usage_generation_duration)
        if isinstance(usage_prompt_duration, (int, float)) and usage_prompt_duration > 0:
            first = float(usage_prompt_duration)
        system_after = system_memory_snapshot()
        record.update(
            ok=True,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=usage.get("total_tokens"),
            ttft_seconds=usage_ttft if isinstance(usage_ttft, (int, float)) else first,
            stream_seconds=stream["seconds"],
            decode_seconds=decode_seconds,
            decode_tokens_per_second=float(usage_generation_tps) if isinstance(usage_generation_tps, (int, float)) else ((completion_tokens / decode_seconds) if decode_seconds and decode_seconds > 0 else None),
            decode_tpt_seconds=(1.0 / float(usage_generation_tps)) if isinstance(usage_generation_tps, (int, float)) and usage_generation_tps > 0 else ((decode_seconds / completion_tokens) if decode_seconds and completion_tokens else None),
            prefill_seconds=float(usage_prompt_duration) if isinstance(usage_prompt_duration, (int, float)) else first,
            prefill_tokens_per_second=float(usage_prompt_tps) if isinstance(usage_prompt_tps, (int, float)) else ((prompt_tokens / first) if first and prompt_tokens else None),
            token_events=events,
            text=stream.get("text"),
            finish_reason=stream.get("finish_reason"),
            usage=usage,
            system_memory_before=system_before,
            system_memory_after=system_after,
            swap_delta_bytes=(system_after.get("swap_used_bytes") - system_before.get("swap_used_bytes")) if isinstance(system_after.get("swap_used_bytes"), int) and isinstance(system_before.get("swap_used_bytes"), int) else None,
            swapins_delta=delta(system_after, system_before, "swapins"),
            swapouts_delta=delta(system_after, system_before, "swapouts"),
            compressions_delta=delta(system_after, system_before, "compressions"),
            decompressions_delta=delta(system_after, system_before, "decompressions"),
        )
    except BaseException as exc:
        record.update(ok=False, error_type=type(exc).__name__, error=str(exc))
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        raise
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
            proc.wait()
        log.close()

    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
