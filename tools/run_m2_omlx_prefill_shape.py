#!/usr/bin/env python3
"""Bounded M2 oMLX prefill scaling-shape measurement.

Runs the production oMLX server once, then submits an increasing-prefix sequence
with max_tokens=1.  Prefix cache is enabled so later requests can measure the
new uncached suffix shape (0->4K, 4K->8K, ...), while retaining the M0.5
known-good model settings.  This is an architectural measurement, not
long-context qualification.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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

from tools.run_short_perf_omlx import delta, system_memory_snapshot


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
    usage = None
    finish_reason = None
    text_parts: list[str] = []
    event_count = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":") or not line.startswith("data:"):
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
                content = (choice.get("delta") or {}).get("content")
                if content:
                    event_count += 1
                    text_parts.append(content)
    return {
        "seconds": time.perf_counter() - started,
        "usage": usage or {},
        "finish_reason": finish_reason,
        "text": "".join(text_parts),
        "event_count": event_count,
    }


def load_prefixes(tokenizer_path: str, source: Path, targets: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    text = source.read_text(errors="replace")
    # Repeat only if the local source fixture is shorter than the requested max.
    ids = tokenizer.encode(text, add_special_tokens=False)
    while len(ids) < max(targets):
        text = text + "\n" + text
        ids = tokenizer.encode(text, add_special_tokens=False)
    prefixes = []
    for target in targets:
        content_ids = ids[:target]
        content = tokenizer.decode(content_ids)
        prefixes.append(
            {
                "target_content_tokens": target,
                "content_token_count": len(content_ids),
                "content": content,
            }
        )
    meta = {
        "tokenizer_path": tokenizer_path,
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "source_token_count_initial": len(ids),
    }
    return prefixes, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontiers", default="4096,8192", help="Comma-separated content-token frontiers; use early-stop manually")
    ap.add_argument("--prompt-source", default="/Users/kioju/ds4/speed-bench/promessi_sposi.txt")
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash")
    ap.add_argument("--model-dir", default="/Volumes/KIOXIA-PRO-1/models/deepseek-ai")
    ap.add_argument("--checkpoint", default="/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--port", type=int, default=18082)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out", default=None)
    ap.add_argument("--omlx-bin", default=str(Path.home() / ".venvs" / "omlx-0.7.0.dev2" / "bin" / "omlx"))
    ap.add_argument("--base-path", default="artifacts/m2/omlx-prefill-shape/server-base")
    ap.add_argument("--model-settings-source", default=str(Path.home() / ".omlx" / "model_settings.json"))
    ap.add_argument("--cache-max-size", default="100GB")
    args = ap.parse_args()

    frontiers = [int(x) for x in args.frontiers.split(",") if x.strip()]
    if frontiers != sorted(frontiers) or any(x <= 0 for x in frontiers):
        raise SystemExit("--frontiers must be positive ascending integers")

    out = Path(args.out or f"artifacts/m2/omlx-prefill-shape/run-{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    base_path = Path(args.base_path)
    if base_path.exists():
        shutil.rmtree(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    model_settings_path = base_path / "model_settings.json"
    shutil.copyfile(args.model_settings_source, model_settings_path)
    cache_dir = base_path / "paged-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    prefixes, prompt_meta = load_prefixes(args.checkpoint, Path(args.prompt_source), frontiers)
    for item in prefixes:
        item["content_sha256"] = __import__("hashlib").sha256(item["content"].encode("utf-8")).hexdigest()

    log_path = out.with_suffix(out.suffix + ".server.log")
    base_url = f"http://127.0.0.1:{args.port}"
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
        "--paged-ssd-cache-dir",
        str(cache_dir),
        "--paged-ssd-cache-max-size",
        args.cache_max_size,
        "--log-level",
        "info",
    ]

    system_before = system_memory_snapshot()
    log = log_path.open("w", buffering=1)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    record: dict[str, Any] = {
        "schema": "ds41f.m2.omlx-prefill-scaling-shape.v1",
        "purpose": "bounded architectural measurement; not long-context qualification",
        "command": cmd,
        "model": args.model,
        "model_dir": args.model_dir,
        "checkpoint": args.checkpoint,
        "base_path": str(base_path),
        "cache_dir": str(cache_dir),
        "model_settings_source": args.model_settings_source,
        "model_settings_path": str(model_settings_path),
        "frontiers": frontiers,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "prompt_meta": prompt_meta,
        "server_log": str(log_path),
        "dwarfstar_upper_bound_note": "DwarfStar Q4 resident throughput is an architectural upper bound only; do not precision-adjust or promote from absolute gap alone.",
        "requests": [],
    }
    try:
        ready_start = time.perf_counter()
        for _ in range(900):
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

        prior_prompt_tokens = 0
        for index, item in enumerate(prefixes):
            content = item.pop("content")
            req_before = system_memory_snapshot()
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            stream = stream_chat(base_url + "/v1/chat/completions", payload, args.timeout)
            req_after = system_memory_snapshot()
            usage = stream.get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            cached_tokens = int(details.get("cached_tokens") or usage.get("cached_tokens") or 0)
            uncached_tokens = max(0, prompt_tokens - cached_tokens)
            prefill_seconds = usage.get("prompt_eval_duration")
            prefill_seconds = float(prefill_seconds) if isinstance(prefill_seconds, (int, float)) else None
            added_vs_prior_prompt = max(0, prompt_tokens - prior_prompt_tokens) if index else prompt_tokens
            req_record = {
                **item,
                "request_index": index,
                "prior_prompt_tokens": prior_prompt_tokens,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "uncached_prompt_tokens": uncached_tokens,
                "added_vs_prior_prompt_tokens": added_vs_prior_prompt,
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "prefill_seconds": prefill_seconds,
                "server_prompt_tokens_per_second_total_prompt_basis": usage.get("prompt_tokens_per_second"),
                "uncached_tokens_per_second": (uncached_tokens / prefill_seconds) if prefill_seconds and prefill_seconds > 0 else None,
                "added_vs_prior_tokens_per_second": (added_vs_prior_prompt / prefill_seconds) if prefill_seconds and prefill_seconds > 0 else None,
                "time_to_first_token": usage.get("time_to_first_token"),
                "generation_duration": usage.get("generation_duration"),
                "generation_tokens_per_second": usage.get("generation_tokens_per_second"),
                "stream_seconds": stream.get("seconds"),
                "event_count": stream.get("event_count"),
                "finish_reason": stream.get("finish_reason"),
                "text": stream.get("text"),
                "usage": usage,
                "system_memory_before": req_before,
                "system_memory_after": req_after,
                "swap_delta_bytes": (req_after.get("swap_used_bytes") - req_before.get("swap_used_bytes")) if isinstance(req_after.get("swap_used_bytes"), int) and isinstance(req_before.get("swap_used_bytes"), int) else None,
                "swapins_delta": delta(req_after, req_before, "swapins"),
                "swapouts_delta": delta(req_after, req_before, "swapouts"),
                "compressions_delta": delta(req_after, req_before, "compressions"),
                "decompressions_delta": delta(req_after, req_before, "decompressions"),
            }
            record["requests"].append(req_record)
            out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            print(f"frontier={item['target_content_tokens']} prompt={prompt_tokens} cached={cached_tokens} uncached={uncached_tokens} prefill_s={prefill_seconds} uncached_tps={req_record['uncached_tokens_per_second']}", flush=True)
            prior_prompt_tokens = prompt_tokens

        system_after = system_memory_snapshot()
        record.update(
            ok=True,
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
