#!/usr/bin/env python3
"""Narrow M2 attribution: oMLX production prefill chunk trace.

Uses the existing oMLX admin throughput benchmark because it sets
``benchmark_trace=True`` and causes the scheduler to log per-prefill chunk
model/overhead timings.  This is not qualification and does not modify oMLX.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_short_perf_omlx import delta, system_memory_snapshot
from tools.run_m2_omlx_prefill_shape import request_json

_TRACE_RE = re.compile(
    r"\[benchmark-prefill\].*?chunk_tokens=(?P<chunk>\d+) "
    r"processed=(?P<before>\d+)->(?P<after>\d+) kv_before=(?P<kv>\d+) "
    r"requested_step=(?P<requested>\d+) boundary_enabled=(?P<boundary>\w+) "
    r"cache_block_size=(?P<block>\d+) .*?model_cache_ms=(?P<model>[0-9.]+) "
    r"total_ms=(?P<total>[0-9.]+) overhead_ms=(?P<overhead>[0-9.]+)"
)


def parse_trace(log_path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in log_path.read_text(errors="replace").splitlines():
        m = _TRACE_RE.search(line)
        if not m:
            continue
        d = m.groupdict()
        rows.append(
            {
                "chunk_tokens": int(d["chunk"]),
                "processed_before": int(d["before"]),
                "processed_after": int(d["after"]),
                "kv_before": int(d["kv"]),
                "requested_step": int(d["requested"]),
                "boundary_enabled": d["boundary"] == "True",
                "cache_block_size": int(d["block"]),
                "model_cache_ms": float(d["model"]),
                "total_ms": float(d["total"]),
                "overhead_ms": float(d["overhead"]),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-lengths", default="4096,8192")
    ap.add_argument("--generation-length", type=int, default=1)
    ap.add_argument("--context-profile", default="novel_en")
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash")
    ap.add_argument("--model-dir", default="/Volumes/KIOXIA-PRO-1/models/deepseek-ai")
    ap.add_argument("--port", type=int, default=18083)
    ap.add_argument("--timeout", type=float, default=2400)
    ap.add_argument("--out", default=None)
    ap.add_argument("--omlx-bin", default=str(Path.home() / ".venvs" / "omlx-0.7.0.dev2" / "bin" / "omlx"))
    ap.add_argument("--base-path", default="artifacts/m2/omlx-prefill-trace/server-base")
    ap.add_argument("--model-settings-source", default=str(Path.home() / ".omlx" / "model_settings.json"))
    ap.add_argument("--no-prefix-cache", action="store_true", help="Disable paged prefix cache to test whether 2048 chunking is caused by cache boundary snapshots")
    ap.add_argument("--prefill-step-patch", type=int, default=0, help="Diagnostic-only sitecustomize patch that changes Scheduler._base_prefill_step_size for this isolated server process")
    args = ap.parse_args()

    prompt_lengths = [int(x) for x in args.prompt_lengths.split(",") if x.strip()]
    out = Path(args.out or f"artifacts/m2/omlx-prefill-trace/run-{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    base_path = Path(args.base_path)
    if base_path.exists():
        shutil.rmtree(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.model_settings_source, base_path / "model_settings.json")
    # Enable admin API for this isolated benchmark base only.
    (base_path / "settings.json").write_text(json.dumps({
        "version": "1.0",
        "auth": {"skip_api_key_verification": True},
    }, indent=2) + "\n")
    cache_dir = base_path / "paged-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(out.suffix + ".server.log")
    base_url = f"http://127.0.0.1:{args.port}"
    patch_dir = None
    env = os.environ.copy()
    if args.prefill_step_patch:
        patch_dir = Path(tempfile.mkdtemp(prefix="ds41f-m2-prefill-step-"))
        (patch_dir / "sitecustomize.py").write_text(textwrap.dedent(f"""
            import logging
            try:
                from omlx.scheduler import Scheduler
                def _m2_base_prefill_step_size(self, processed_tokens, remaining_tokens):
                    return {int(args.prefill_step_patch)}
                Scheduler._base_prefill_step_size = _m2_base_prefill_step_size
                logging.getLogger(__name__).warning(
                    "M2 diagnostic patched Scheduler._base_prefill_step_size=%d",
                    {int(args.prefill_step_patch)},
                )
            except Exception:
                logging.getLogger(__name__).exception("M2 diagnostic prefill step patch failed")
        """))
        env["PYTHONPATH"] = str(patch_dir) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [
        args.omlx_bin, "serve",
        "--model-dir", args.model_dir,
        "--host", "127.0.0.1", "--port", str(args.port),
        "--base-path", str(base_path),
        "--memory-guard", "off",
    ]
    if args.no_prefix_cache:
        cmd.append("--no-cache")
    else:
        cmd.extend([
            "--paged-ssd-cache-dir", str(cache_dir),
            "--paged-ssd-cache-max-size", "100GB",
        ])
    cmd.extend(["--log-level", "info"])
    system_before = system_memory_snapshot()
    log = log_path.open("w", buffering=1)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True, env=env)
    record: dict[str, Any] = {
        "schema": "ds41f.m2.omlx-prefill-trace.v1",
        "purpose": "narrow attribution of scheduler chunking/model-vs-overhead; not qualification",
        "command": cmd,
        "prompt_lengths": prompt_lengths,
        "generation_length": args.generation_length,
        "context_profile": args.context_profile,
        "no_prefix_cache": args.no_prefix_cache,
        "prefill_step_patch": args.prefill_step_patch or None,
        "server_log": str(log_path),
    }
    try:
        for _ in range(900):
            try:
                status, _ = request_json("GET", base_url + "/health", timeout=2)
                if status == 200:
                    break
            except Exception:
                pass
            if proc.poll() is not None:
                raise RuntimeError(f"server exited early with {proc.poll()}; see {log_path}")
            time.sleep(0.5)
        else:
            raise RuntimeError("server did not become ready")

        payload = {
            "model_id": args.model,
            "prompt_lengths": prompt_lengths,
            "generation_length": args.generation_length,
            "batch_sizes": [],
            "context_profile": args.context_profile,
            "warmup_mode": "quick",
        }
        status, started = request_json("POST", base_url + "/admin/api/bench/start", payload, timeout=10)
        if status != 200:
            raise RuntimeError(f"benchmark start failed: {status} {started}")
        bench_id = started["bench_id"]
        record["bench_id"] = bench_id
        deadline = time.time() + args.timeout
        result = None
        while time.time() < deadline:
            status, result = request_json("GET", base_url + f"/admin/api/bench/{bench_id}/results", timeout=10)
            if status == 200 and result.get("status") in {"completed", "error", "cancelled"}:
                break
            time.sleep(2)
        else:
            raise RuntimeError("benchmark timed out")
        record["benchmark_result"] = result
        trace = parse_trace(log_path)
        record["trace_chunks"] = trace
        by_request: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for row in trace:
            if row["processed_before"] == 0 and current:
                by_request.append(current)
                current = []
            current.append(row)
        if current:
            by_request.append(current)
        summaries = []
        for chunks in by_request:
            model_ms = sum(x["model_cache_ms"] for x in chunks)
            total_ms = sum(x["total_ms"] for x in chunks)
            overhead_ms = sum(x["overhead_ms"] for x in chunks)
            summaries.append({
                "chunks": [x["chunk_tokens"] for x in chunks],
                "chunk_count": len(chunks),
                "tokens": sum(x["chunk_tokens"] for x in chunks),
                "model_cache_ms_sum": model_ms,
                "total_ms_sum": total_ms,
                "overhead_ms_sum": overhead_ms,
                "overhead_fraction": overhead_ms / total_ms if total_ms > 0 else None,
                "boundary_enabled": chunks[0]["boundary_enabled"] if chunks else None,
                "cache_block_size": chunks[0]["cache_block_size"] if chunks else None,
            })
        record["trace_summaries"] = summaries
        system_after = system_memory_snapshot()
        record.update(
            ok=result is not None and result.get("status") == "completed",
            system_memory_before=system_before,
            system_memory_after=system_after,
            swap_delta_bytes=(system_after.get("swap_used_bytes") - system_before.get("swap_used_bytes")) if isinstance(system_after.get("swap_used_bytes"), int) and isinstance(system_before.get("swap_used_bytes"), int) else None,
            swapins_delta=delta(system_after, system_before, "swapins"),
            swapouts_delta=delta(system_after, system_before, "swapouts"),
        )
    except BaseException as exc:
        record.update(ok=False, error_type=type(exc).__name__, error=str(exc), trace_chunks=parse_trace(log_path) if log_path.exists() else [])
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
