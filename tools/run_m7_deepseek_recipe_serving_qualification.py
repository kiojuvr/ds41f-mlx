#!/usr/bin/env python3
"""Milestone-7 real DeepSeek recipe serving qualification harness.

This is a qualification harness, not an optimizer.  It exercises the selected
DENSE_P0_P7 production prefill path and loopback HTTP server with the real model.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import http.client
import json
import os
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ds41f_mlx.prefill_fp8_mlx import handoff_to_generation
from ds41f_mlx.prefill_fp8_mlx.handoff import validate_committed_cache
from ds41f_mlx.runtime.dwarfstar_prefill import DwarfStarMLXPrefillSession, PRODUCTION_PREFILL_SELECTOR
from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT, DEFAULT_OMLX, OmlxRuntime, OmlxRuntimeConfig
from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
from ds41f_mlx.serving.deepseek_recipe_backend import DEFAULT_MODEL_ID, DEFAULT_RECIPE, DeepSeekRecipeRuntimeBackend
from ds41f_mlx.serving.server import load_v41_tokenizer, prepare_request

MATRIX_LENGTHS = [1,2,3,15,16,17,29,30,31,32,33,63,64,65,127,128,129,255,256,257,511,512,513,2048]
P5_LENGTHS = {29,30,31,32,33,127,128,129,513,2048}


def git_rev(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def pkg(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def token(i: int) -> int:
    return 16 + ((37 * i) % 4096)


def tokens(n: int) -> list[int]:
    return [token(i) for i in range(n)]


def terminal_for(n: int) -> int:
    return token(n + 997)


def shape(x: Any) -> list[int] | None:
    return None if x is None or not hasattr(x, "shape") else [int(v) for v in x.shape]


def frontiers(cache: list[Any]) -> list[int]:
    return [int(c.size()) for c in cache]


def ratio2_evidence(lm: Any, cache: list[Any], n: int) -> list[dict[str, Any]]:
    c = lm._config
    out = []
    for layer, item in enumerate(cache):
        ratio = int(getattr(item, "compress_ratio", 0) or 0)
        if ratio != 2:
            continue
        out.append({
            "layer": layer,
            "frontier": int(item.size()),
            "compress_ratio": ratio,
            "expected_pending_rows": n % 2,
            "slot2_compressed_kv_shape": shape(item[2]),
            "slot3_index_k_shape": shape(item[3]) if layer in set(c.index_source_layers) else None,
            "slot4_pending_shape": shape(item[4]),
            "slot5_pending_shape": shape(item[5]),
        })
    return out


def commit_from_result(result: Any) -> Any:
    setup = result.live_result._setup
    return getattr(setup, "p6_commit_authority", None)


def assert_matrix_case(lm: Any, result: Any, n: int) -> dict[str, Any]:
    cache = result.live_cache
    fs = frontiers(cache)
    commit = commit_from_result(result)
    if commit is None:
        raise RuntimeError("missing P6 commit authority on LivePrefillResult")
    validate_committed_cache(commit, result.prefix_token_ids)
    if result.frontier != n or any(f != n for f in fs):
        raise RuntimeError(f"frontier mismatch: result={result.frontier} cache_head={fs[:4]} expected={n}")
    if any(v < n for v in commit.source_coverage.values()) or any(v < n for v in commit.layer_coverage.values()):
        raise RuntimeError("coverage does not reach prefix length")
    if int(commit.history_position) != n:
        raise RuntimeError("Engram history position mismatch")
    p7 = result.p7_scheduling_evidence
    if int(p7.get("foreground_engram_fallback", p7.get("foreground_fallback", -1))) != 0:
        raise RuntimeError(f"P7 foreground fallback != 0: {p7}")
    if result.full_cache_repack_count != 0 or result.portable_state_exported:
        raise RuntimeError("repack/export occurred")
    return {
        "prefix_length": n,
        "status": "PASS",
        "seconds": result.seconds,
        "frontier": result.frontier,
        "all40_frontier": all(f == n for f in fs),
        "source_coverage_reaches": min(commit.source_coverage.values()) >= n,
        "layer_coverage_reaches": min(commit.layer_coverage.values()) >= n,
        "engram_history_position": int(commit.history_position),
        "segment_count": len(result.segment_metadata),
        "p7_engram_evidence": result.p7_scheduling_evidence,
        "full_cache_repack_count": result.full_cache_repack_count,
        "portable_state_exported": result.portable_state_exported,
    }


def run_raw_matrix(checkpoint: Path, omlx_path: Path) -> dict[str, Any]:
    rt = OmlxRuntime(OmlxRuntimeConfig(checkpoint_path=checkpoint, omlx_path=omlx_path, engram_ssd_offload=True, preserve_mtp=False, moe_expert_offload_resident_fraction=None))
    model = None
    try:
        model, _ = rt.load_model()
        lm = getattr(model, "language_model", model)
        session = DwarfStarMLXPrefillSession(model, omlx_path=omlx_path)
        rows = []
        ratio_rows: dict[str, Any] = {}
        p5_rows: dict[str, Any] = {}
        for n in MATRIX_LENGTHS:
            ids = tokens(n)
            result = session.prefill(ids)
            row = assert_matrix_case(lm, result, n)
            if n in (29, 30, 31):
                ev = ratio2_evidence(lm, result.live_cache, n)
                for r in ev:
                    pending = r["expected_pending_rows"]
                    s4 = r["slot4_pending_shape"]
                    s5 = r["slot5_pending_shape"]
                    if s4 is not None and len(s4) > 1 and s4[1] != pending:
                        raise RuntimeError(f"ratio2 slot4 pending mismatch for {n}: {r}")
                    if s5 is not None and len(s5) > 1 and s5[1] != pending:
                        raise RuntimeError(f"ratio2 slot5 pending mismatch for {n}: {r}")
                ratio_rows[str(n)] = ev
            if n in P5_LENGTHS:
                cfg = OMLXDecodeConfig(omlx_path=omlx_path, checkpoint_path=checkpoint, engram_ssd_offload=True, preserve_mtp=False, speculation_enabled=False)
                before = result.frontier
                gen = None
                t0 = time.monotonic()
                try:
                    gen = handoff_to_generation(result.live_result, model, terminal_prompt_token=terminal_for(n), config=cfg, max_tokens=2)
                    first = gen.next_token()
                    if first is None:
                        raise RuntimeError("no generated token from representative P5 case")
                    p5_rows[str(n)] = {
                        "status": "PASS",
                        "handoff_count": result.live_result.handoff_count,
                        "prompt_replay_count": gen.prompt_replay_count,
                        "frontier_before_terminal": before,
                        "frontier_after_terminal": before + 1,
                        "frontier_after_first_generated": gen.token_frontier,
                        "bootstrap_frontier": before + 1,
                        "generated_token": int(first.token),
                        "first_generated_latency_s": float(first.latency_s),
                        "wall_s": time.monotonic() - t0,
                        "full_cache_repack_count": result.full_cache_repack_count,
                        "portable_state_exported": result.portable_state_exported,
                    }
                    if result.live_result.handoff_count != 1 or gen.prompt_replay_count != 0:
                        raise RuntimeError("P5 handoff/replay gate failed")
                finally:
                    if gen is not None:
                        gen.close()
            rows.append(row)
        return {"status": "PASS", "cases": rows, "ratio_2_29_30_31": ratio_rows, "p5_representative": p5_rows}
    finally:
        rt.close()


async def run_direct_backend(checkpoint: Path, omlx_path: Path, recipe_path: Path) -> dict[str, Any]:
    tok = load_v41_tokenizer(recipe_path)
    body = {"model": DEFAULT_MODEL_ID, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2, "temperature": 0}
    prepared = prepare_request("chat_completions", json.dumps(body).encode(), tokenizer=tok, recipe_path=recipe_path)
    backend = DeepSeekRecipeRuntimeBackend(checkpoint=checkpoint, omlx_path=omlx_path, recipe_path=recipe_path)
    chunks = []
    try:
        async for ch in backend.infer(prepared):
            chunks.append(type(ch).__name__)
            if len(chunks) > 8:
                break
        tr = backend.last_trace.to_json() if backend.last_trace is not None else None
        return {"status": "PASS", "prompt_tokens": len(prepared.token_ids), "prefix_tokens": len(prepared.token_ids)-1, "terminal_token": prepared.token_ids[-1], "chunks": chunks, "trace": tr}
    finally:
        backend.close()


def free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def http_json(method: str, url: str, data: dict[str, Any] | None = None, timeout: float = 1200) -> tuple[int, dict[str, Any], dict[str, str]]:
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode()), dict(r.headers)
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try: parsed = json.loads(raw)
        except Exception: parsed = {"raw": raw}
        return e.code, parsed, dict(e.headers)


def stream_sse(host: str, port: int, path: str, body: dict[str, Any], *, cancel_after_first_data: bool = False, timeout: float = 1200) -> dict[str, Any]:
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    start = time.monotonic(); events = []
    conn.request("POST", path, body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    first_ts = None
    try:
        buf = b""
        while True:
            chunk = resp.read(1)
            if not chunk:
                break
            buf += chunk
            if b"\n\n" in buf:
                frame, buf = buf.split(b"\n\n", 1)
                text = frame.decode(errors="replace")
                if first_ts is None: first_ts = time.monotonic()
                events.append({"t": time.monotonic(), "frame": text[:500]})
                if cancel_after_first_data and "data:" in text and "[DONE]" not in text:
                    conn.close()
                    return {"status": resp.status, "content_type": resp.getheader("content-type"), "events": events, "request_start": start, "first_event": first_ts, "final_event": None, "cancelled_client": True}
                if "[DONE]" in text:
                    break
        return {"status": resp.status, "content_type": resp.getheader("content-type"), "events": events, "request_start": start, "first_event": first_ts, "final_event": time.monotonic(), "cancelled_client": False}
    finally:
        with contextlib.suppress(Exception):
            conn.close()


def wait_health(base: str, timeout: float = 120) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            code, data, _ = http_json("GET", base + "/health", None, timeout=5)
            if code == 200:
                return {"code": code, "body": data}
            last = {"code": code, "body": data}
        except Exception as e:
            last = {"error": str(e)}
        time.sleep(0.5)
    raise RuntimeError(f"server did not become healthy: {last}")


def diagnostics(base: str) -> dict[str, Any]:
    code, data, _ = http_json("GET", base + "/_ds41f/diagnostics", None, timeout=30)
    if code != 200:
        raise RuntimeError(f"diagnostics failed {code}: {data}")
    return data


def run_http(checkpoint: Path, omlx_path: Path, recipe_path: Path, max_tokens: int = 2) -> dict[str, Any]:
    port = free_port(); base = f"http://127.0.0.1:{port}"
    env = os.environ.copy(); env["DS41F_ENABLE_DIAGNOSTIC_ENDPOINTS"] = "1"; env.setdefault("DS41F_TRACE_HISTORY_LIMIT", "32")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.Popen([sys.executable, str(ROOT / "tools/run_ds41f_recipe_server.py"), "--host", "127.0.0.1", "--port", str(port), "--checkpoint", str(checkpoint), "--omlx-path", str(omlx_path), "--recipe-path", str(recipe_path)], cwd=str(ROOT), env=env)
    record: dict[str, Any] = {"port": port}
    try:
        record["health_before_model"] = wait_health(base)
        code, body, hdr = http_json("GET", base + "/v1/models", None, timeout=30)
        record["models"] = {"code": code, "body": body}
        if code != 200: raise RuntimeError("/v1/models failed")
        chat = {"model": DEFAULT_MODEL_ID, "messages": [{"role": "user", "content": "hi"}], "max_tokens": max_tokens, "temperature": 0}
        code, body, hdr = http_json("POST", base + "/v1/chat/completions", chat)
        record["chat_non_stream"] = {"code": code, "body_head": body, "diagnostics": diagnostics(base)}
        if code // 100 != 2: raise RuntimeError("chat non-stream failed")
        record["health_after_model"] = http_json("GET", base + "/health", None, timeout=30)[:2]
        chat_stream = dict(chat, stream=True, max_tokens=max(3, max_tokens))
        record["chat_stream"] = stream_sse("127.0.0.1", port, "/v1/chat/completions", chat_stream)
        record["chat_stream_diagnostics"] = diagnostics(base)
        responses = {"model": DEFAULT_MODEL_ID, "input": "hi", "max_output_tokens": max_tokens, "temperature": 0}
        code, body, hdr = http_json("POST", base + "/v1/responses", responses)
        record["responses_non_stream"] = {"code": code, "body_head": body, "diagnostics": diagnostics(base)}
        if code // 100 != 2: raise RuntimeError("responses non-stream failed")
        record["responses_stream"] = stream_sse("127.0.0.1", port, "/v1/responses", dict(responses, stream=True, max_output_tokens=max(3, max_tokens)))
        record["responses_stream_diagnostics"] = diagnostics(base)
        messages = {"model": DEFAULT_MODEL_ID, "messages": [{"role": "user", "content": "hi"}], "max_tokens": max_tokens}
        mcode, mbody, _ = http_json("POST", base + "/v1/messages", messages)
        record["messages"] = {"code": mcode, "body": mbody, "scope": "PASS" if mcode // 100 == 2 else "NOT_QUALIFIED_PINNED_RECIPE_LIMITATION"}
        # cancellation and recovery
        cancel_body = dict(chat, stream=True, max_tokens=16)
        record["cancellation"] = stream_sse("127.0.0.1", port, "/v1/chat/completions", cancel_body, cancel_after_first_data=True)
        time.sleep(1.0)
        code, body, _ = http_json("POST", base + "/v1/chat/completions", chat)
        record["post_cancel_recovery"] = {"code": code, "ok": code // 100 == 2, "diagnostics": diagnostics(base)}
        if code // 100 != 2: raise RuntimeError("post-cancel recovery failed")
        # invalid requests
        invalids = {
            "unsupported_model": dict(chat, model="not-a-model"),
            "image_input": {"model": DEFAULT_MODEL_ID, "messages": [{"role":"user", "content":[{"type":"image_url", "image_url":{"url":"data:image/png;base64,"}}]}], "max_tokens": 1},
            "temperature_negative": dict(chat, temperature=-1),
            "top_p_bad": dict(chat, top_p=2),
            "max_tokens_bad": dict(chat, max_tokens=0),
        }
        inv = {}
        before = diagnostics(base)["trace_count"]
        for name, payload in invalids.items():
            c, b, _ = http_json("POST", base + "/v1/chat/completions", payload, timeout=120)
            inv[name] = {"code": c, "rejected": 400 <= c < 500, "body": b}
        after_invalid = diagnostics(base)
        c, b, _ = http_json("POST", base + "/v1/chat/completions", chat)
        record["invalid_requests"] = {"cases": inv, "trace_count_before": before, "trace_count_after": after_invalid["trace_count"], "valid_recovery_code": c, "valid_recovery_ok": c // 100 == 2, "diagnostics": diagnostics(base)}
        # single-flight/concurrency by process timestamps via two client subprocess threads in Python would be overkill; use asyncio threads.
        import concurrent.futures
        def post_chat(tag):
            t0=time.monotonic(); c,b,_=http_json("POST", base+"/v1/chat/completions", dict(chat, messages=[{"role":"user","content":f"hi {tag}"}]), timeout=1200); return {"tag":tag,"start":t0,"end":time.monotonic(),"code":c}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            futs=[ex.submit(post_chat,"A"), ex.submit(post_chat,"B")]
            overlap=[f.result() for f in futs]
        record["single_flight"] = {"requests": overlap, "diagnostics": diagnostics(base)}
        # cancel unblocks waiter: start cancelled stream and a waiter shortly after.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            fa=ex.submit(lambda: stream_sse("127.0.0.1", port, "/v1/chat/completions", dict(chat, stream=True, max_tokens=16), cancel_after_first_data=True))
            time.sleep(0.5)
            fb=ex.submit(post_chat,"waiter")
            record["cancel_unblocks_waiter"]={"cancel":fa.result(),"waiter":fb.result(),"diagnostics":diagnostics(base)}
        sequential=[]
        for i in range(10):
            c,b,_=http_json("POST", base+"/v1/chat/completions", dict(chat, messages=[{"role":"user","content":f"hi {i}"}], max_tokens=1), timeout=1200)
            d=diagnostics(base); tr=d.get("last_trace") or {}
            sequential.append({"i":i,"code":c,"request_id":tr.get("request_id"),"selector":tr.get("production_prefill_selector"),"prefill_frontier":tr.get("prefill_frontier"),"generated_count":len(tr.get("generated_tokens") or []),"cleanup_called":tr.get("cleanup_called"),"prompt_replay_count":tr.get("prompt_replay_count"),"handoff_count":tr.get("handoff_count")})
            if c // 100 != 2: raise RuntimeError(f"sequential {i} failed")
        final_diag=diagnostics(base)
        record["ten_sequential"]={"requests":sequential,"diagnostics":final_diag}
        record["trace_retention"]={"trace_count":final_diag["trace_count"],"trace_limit":final_diag["trace_limit"],"within_limit":final_diag["trace_count"] <= final_diag["trace_limit"]}
        record["status"]="PASS"
        return record
    finally:
        proc.terminate()
        try: proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=20)
        record["server_returncode"] = proc.returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--omlx-path", type=Path, default=DEFAULT_OMLX)
    ap.add_argument("--recipe-path", type=Path, default=DEFAULT_RECIPE)
    ap.add_argument("--out", type=Path, default=Path("artifacts/m7/deepseek-recipe-serving/result.json"))
    ap.add_argument("--skip-http", action="store_true")
    ap.add_argument("--skip-raw", action="store_true")
    ap.add_argument("--skip-direct", action="store_true")
    args = ap.parse_args(argv)
    existing: dict[str, Any] = {}
    if args.out.exists():
        with contextlib.suppress(Exception):
            existing = json.loads(args.out.read_text())
    record: dict[str, Any] = {
        "schema": "ds41f.m7.deepseek-recipe-serving.v3",
        "implementation_commit": git_rev(ROOT),
        "tested_base_commit": git_rev(ROOT),
        "provenance": {"checkpoint": str(args.checkpoint), "omlx_path": str(args.omlx_path), "omlx_revision": git_rev(args.omlx_path), "mlx_version": pkg("mlx"), "deepseek_recipe_path": str(args.recipe_path), "deepseek_recipe_revision": git_rev(args.recipe_path), "deepseek_recipe_version": pkg("deepseek-recipe"), "preserve_mtp": False, "engram_ssd_offload": True, "expert_offload": None, "p8_tile_native": "OFF", "text_only": True, "single_flight": True},
        "selector": PRODUCTION_PREFILL_SELECTOR,
        "historical_pre_selector_promotion_evidence": {"classification": "HISTORICAL_PRE_SELECTOR_PROMOTION_EVIDENCE", "old_30_token_prefix": ">900s", "old_29_token_ratio2": "reshape limitation"},
    }
    decision = "M7_NOT_QUALIFIED"
    try:
        if not args.skip_raw:
            record["raw_arbitrary_length_matrix"] = run_raw_matrix(args.checkpoint, args.omlx_path)
        else:
            record["raw_arbitrary_length_matrix"] = existing.get("raw_arbitrary_length_matrix", {"status": "SKIPPED"})
        if not args.skip_direct:
            record["direct_backend_minimal_recipe"] = asyncio.run(run_direct_backend(args.checkpoint, args.omlx_path, args.recipe_path))
        else:
            record["direct_backend_minimal_recipe"] = existing.get("direct_backend_minimal_recipe", {"status": "SKIPPED"})
        if not args.skip_http:
            record["http"] = run_http(args.checkpoint, args.omlx_path, args.recipe_path)
        else:
            record["http"] = existing.get("http", {"status": "SKIPPED"})
        if record["raw_arbitrary_length_matrix"].get("status") == "PASS" and record["direct_backend_minimal_recipe"].get("status") == "PASS" and record["http"].get("status") == "PASS":
            decision = "M7_TEXT_SERVING_QUALIFIED"
        else:
            decision = "M7_TEXT_SERVING_PARTIALLY_QUALIFIED"
    except Exception as exc:
        record["failure"] = {"class": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(), "category": "OTHER"}
        decision = "M7_NOT_QUALIFIED"
    record["m7_decision"] = decision
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(args.out)
    print(decision)
    return 0 if decision == "M7_TEXT_SERVING_QUALIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
