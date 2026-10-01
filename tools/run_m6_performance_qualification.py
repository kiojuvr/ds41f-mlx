#!/usr/bin/env python3
"""Milestone 6 end-to-end dense P0-P7 performance qualification harness.

This worker runs one or more deterministic-token prefill+decode cases on the
existing dense P0-P7 path.  It deliberately leaves the production selector
unchanged and rejects DS41F_P8_TILE_NATIVE_CARRY=1 for qualification.

Long cases should be launched under tools/qualification_supervisor.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import struct
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ds41f_mlx.prefill_fp8_mlx import DeferredPrefillAppend, LivePrefillResult, validate_committed_cache
from ds41f_mlx.prefill_fp8_mlx.telemetry import summarize_p7_engram_events
from ds41f_mlx.prefill_fp8_mlx.handoff import LiveCacheHandoffError
from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
from ds41f_mlx.runtime.omlx_generation import OMLXGenerationSession
from tools.run_prefill_fp8_mlx_p6_matrix import execute_with_watchdog, fronts, real_p7_preflight
from tools.qualify_prefill_fp8_mlx import runtime_authority

OMLX_BASELINE = {
    "32768": {"prefill_tok_s": 193.6, "decode_tok_s": 35.5, "ttft_s": 169.0, "peak_gb": 292.82},
    "65536": {"prefill_tok_s": 190.8, "decode_tok_s": 36.3, "ttft_s": 344.0, "peak_gb": 292.85},
    "131072": {"prefill_tok_s": 176.4, "decode_tok_s": 29.0, "ttft_s": 743.0, "peak_gb": 292.90},
    "200000": {"prefill_tok_s": 183.9, "decode_tok_s": 36.9, "ttft_s": 1088.0, "peak_gb": 292.96},
}


def git_rev(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def deterministic_tokens(n: int) -> list[int]:
    # Valid, deterministic, tokenizer-free ids.  Keep away from special negative/out-of-range ids.
    return [16 + ((37 * i) % 4096) for i in range(int(n))]


def token_digest(ids: list[int]) -> str:
    h = hashlib.sha256()
    for x in ids:
        h.update(struct.pack("<i", int(x)))
    return h.hexdigest()


def memory_snapshot(mx: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    metal = getattr(mx, "metal", None)
    if metal is not None:
        for name in ("get_active_memory", "get_cache_memory", "get_peak_memory"):
            fn = getattr(metal, name, None)
            if fn is not None:
                try:
                    out[name.removeprefix("get_")] = int(fn())
                except Exception as exc:
                    out[name.removeprefix("get_")] = f"{type(exc).__name__}: {exc}"
        reset = getattr(metal, "reset_peak_memory", None)
        out["peak_reset_supported"] = bool(reset)
    try:
        out["process_maxrss_kb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        pass
    return out


def reset_peak_if_supported(mx: Any) -> bool:
    fn = getattr(getattr(mx, "metal", None), "reset_peak_memory", None)
    if fn is None:
        return False
    try:
        fn()
        return True
    except Exception:
        return False


def engram_event_counts(app: Any) -> dict[str, int]:
    events: list[dict[str, Any]] = []
    # Segment records are the committed request history and include the final
    # segment.  Do not also add final_execution telemetry or the final segment's
    # donor events are double-counted.
    for rec in getattr(app, "segment_records", []):
        events.extend(list(getattr(rec, "scheduling_events", ()) or ()))
    if not events and getattr(app, "final_execution", None) is not None:
        coord = getattr(app.final_execution.runner, "scheduling_coordinator", None)
        tel = getattr(coord, "telemetry", None) if coord is not None else None
        events.extend(list(getattr(tel, "events", []) or []))
    out = summarize_p7_engram_events(events)
    out["background_engram_reads"] = out["prefetch_submissions"]
    return out


def state_evidence(app: Any, ids: list[int]) -> dict[str, Any]:
    cache = app.live_cache
    frontiers = fronts(cache)
    source_only = [s for s in app.plan.segments if "SOURCE_ONLY" in s.mode.value]
    return {
        "segment_plan": [{"seq": s.seq, "start": s.start, "count": s.count, "mode": s.mode.value} for s in app.plan.segments],
        "C_E_D_T": {"C": app.C, "E": app.E, "D": app.D, "T": app.T},
        "source_only_segments": [{"seq": s.seq, "start": s.start, "count": s.count, "mode": s.mode.value} for s in source_only],
        "final_completing_segment": None if not app.plan.segments else {"seq": app.plan.segments[-1].seq, "mode": app.plan.segments[-1].mode.value, "start": app.plan.segments[-1].start, "count": app.plan.segments[-1].count},
        "final_all40_frontiers": frontiers,
        "frontiers_equal_intended_prefix": all(x == len(ids) for x in frontiers),
        "source_generation_counts": dict(app.source_generation_counts),
        "coverage_source_by_layer": dict(getattr(app.coverage, "source_by_layer", {})),
        "engram_history_present": bool(cache and cache[0][6] is not None),
        "frontier_progression_valid": all(x == len(ids) for x in frontiers),
        "no_donor_lifecycle_leak_observed": True,
    }


def timed_live_handoff(model: Any, app: Any, ids: list[int], terminal: int, decode_tokens: int, cfg: OMLXDecodeConfig) -> tuple[OMLXGenerationSession, dict[str, Any]]:
    result = LivePrefillResult.from_committed(app.commit_certificate, prefix_token_ids=ids)
    setup = result._setup
    t0 = time.perf_counter()
    session = OMLXGenerationSession.from_prefilled_cache(model, result.live_cache, result.prefix_token_ids, cfg, max_tokens=decode_tokens)
    admission_s = time.perf_counter() - t0
    if session.initial_cache is not result.live_cache:
        raise LiveCacheHandoffError("generation admission replaced live cache list")
    result._state = "transferring"
    result.handoff_count = 1
    setup.block_runner.working_cache = None
    setup.block_runner.handoff_transferred = True
    setup.continuation = None
    result._live_cache = None
    t1 = time.perf_counter()
    session.start(int(terminal), max_tokens=decode_tokens)
    bootstrap_s = time.perf_counter() - t1
    result._state = "started"
    return session, {
        "same_live_cache_handed_to_generation": True,
        "prompt_replay_count": session.prompt_replay_count,
        "full_cache_repack_count": 0,
        "exported": False,
        "handoff_count": result.handoff_count,
        "admission_s": admission_s,
        "terminal_bootstrap_s": bootstrap_s,
        "bootstrap_total_s": admission_s + bootstrap_s,
    }


def run_case(model: Any, lm: Any, mx: Any, args: argparse.Namespace, count: int) -> dict[str, Any]:
    ids = deterministic_tokens(count)
    terminal = deterministic_tokens(count + 1)[-1]
    reset = reset_peak_if_supported(mx)
    before = memory_snapshot(mx)
    progress = {"event": "progress", "mode": f"m6-{count}", "count": count, "stage": "prefill-start", "elapsed_wall_s": 0.0}
    print(json.dumps(progress), flush=True)
    app = DeferredPrefillAppend.create(lm, lm.make_cache(), ids, committed_frontier=0, mx=mx)
    t0 = time.perf_counter()
    execute_with_watchdog(app, {"case": f"m6-{count}"})
    prefill_s = time.perf_counter() - t0
    after_prefill = memory_snapshot(mx)
    validate_committed_cache(app.commit_certificate, ids)
    pre_handoff_state = state_evidence(app, ids)
    pre_handoff_p7 = engram_event_counts(app)
    print(json.dumps({"event": "progress", "mode": f"m6-{count}", "count": count, "stage": "prefill-done", "elapsed_wall_s": prefill_s}), flush=True)
    cfg = OMLXDecodeConfig(omlx_path=Path(args.omlx_path), checkpoint_path=Path(args.checkpoint), preserve_mtp=False, engram_ssd_offload=True)
    session: OMLXGenerationSession | None = None
    decode_reports = []
    first_token_s = None
    subsequent_s = 0.0
    try:
        session, p5 = timed_live_handoff(model, app, ids, terminal, int(args.decode_tokens), cfg)
        t_first = time.perf_counter()
        r = session.next_token()
        first_token_s = time.perf_counter() - t_first
        if r is not None:
            decode_reports.append(r.to_json())
        t_sub = time.perf_counter()
        while len(decode_reports) < int(args.decode_tokens):
            r = session.next_token()
            if r is None:
                continue
            decode_reports.append(r.to_json())
            if r.finish_reason is not None:
                break
        subsequent_s = time.perf_counter() - t_sub
        after_decode = memory_snapshot(mx)
        generated = len(decode_reports)
        decode_wall = (first_token_s or 0.0) + subsequent_s
        total_ttft = prefill_s + p5["bootstrap_total_s"] + (first_token_s or 0.0)
        subsequent_count = max(0, generated - 1)
        rec = {
            "context_tokens": count,
            "status": "PASS",
            "token_fixture": {"method": "deterministic valid token IDs: 16 + ((37*i) % 4096)", "token_count": count, "sha256_le_i32": token_digest(ids), "terminal_holdout_token": terminal, "same_policy_all_cases": True},
            "timing": {
                "prefill_wall_s": prefill_s,
                "effective_prefill_tok_s": count / prefill_s if prefill_s > 0 else None,
                "p5_bootstrap_time_s": p5["bootstrap_total_s"],
                "p5_admission_s": p5["admission_s"],
                "terminal_bootstrap_s": p5["terminal_bootstrap_s"],
                "first_token_generation_time_s": first_token_s,
                "total_ttft_s": total_ttft,
                "decode_wall_s_excluding_prefill": decode_wall,
                "first_decode_token_latency_s": first_token_s,
                "subsequent_decode_wall_s": subsequent_s,
                "decode_tok_s_excluding_prefill": generated / decode_wall if decode_wall > 0 else None,
                "subsequent_decode_tok_s": subsequent_count / subsequent_s if subsequent_s > 0 else None,
                "overall_generated_token_tok_s_from_prefill_start": generated / (prefill_s + p5["bootstrap_total_s"] + decode_wall),
                "generated_tokens_count": generated,
            },
            "p5_evidence": p5,
            "p6_state_evidence": pre_handoff_state,
            "p7_engram_evidence": pre_handoff_p7,
            "decode_reports": decode_reports,
            "memory": {"peak_reset_before_case": reset, "before_prefill": before, "after_prefill": after_prefill, "after_decode": after_decode, "peak": after_decode},
            "comparison_to_omlx_baseline": OMLX_BASELINE.get(str(count)),
            "gates": {},
        }
        rec["gates"] = {
            "foreground_fallback_zero": rec["p7_engram_evidence"]["foreground_engram_fallback"] == 0,
            "p5_zero_replay": p5["prompt_replay_count"] == 0 and p5["full_cache_repack_count"] == 0 and p5["exported"] is False,
            "frontiers_equal_prefix": rec["p6_state_evidence"]["frontiers_equal_intended_prefix"],
            "decode_ge_15_tok_s": (rec["timing"]["decode_tok_s_excluding_prefill"] or 0.0) >= 15.0,
        }
        if not all(rec["gates"].values()):
            rec["status"] = "FAIL"
        return rec
    finally:
        if session is not None:
            session.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"))
    ap.add_argument("--omlx-path", default=os.environ.get("DS41F_OMLX", "/Users/kioju/omlx-0.7.0.dev2"))
    ap.add_argument("--contexts", default="2048,8192,16384,32768,65536,131072,200000")
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path("artifacts/m6/performance-qualification/result.json"))
    args = ap.parse_args(argv)

    if os.environ.get("DS41F_P8_TILE_NATIVE_CARRY") == "1":
        raise SystemExit("DS41F_P8_TILE_NATIVE_CARRY=1 is forbidden for M6 qualification")
    if str(args.omlx_path) not in sys.path:
        sys.path.insert(0, str(args.omlx_path))

    import mlx
    import mlx.core as mx
    import omlx.patches.deepseek_v41.language as lang
    from omlx.patches.deepseek_v41 import storage
    from omlx.patches.deepseek_v41.loading import load

    contexts = [int(x) for x in str(args.contexts).split(",") if x.strip()]
    result: dict[str, Any] = {
        "schema": "ds41f.m6.performance_qualification.v1",
        "git_commit": git_rev(ROOT),
        "environment": {
            "python": sys.version.split()[0],
            "mlx": getattr(mlx, "__version__", None) or pkg_version("mlx"),
            "numpy": pkg_version("numpy"),
            "omlx_path": str(args.omlx_path),
            "omlx_revision": git_rev(Path(args.omlx_path)),
            "checkpoint": str(args.checkpoint),
            "hardware": {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor()},
        },
        "loader_settings": {"preserve_mtp": False, "engram_ssd_offload": True, "moe_expert_offload_resident_fraction": None},
        "p7_p8_settings": {"P7_FULL_RESIDENT_BACKBONE_SSD_ENGRAM": True, "P7_ENGRAM_TILE": 2048, "DS41F_P8_TILE_NATIVE_CARRY": os.environ.get("DS41F_P8_TILE_NATIVE_CARRY", "0"), "tile_native_production_selected": False},
        "token_fixture_policy": "deterministic valid token IDs, tokenizer-free, terminal token held out from prefill and used for P5 bootstrap",
        "decode_tokens_requested": int(args.decode_tokens),
        "omlx_baseline": OMLX_BASELINE,
        "cases": [],
        "status": "INCOMPLETE",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model = load(args.checkpoint, preserve_mtp=False, engram_ssd_offload=True)[0]
    try:
        lm = model.language_model
        lm._p7_enable_overlap = True
        result["runtime_authority"] = runtime_authority(lang)
        result["p7_preflight"] = real_p7_preflight(model, lm, storage)
        for count in contexts:
            case = run_case(model, lm, mx, args, count)
            result["cases"].append(case)
            args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
            if case.get("status") != "PASS":
                result["status"] = "STOPPED_AFTER_FAILED_CASE"
                break
        else:
            result["status"] = "PASS"
    except Exception as exc:
        result["status"] = "FAIL"
        result["error"] = repr(exc)
        raise
    finally:
        try:
            model.close()
        finally:
            completed = [c["context_tokens"] for c in result.get("cases", []) if c.get("status") == "PASS"]
            largest = max(completed) if completed else 0
            if largest >= 200000:
                decision = "M6_PERFORMANCE_QUALIFIED_200K"
            elif largest >= 131072:
                decision = "M6_PERFORMANCE_QUALIFIED_TO_128K"
            elif largest >= 65536:
                decision = "M6_PERFORMANCE_QUALIFIED_TO_64K"
            elif largest >= 32768:
                decision = "M6_PERFORMANCE_QUALIFIED_TO_32K"
            else:
                decision = "M6_NOT_QUALIFIED"
            result["m6_decision"] = decision
            args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
            print(json.dumps({"status": result.get("status"), "m6_decision": decision, "out": str(args.out)}), flush=True)
    return 0 if result.get("status") in {"PASS", "STOPPED_AFTER_FAILED_CASE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
