#!/usr/bin/env python3
"""M5 practical oMLX base decode production-path qualification.

This runner exercises the promoted no-replay BatchGenerator/GenerationBatch path
from a real DwarfStar-derived PrefillContinuationState.  It is backend-local: it
validates lifecycle, determinism, no replay, and practical throughput without
reopening cross-backend numerical trajectory gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT, DEFAULT_OMLX, OmlxRuntime, OmlxRuntimeConfig
from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
from ds41f_mlx.runtime.omlx_generation import OMLXGenerationSession
from tools.run_m4_omlx_base_decode_qualification import build_prefill_state


def clean(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist() if v.size <= 32 else {"shape": list(v.shape), "dtype": str(v.dtype)}
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    return v


def git_rev(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def mx_digest(x: Any) -> dict[str, Any] | None:
    if x is None:
        return None
    import mlx.core as mx
    mx.eval(x)
    arr = np.asarray(x)
    return {"shape": list(arr.shape), "dtype": str(arr.dtype), "sha256": hashlib.sha256(np.ascontiguousarray(arr).view(np.uint8)).hexdigest()}


def perf_gate(timing: dict[str, Any], threshold: float) -> dict[str, Any]:
    median_tok_s = timing.get("median_tok_s")
    passed = bool(median_tok_s is not None and median_tok_s >= threshold)
    regression = bool(median_tok_s is not None and median_tok_s < 1.0)
    return {
        "threshold_median_tok_s": threshold,
        "passed": passed,
        "singleton_forward_regression_guard": None if not regression else "PRACTICAL_DECODE_SUBSTRATE_REGRESSION_TO_SINGLETON_FORWARD",
    }


def make_session(model: Any, state: Any, cfg: OMLXDecodeConfig, max_tokens: int) -> OMLXGenerationSession:
    return OMLXGenerationSession.from_prefill_state(model, state, cfg, max_tokens=max_tokens)


def run_once(model: Any, state: Any, cfg: OMLXDecodeConfig, *, first_input: int, steps: int) -> tuple[OMLXGenerationSession, dict[str, Any]]:
    session = make_session(model, state, cfg, max_tokens=steps)
    admitted_offsets = session.cache_offsets(session.initial_cache)
    engram_before = mx_digest(session.initial_cache[0].cache[6])
    session.start(first_input, max_tokens=steps)
    reports = session.generate(steps)
    active_offsets = session.active_cache_offsets()
    active_cache = session._final_cache or session._generation_cache()
    engram_after = None if active_cache is None else mx_digest(active_cache[0].cache[6])
    record = {
        "metadata": session.metadata().to_json(),
        "admission": session.admission_report.to_json() if session.admission_report else None,
        "admitted_cache_offsets_before_generationbatch": list(admitted_offsets),
        "prompt_bootstrap_responses": session.prompt_bootstrap_responses,
        "no_replay_evidence": {
            "admitted_frontier_before_GenerationBatch": session.admitted_frontier,
            "number_of_prefix_tokens_recomputed": session.prompt_replay_count,
            "BatchGenerator_insert_prompts": [[first_input]],
            "BatchGenerator_insert_all_tokens": [session.prefix_tokens],
        },
        "token_accounting": {
            "prefix_tokens": session.prefix_tokens,
            "first_decode_input_consumed_by_generationbatch_bootstrap": first_input,
            "generated_tokens": list(session.generated_tokens),
            "step_reports": [r.to_json() for r in reports],
            "frontier_after_bootstrap": session.admitted_frontier + 1,
            "final_frontier": session.token_frontier,
            "frontier_progression_expected_final": session.admitted_frontier + 1 + len(session.generated_tokens),
            "frontier_progression_correct": session.token_frontier == session.admitted_frontier + 1 + len(session.generated_tokens),
        },
        "cache_offsets_after_generation": None if active_offsets is None else list(active_offsets),
        "cache_offset_progression_correct": None if active_offsets is None else all(o == session.token_frontier for o in active_offsets),
        "engram_history": {
            "before_bootstrap": engram_before,
            "after_generation": engram_after,
            "progression_observed": bool(engram_before is not None and engram_after is not None and engram_after["sha256"] != engram_before["sha256"]),
            "note": "Engram slot is rolling history; same shape is acceptable when digest changes with generated-token history.",
        },
        "performance": session.timing_summary(),
    }
    return session, record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    ap.add_argument("--omlx-path", default=os.environ.get("DS41F_OMLX", str(DEFAULT_OMLX)))
    ap.add_argument("--native-out-dir", default="artifacts/m5/practical-omlx-base-decode/native")
    ap.add_argument("--out", default="artifacts/m5/practical-omlx-base-decode/result.json")
    ap.add_argument("--prefill-tokens", default="0,3")
    ap.add_argument("--first-input", type=int, default=15)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--throughput-threshold", type=float, default=15.0)
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint)
    omlx = Path(args.omlx_path)
    if str(omlx) not in sys.path:
        sys.path.insert(0, str(omlx))
    import mlx
    import mlx.core as mx

    prompt = [int(x) for x in args.prefill_tokens.split(",") if x]
    rec: dict[str, Any] = {
        "schema": "ds41f.m5.practical-omlx-base-decode.v1",
        "runtime_provenance": {
            "checkpoint": str(checkpoint),
            "omlx_path": str(omlx),
            "omlx_git_revision": git_rev(omlx),
            "mlx_version": getattr(mlx, "__version__", None) or _pkg_version("mlx"),
            "repository_head": git_rev(ROOT),
        },
        "fixture": {"prefix_tokens": prompt, "first_decode_input": args.first_input, "generated_transactions_requested": args.steps},
        "execution_substrate": "mlx_lm.generate.BatchGenerator.insert(caches=..., all_tokens=...) -> GenerationBatch",
        "mtp_dspark_status": "OFF (preserve_mtp=False; speculation disabled)",
        "admission_source": "DwarfStarPrefillVerticalSliceExecutor live PrefillContinuationState",
        "gates": {},
    }

    cfg = OMLXDecodeConfig(omlx_path=omlx, checkpoint_path=checkpoint, engram_ssd_offload=True, preserve_mtp=False, speculation_enabled=False)
    rt = OmlxRuntime(OmlxRuntimeConfig(omlx_path=omlx, checkpoint_path=checkpoint, engram_ssd_offload=True, preserve_mtp=False))
    model, _processor = rt.load_model()
    try:
        prefill_a = build_prefill_state(checkpoint, Path(args.native_out_dir) / "run_a", prompt)
        prefill_b = build_prefill_state(checkpoint, Path(args.native_out_dir) / "run_b", prompt)
        rec["prefill"] = {
            "run_a_ok": bool(prefill_a.ok),
            "run_b_ok": bool(prefill_b.ok),
            "continuation_state_type": "PrefillContinuationState",
            "committed": bool(prefill_a.continuation_state.committed and prefill_b.continuation_state.committed),
            "token_frontier": int(prefill_a.continuation_state.token_frontier),
            "physical_state_contract": "OMLXDecodeStateAdapter preserves window KV, compressed-KV physical FP4/E4M3 payloads when present, index K, pending compressor state, Engram/token history, and frontier from PrefillContinuationState; no lossy regeneration from semantic arrays for physical compressed state.",
        }
        s1, r1 = run_once(model, prefill_a.continuation_state, cfg, first_input=args.first_input, steps=args.steps)
        s2, r2 = run_once(model, prefill_b.continuation_state, cfg, first_input=args.first_input, steps=args.steps)
        rec["run_a"] = r1
        rec["run_b"] = r2
        seq_a = r1["token_accounting"]["generated_tokens"]
        seq_b = r2["token_accounting"]["generated_tokens"]
        rec["determinism"] = {
            "same_checkpoint_omlx_mlx_mtpoff_prefix_first_input_greedy": True,
            "generated_token_sequence_identical": seq_a == seq_b,
            "run_a_generated_tokens": seq_a,
            "run_b_generated_tokens": seq_b,
            "frontier_progression_identical": r1["token_accounting"]["final_frontier"] == r2["token_accounting"]["final_frontier"],
        }
        # Bounded cancel/stop seam on a fresh admitted state: stop before another token is requested.
        cancel_sess = make_session(model, prefill_a.continuation_state, cfg, max_tokens=args.steps)
        cancel_sess.start(args.first_input, max_tokens=args.steps)
        before_cancel_frontier = cancel_sess.token_frontier
        cancel_sess.stop("bounded_cancel_after_bootstrap_before_next_commit")
        rec["stop_cancel"] = {
            "supported": True,
            "policy": "best-effort BatchGenerator extract_cache; no additional token requested/committed after stop",
            "frontier_before_cancel": before_cancel_frontier,
            "frontier_after_cancel": cancel_sess.token_frontier,
            "stop_reason": cancel_sess.stop_reason,
        }
        rec["reset"] = {"status": "DEFERRED", "seam": "create fresh OMLXGenerationSession from PrefillContinuationState; direct GenerationBatch reset not promoted"}
        rec["fork"] = {"status": "DEFERRED", "seam": "fork PrefillContinuationState or diagnostic OMLXDecodeSession before GenerationBatch start; do not clone live scheduler state yet"}
        perf = perf_gate(r1["performance"], args.throughput_threshold)
        rec["performance_gate"] = perf
        rec["memory"] = {"ru_maxrss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "note": "platform units; representative process max RSS if available cheaply"}
        rec["gates"] = {
            "real_prefill_continuation_state_admitted": bool(prefill_a.continuation_state.committed),
            "no_prompt_replay": r1["no_replay_evidence"]["number_of_prefix_tokens_recomputed"] == 0,
            "generationbatch_owns_execution": r1["metadata"]["execution_substrate"].startswith("mlx_lm.generate.BatchGenerator"),
            "deterministic_bounded_run": rec["determinism"]["generated_token_sequence_identical"],
            "frontier_progression_correct": bool(r1["token_accounting"]["frontier_progression_correct"]),
            "cache_offset_progression_correct": bool(r1["cache_offset_progression_correct"]),
            "engram_history_progression_observed": bool(r1["engram_history"]["progression_observed"]),
            "performance_median_tok_s_ge_threshold": bool(perf["passed"]),
            "not_singleton_forward_regression": perf["singleton_forward_regression_guard"] is None,
        }
        qualified = all(rec["gates"].values())
        rec["classification"] = "PRACTICAL_OMLX_BASE_DECODE_PRODUCTION_PATH_QUALIFIED" if qualified else (perf["singleton_forward_regression_guard"] or "PRACTICAL_OMLX_BASE_DECODE_PRODUCTION_PATH_NOT_QUALIFIED")
        rec["deferred_status"] = {"MTP_DSpark": "PENDING/OFF", "long_session_robustness": "DEFERRED", "KV_restore": "DEFERRED", "API_serving": "PENDING"}
        s1.close(); s2.close(); cancel_sess.close()
    finally:
        rt.close()
        mx.synchronize()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(clean(rec), indent=2, sort_keys=True) + "\n")
    print(out)
    return 0 if rec.get("classification") == "PRACTICAL_OMLX_BASE_DECODE_PRODUCTION_PATH_QUALIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
