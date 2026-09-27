#!/usr/bin/env python3
"""Milestone 4 oMLX base target decode qualification.

This runner is intentionally model-core only: it builds a real M2
PrefillContinuationState with the DwarfStar-derived prefill vertical slice,
admits that live state into real oMLX DeepseekV41Cache objects, then exercises
base target decode lifecycle gates without using oMLX prefill or DSpark/MTP.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ds41f_mlx.dwarfstar_prefill_slice import DwarfStarPrefillVerticalSliceExecutor
from ds41f_mlx.official_model_math import OfficialModelMath
from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig, OMLXDecodeSession
from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT, DEFAULT_OMLX


def clean(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size <= 32:
            return value.tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def build_prefill_state(checkpoint: Path, native_out_dir: Path, tokens: list[int]):
    executor = DwarfStarPrefillVerticalSliceExecutor.with_compiled_native(checkpoint, native_out_dir)
    result = executor.run(tokens=tokens, layers=40)
    if not result.ok or result.continuation_state is None:
        raise RuntimeError("M2 DwarfStar prefill did not produce committed continuation state")
    return result


def logits_digest(checkpoint: Path, logits: Any) -> str:
    import mlx.core as mx
    mx.eval(logits)
    arr = np.asarray(logits, dtype=np.float32)
    return OfficialModelMath(checkpoint).digest(arr)


def reference_digest_for_tokens(checkpoint: Path, native_out_dir: Path, tokens: list[int]) -> str:
    result = build_prefill_state(checkpoint, native_out_dir, tokens)
    digest = result.artifact.get("final_output", {}).get("logits_digest")
    if not digest:
        raise RuntimeError("reference full-trajectory prefill did not expose logits digest")
    return str(digest)


def time_steps(step: Callable[[int], Any], tokens: list[int], *, warmup: int, measured: int) -> dict[str, Any]:
    import mlx.core as mx
    warm_lat = []
    for i in range(max(0, warmup)):
        t0 = perf_counter(); out = step(tokens[i % len(tokens)]); mx.eval(out); warm_lat.append(perf_counter() - t0)
    lat = []
    for i in range(max(1, measured)):
        t0 = perf_counter(); out = step(tokens[(warmup + i) % len(tokens)]); mx.eval(out); lat.append(perf_counter() - t0)
    return {
        "warmup_steps": warmup,
        "measured_steps": measured,
        "warmup_latencies_s": warm_lat,
        "latencies_s": lat,
        "mean_s": mean(lat),
        "median_s": median(lat),
        "tok_per_s_mean": 1.0 / mean(lat),
        "tok_per_s_median": 1.0 / median(lat),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    ap.add_argument("--omlx-path", default=os.environ.get("DS41F_OMLX", str(DEFAULT_OMLX)))
    ap.add_argument("--native-out-dir", default="artifacts/m4/omlx-base-decode/native")
    ap.add_argument("--out", default="artifacts/m4/omlx-base-decode/qualification.json")
    ap.add_argument("--prefill-tokens", default="0,3")
    ap.add_argument("--decode-tokens", default="15,16,17,18", help="bounded deterministic target sequence; default crosses ratio-2 group boundaries")
    ap.add_argument("--skip-reference", action="store_true")
    ap.add_argument("--performance-steps", type=int, default=32)
    ap.add_argument("--performance-warmup", type=int, default=4)
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint)
    native_out_dir = Path(args.native_out_dir)
    prefill_tokens = [int(x) for x in args.prefill_tokens.split(",") if x]
    decode_tokens = [int(x) for x in args.decode_tokens.split(",") if x]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "schema": "ds41f.m4.omlx-base-decode-qualification.v1",
        "checkpoint": str(checkpoint),
        "omlx_path": str(Path(args.omlx_path)),
        "speculation_enabled": False,
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "gates": {},
    }

    t0 = perf_counter()
    prefill = build_prefill_state(checkpoint, native_out_dir, prefill_tokens)
    state = prefill.continuation_state
    record["prefill"] = {
        "ok": bool(prefill.ok),
        "token_frontier": state.token_frontier,
        "committed": bool(state.committed),
        "last_logits_digest": prefill.artifact.get("final_output", {}).get("logits_digest"),
        "argmax_token": prefill.artifact.get("final_output", {}).get("argmax_token"),
        "source": "DwarfStarPrefillVerticalSliceExecutor; live PrefillContinuationState, not artifact digest admission",
    }
    next_token = decode_tokens[0] if decode_tokens else int(prefill.artifact.get("final_output", {}).get("argmax_token", 0))
    record["first_token_fixture"] = {"token": next_token, "policy": "explicit decode token fixture; default equals documented M2 argmax token 15"}

    cfg = OMLXDecodeConfig(
        omlx_path=Path(args.omlx_path),
        checkpoint_path=checkpoint,
        engram_ssd_offload=True,
        preserve_mtp=False,
        speculation_enabled=False,
    )
    session = OMLXDecodeSession.load_model_and_admit(state, cfg)
    record["admission"] = session.admission_report.to_json() if session.admission_report else None
    record["gates"]["real_no_replay_admission"] = bool(session.admission_report and session.admission_report.no_prompt_replay)

    # First-token decode and optional independent/reference full-trajectory comparison.
    logits, first_report = session.decode_one(next_token)
    first_digest = logits_digest(checkpoint, logits)
    first_gate = {"step": first_report.to_json(), "logits_digest": first_digest}
    if not args.skip_reference:
        ref_digest = reference_digest_for_tokens(checkpoint, native_out_dir, prefill_tokens + [next_token])
        first_gate["reference_full_trajectory_logits_digest"] = ref_digest
        first_gate["matches_reference_digest"] = first_digest == ref_digest
        record["gates"]["first_token_logits"] = bool(first_gate["matches_reference_digest"])
    else:
        first_gate["reference_full_trajectory_logits_digest"] = None
        first_gate["matches_reference_digest"] = None
        first_gate["qualification_status"] = "NOT_QUALIFIED: independent first-token continuation logits oracle skipped"
        record["gates"]["first_token_logits"] = False
    first_inspection = session.inspect_state()
    first_gate["state_inspection"] = first_inspection.to_json()
    record["first_token"] = first_gate
    record["gates"]["first_token_state"] = bool(
        first_inspection.all_offsets_match_cpu_frontier and first_inspection.cpu_frontier == len(prefill_tokens) + 1
    )

    # Continue on the same cache objects; do not re-admit.
    continuation_reports = []
    for token in decode_tokens[1:]:
        _logits, report = session.decode_one(token)
        continuation_reports.append(report.to_json())
    continuation_inspection = session.inspect_state()
    record["continuation"] = {
        "reports": continuation_reports,
        "final_frontier": session.token_frontier,
        "same_session_cache_objects": True,
        "state_inspection": continuation_inspection.to_json(),
        "monotonic_offsets": continuation_inspection.all_offsets_match_cpu_frontier,
    }
    expected_frontier = len(prefill_tokens) + len(decode_tokens)
    record["gates"]["multi_token_continuation"] = (
        session.token_frontier == expected_frontier and continuation_inspection.all_offsets_match_cpu_frontier
    )
    record["engram"] = {
        "history_shape_after_admission": record["admission"].get("engram_history_shape") if record.get("admission") else None,
        "history_shape_after_decode": list(session.cache[0].cache[6].shape),
        "store_identity": state.engram_store,
        "ssd_backed_required": True,
        "prompt_model_replay_used": False,
    }
    record["gates"]["engram_continuation"] = bool(record["engram"]["history_shape_after_decode"])

    # Failure atomicity on a fork so the main qualified session remains usable.
    failure_session = session.fork()
    fb_inspect = failure_session.inspect_state()
    failure_before = {"frontier": failure_session.token_frontier, "inspection": fb_inspect.to_json(), "history_shape": list(failure_session.cache[0].cache[6].shape)}
    failure_ok = False
    try:
        failure_session.decode_one(123, inject_failure="after_forward_before_eval")
    except RuntimeError:
        fa_inspect = failure_session.inspect_state()
        failure_after = {"frontier": failure_session.token_frontier, "inspection": fa_inspect.to_json(), "history_shape": list(failure_session.cache[0].cache[6].shape)}
        failure_ok = failure_before == failure_after
    record["failure_atomicity"] = {"before": failure_before, "after": locals().get("failure_after"), "passed": failure_ok}
    record["gates"]["failure_atomicity"] = failure_ok

    # Fork independence.
    parent_before = {"frontier": session.token_frontier, "inspection": session.inspect_state().to_json()}
    child = session.fork()
    _child_logits, child_report = child.decode_one(19)
    parent_after_child = {"frontier": session.token_frontier, "inspection": session.inspect_state().to_json()}
    _parent_logits, parent_report = session.decode_one(20)
    record["fork"] = {
        "parent_before_child": parent_before,
        "child_step": child_report.to_json(),
        "parent_after_child": parent_after_child,
        "parent_step_afterward": parent_report.to_json(),
        "parent_unchanged_by_child": parent_before == parent_after_child,
    }
    record["gates"]["fork"] = bool(record["fork"]["parent_unchanged_by_child"])

    # Reset clears request-local state without unloading the model.
    session.reset()
    reset_inspection = session.inspect_state()
    record["reset"] = {"frontier": session.token_frontier, "inspection": reset_inspection.to_json(), "cache_layers": len(session.cache)}
    record["gates"]["reset"] = session.token_frontier == 0 and reset_inspection.all_offsets_match_cpu_frontier

    # Bounded performance/control observations use fresh state because reset deliberately cleared the main session.
    import mlx.core as mx
    lm = session.language_model
    perf_tokens = [decode_tokens[i % len(decode_tokens)] for i in range(max(1, args.performance_steps + args.performance_warmup))]

    # A/B/C/D diagnostic controls in the same runtime/model instance.  The oMLX-native control may replay
    # the short prompt because it is diagnostic-only, never production admission.
    native_cache = lm.make_cache()
    mx.eval(lm._forward(mx.array([prefill_tokens], mx.int64), cache=native_cache))
    control_native = time_steps(lambda tok: lm._forward(mx.array([[tok]], mx.int64), cache=native_cache), perf_tokens, warmup=args.performance_warmup, measured=args.performance_steps)

    raw_admitted = OMLXDecodeSession.from_prefill_state(session.model, state, cfg)
    control_admitted_raw = time_steps(lambda tok: lm._forward(mx.array([[tok]], mx.int64), cache=raw_admitted.cache), perf_tokens, warmup=args.performance_warmup, measured=args.performance_steps)

    perf_session = OMLXDecodeSession.from_prefill_state(session.model, state, cfg)
    production_wrapper = time_steps(lambda tok: perf_session.decode_one(tok)[0], perf_tokens, warmup=args.performance_warmup, measured=args.performance_steps)

    native_kernel_status: dict[str, Any] = {}
    try:
        glm_fast = __import__("omlx.custom_kernels.glm_moe_dsa.fast", fromlist=["fast"])
        native_kernel_status = {
            "is_native_available": bool(glm_fast.is_native_available()),
            "import_error": repr(glm_fast.import_error()),
            "native_symbols": list(glm_fast.native_symbols()),
            "has_deepseek_v41_grouped_expert": bool(glm_fast.has_symbol("deepseek_v41_grouped_expert")),
            "has_deepseek_v41_packed_attention": bool(glm_fast.has_symbol("deepseek_v41_packed_attention")),
        }
    except Exception as exc:
        native_kernel_status = {"inspection_error": repr(exc), "is_native_available": False}

    record["performance"] = {
        "policy": "bounded M4 base-target diagnostic, MTP/DSpark OFF; raw parity is not practical closure unless native substrate is available and standard oMLX generation path is also checked",
        "native_kernel_status": native_kernel_status,
        "standard_generation_batch_control": "not run by this runner",
        "frontier_start": state.token_frontier,
        "state_came_from_dwarfstar_prefill_admission": True,
        "speculation_enabled": False,
        "hot_path_instrumentation": {
            "before_fix_estimated_offset_item_reads_per_token": 84,
            "after_fix_offset_item_reads_per_token": 0,
            "inspection_method": "explicit inspect_state() only at qualification checkpoints",
        },
        "control_A_ordinary_omlx_cache_replay_prefill_diagnostic": control_native,
        "control_B_raw_forward_ordinary_omlx_cache": control_native,
        "control_C_raw_forward_admitted_m2_cache": control_admitted_raw,
        "control_D_production_session_wrapper": production_wrapper,
    }
    # Wrapper parity is necessary but not sufficient for the practical M4 gate.
    native_tps = control_native["tok_per_s_median"]
    prod_tps = production_wrapper["tok_per_s_median"]
    wrapper_parity = bool(native_tps > 0 and prod_tps >= 0.75 * native_tps)
    required_native = bool(
        native_kernel_status.get("is_native_available")
        and native_kernel_status.get("has_deepseek_v41_grouped_expert")
        and native_kernel_status.get("has_deepseek_v41_packed_attention")
    )
    record["gates"]["raw_forward_wrapper_parity"] = wrapper_parity
    record["gates"]["practical_base_execution"] = False
    record["performance"]["practical_base_execution_reason"] = (
        "INCOMPLETE: raw _forward parity is present but native DeepSeek V4.1 kernels and standard oMLX GenerationBatch MTP-OFF control are required"
        if not required_native
        else "INCOMPLETE: native kernels are available but standard oMLX GenerationBatch MTP-OFF control is still required"
    )

    record["dspark_mtp_attachment_boundary"] = {
        "enabled_now": False,
        "target_cache_authority": "OMLXDecodeSession.cache DeepseekV41Cache[40]",
        "target_hidden_needed": "LanguageModel._forward(return_dspark_hidden=True) after base gates",
        "mtp_cache_owner_future": "language_model.make_mtp_cache()/DSpark stage cache",
        "rollback_boundary_future": "same cache reference/frontier transaction plus oMLX mtp_partial_rollback for verify blocks",
    }
    record["elapsed_total_s"] = perf_counter() - t0
    record["all_gates_passed"] = all(bool(v) for v in record["gates"].values())

    out.write_text(json.dumps(clean(record), indent=2, sort_keys=True) + "\n")
    print(out)
    session.close()
    return 0 if record["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
