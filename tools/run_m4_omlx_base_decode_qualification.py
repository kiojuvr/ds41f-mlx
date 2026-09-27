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
from time import perf_counter
from typing import Any

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    ap.add_argument("--omlx-path", default=os.environ.get("DS41F_OMLX", str(DEFAULT_OMLX)))
    ap.add_argument("--native-out-dir", default="artifacts/m4/omlx-base-decode/native")
    ap.add_argument("--out", default="artifacts/m4/omlx-base-decode/qualification.json")
    ap.add_argument("--prefill-tokens", default="0,3")
    ap.add_argument("--decode-tokens", default="15,16,17,18", help="bounded deterministic target sequence; default crosses ratio-2 group boundaries")
    ap.add_argument("--skip-reference", action="store_true")
    ap.add_argument("--performance-steps", type=int, default=8)
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
    record["first_token"] = first_gate
    record["gates"]["first_token_state"] = bool(first_report.state_changed["all_layer_offsets_advanced_by_one"])

    # Continue on the same cache objects; do not re-admit.
    continuation_reports = []
    for token in decode_tokens[1:]:
        _logits, report = session.decode_one(token)
        continuation_reports.append(report.to_json())
    record["continuation"] = {
        "reports": continuation_reports,
        "final_frontier": session.token_frontier,
        "same_session_cache_objects": True,
        "monotonic_offsets": all(
            all(offset == report["frontier_after"] for offset in report["cache_offsets"])
            for report in continuation_reports
        ),
    }
    expected_frontier = len(prefill_tokens) + len(decode_tokens)
    record["gates"]["multi_token_continuation"] = session.token_frontier == expected_frontier and all(
        all(offset == report["frontier_after"] for offset in report["cache_offsets"]) for report in continuation_reports
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
    failure_before = {"frontier": failure_session.token_frontier, "offsets": list(failure_session.cache_offsets()), "history_shape": list(failure_session.cache[0].cache[6].shape)}
    failure_ok = False
    try:
        failure_session.decode_one(123, inject_failure="after_forward_before_eval")
    except RuntimeError:
        failure_after = {"frontier": failure_session.token_frontier, "offsets": list(failure_session.cache_offsets()), "history_shape": list(failure_session.cache[0].cache[6].shape)}
        failure_ok = failure_before == failure_after
    record["failure_atomicity"] = {"before": failure_before, "after": locals().get("failure_after"), "passed": failure_ok}
    record["gates"]["failure_atomicity"] = failure_ok

    # Fork independence.
    parent_before = {"frontier": session.token_frontier, "offsets": list(session.cache_offsets())}
    child = session.fork()
    _child_logits, child_report = child.decode_one(19)
    parent_after_child = {"frontier": session.token_frontier, "offsets": list(session.cache_offsets())}
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
    reset_offsets = list(session.cache_offsets())
    record["reset"] = {"frontier": session.token_frontier, "offsets": reset_offsets, "cache_layers": len(session.cache)}
    record["gates"]["reset"] = session.token_frontier == 0 and all(x == 0 for x in reset_offsets)

    # Bounded performance observation uses a fresh admitted session because reset deliberately cleared state.
    perf_session = OMLXDecodeSession.from_prefill_state(session.model, state, cfg)
    warm = decode_tokens[0]
    perf_session.decode_one(warm)
    perf_tokens = [decode_tokens[i % len(decode_tokens)] for i in range(max(1, args.performance_steps))]
    p0 = perf_counter()
    perf_session.decode_many(perf_tokens)
    p_s = perf_counter() - p0
    record["performance"] = {
        "measured_tokens": len(perf_tokens),
        "elapsed_s": p_s,
        "tok_per_s": len(perf_tokens) / p_s if p_s > 0 else None,
        "warmup_tokens": 1,
        "frontier_start": state.token_frontier,
        "state_came_from_dwarfstar_prefill_admission": True,
        "speculation_enabled": False,
    }
    record["gates"]["practical_base_execution"] = bool(record["performance"]["tok_per_s"] and record["performance"]["tok_per_s"] > 1.0)

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
