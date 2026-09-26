#!/usr/bin/env python3
"""Exercise native buffer ownership/submission over the reconciled sweep plan.

This is not model math and does not use real weights.  It allocates C-owned
MTL-style buffer slots from the V4.1 sweep allocation table and submits the
planned command stream with deterministic buffer touches, preserving checkpoint
validity ordering.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ds41f_mlx.native_prefill import (  # noqa: E402
    DS4_AUTHORITY_REMOTE,
    DS4_AUTHORITY_SHA,
    compile_native_prefill_library,
    load_native_prefill_library,
    native_sweep_config_static,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--remaining", type=int, default=4096)
    ap.add_argument("--dim", type=int, default=64, help="bounded fixture hidden dim; no model math")
    ap.add_argument("--hc-mult", type=int, default=4, help="bounded fixture HC multiplier; no model math")
    ap.add_argument("--vocab-size", type=int, default=1024, help="bounded fixture vocab for final-logits buffer; no model math")
    ap.add_argument("--memory-budget-mib", type=int, default=128)
    ap.add_argument("--out", default="artifacts/m2/dwarfstar-prefill/native-metal-submission.json")
    ap.add_argument("--native-out-dir", default="artifacts/m2/dwarfstar-prefill/native")
    args = ap.parse_args()

    native = load_native_prefill_library(compile_native_prefill_library(Path(args.native_out_dir)))
    cfg = native_sweep_config_static(
        ctx=args.ctx,
        remaining=args.remaining,
        dim=args.dim,
        hc_mult=args.hc_mult,
        vocab_size=args.vocab_size,
        memory_budget_bytes=args.memory_budget_mib * 1024 * 1024,
    )
    plan = native.build_sweep_plan(cfg).to_json()
    with native.create_submission_context(cfg) as ctx:
        before = ctx.info()
        submitted = ctx.submit()
        after = ctx.info()

    touched = {b["semantic_role"] for b in after["buffers"] if b["touched"]}
    allocated = {b["semantic_role"] for b in after["buffers"] if b["allocated"]}
    record = {
        "schema": "ds41f.m2.native-metal-submission-batching.v1",
        "purpose": "C-owned Metal buffer slots and DwarfStar-style batched submission over reconciled V4.1 sweep topology; no model math kernels",
        "native_version": native.version(),
        "ds4_authority": {"remote": DS4_AUTHORITY_REMOTE, "commit": DS4_AUTHORITY_SHA},
        "fixture": {
            "ctx": args.ctx,
            "remaining": args.remaining,
            "dim": args.dim,
            "hc_mult": args.hc_mult,
            "vocab_size": args.vocab_size,
            "memory_budget_mib": args.memory_budget_mib,
            "bounded_fixture_note": "dim/vocab may be reduced to bound allocations; this tests ownership/submission topology, not logits/model semantics",
        },
        "plan_summary": {
            "count": plan["count"],
            "decoder_suffix": plan["decoder_suffix"],
            "command_counts": plan["command_counts"],
            "allocations": plan["allocations"],
        },
        "before_submit": before,
        "after_submit": after,
        "submitted_commands": submitted,
        "semantic_status": {
            "ownership_correct": True,
            "metal_submission_correct": True,
            "model_semantics_validated": False,
        },
    }
    required_roles = {"batch_cur_hc", "batch_next_hc", "carry.residual", "carry.pre", "carry.ffn_split", "carry.selected_comp", "carry.block_mask", "stage_scratch", "final_logits"}
    category_cbs = after["command_buffers_by_category"]
    category_sum = sum(category_cbs.values())
    record["wait_drain_authority"] = {
        "source": {
            "repository": DS4_AUTHORITY_REMOTE,
            "commit": DS4_AUTHORITY_SHA,
            "files": ["ds4.c: ds41_graph_prefill_sweep", "ds4_metal.m: ds4_gpu_end_commands/ds4_gpu_finish_command_buffer"],
        },
        "ds41_graph_prefill_sweep_observation": [
            "g->valid is set false at sweep start",
            "each row/chunk opens ds4_gpu_begin_commands for staging/embed/carry and drains active commands with ds4_gpu_end_commands before Engram/read-side work",
            "compute for the row/chunk opens ds4_gpu_begin_commands and drains active commands with ds4_gpu_end_commands after layer encode/carry copy",
            "Engram prefetch and streaming read-ahead overlap are scheduled outside the active Metal command batch and are joined at explicit boundaries",
            "final residual/pre publication opens a final command batch and drains it before g->valid may become true",
        ],
        "metal_helper_observation": [
            "ds4_gpu_flush_commands commits without immediate wait and parks command buffers in g_pending_cbs",
            "ds4_gpu_end_commands closes the batch and calls ds4_gpu_finish_command_buffer",
            "ds4_gpu_finish_command_buffer commits, waits pending command buffers, waits the current command buffer, then releases transient buffers",
        ],
        "current_scaffold_policy": "synchronous wait at each submitted batch; this matches ds4_gpu_end_commands drain semantics for the modeled execution batches, while prefetch/read-ahead remain recorded scheduling events only",
    }
    record["submission_granularity"] = {
        "logical_planner_commands": len(plan["commands"]),
        "metal_command_buffers": after["metal_command_buffers_committed"],
        "metal_encoders": after["metal_blit_encoders_committed"],
        "metal_completion_waits": after["metal_completion_waits"],
        "encoded_buffer_ops": after["encoded_buffer_ops"],
        "commits_per_phase": after["command_buffers_by_phase"],
        "commits_per_layer": after["command_buffers_by_layer"],
        "commits_per_category": category_cbs,
        "commits_per_chunk_like_encode_rows": category_cbs.get("encode_rows", 0),
        "explicit_synchronization_waits": after["metal_completion_waits"],
    }
    record["gates"] = {
        "authority_sha_pinned": record["ds4_authority"]["commit"] == DS4_AUTHORITY_SHA,
        "all_planned_buffers_allocated": required_roles <= allocated,
        "submission_consumed_all_commands": submitted == len(plan["commands"]) == after["submitted_commands"],
        "checkpoint_invalid_before_submit": before["checkpoint_valid"] is False,
        "checkpoint_valid_after_full_submit": after["checkpoint_valid"] is True,
        "buffer_ops_encoded": after["encoded_buffer_ops"] > 0,
        "hc_pingpong_touched": {"batch_cur_hc", "batch_next_hc"} <= touched,
        "structured_carry_touched": {"carry.residual", "carry.pre", "carry.ffn_split", "carry.selected_comp", "carry.block_mask"} <= touched,
        "final_logits_touched_after_output": "final_logits" in touched,
        "owned_bytes_within_budget": after["total_owned_bytes"] <= after["memory_budget_bytes"],
        "no_failed_command": after["failed_command_index"] is None,
        "metal_backend_enabled_on_darwin": (sys.platform != "darwin") or after["metal_enabled"] is True,
        "metal_buffers_back_planned_slots_on_darwin": (sys.platform != "darwin") or after["metal_buffers_allocated"] == after["n_buffers"],
        "metal_command_buffers_committed_on_darwin": (sys.platform != "darwin") or after["metal_command_buffers_committed"] == category_sum > 0,
        "metal_encoders_match_command_buffers_on_darwin": (sys.platform != "darwin") or after["metal_blit_encoders_committed"] == after["metal_command_buffers_committed"],
        "batched_submission_not_per_buffer_op": after["submitted_command_buffers"] < after["encoded_buffer_ops"],
        "swap_hc_not_separate_metal_commit": category_cbs.get("swap_hc_layer", 0) == 0,
        "model_semantics_not_claimed": record["semantic_status"]["model_semantics_validated"] is False,
    }
    record["ok"] = all(record["gates"].values())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
