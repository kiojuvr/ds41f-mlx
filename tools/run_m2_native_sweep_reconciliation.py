#!/usr/bin/env python3
"""Build native C V4.1 sweep topology/lifetime reconciliation artifacts.

No model load and no Metal/model math.  This exercises bounded static planner
fixtures against the pinned ds41_graph_prefill_sweep contract.
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


def summarize(plan: dict) -> dict:
    commands = plan["commands"]
    decoder_rows = [c["rows"] for c in commands if c["kind_name"] == "begin_layer" and c["phase"] == "decoder_suffix"]
    return {
        "count": plan["count"],
        "prefill_cap": plan["prefill_cap"],
        "encoder_chunk": plan["encoder_chunk"],
        "wide": plan["wide"],
        "decoder_suffix": plan["decoder_suffix"],
        "defer_decoder_candidate": plan["defer_decoder_candidate"],
        "encoder_row_layer_work": plan["encoder_row_layer_work"],
        "decoder_suffix_row_layer_work": plan["decoder_suffix_row_layer_work"],
        "total_row_layer_work": plan["total_row_layer_work"],
        "command_counts": plan["command_counts"],
        "decoder_suffix_rows_by_layer": decoder_rows,
        "allocations": len(plan["allocations"]),
        "prefetch_event_count": plan["prefetch_event_count"],
        "checkpoint_transition_count": plan["checkpoint_transition_count"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/m2/dwarfstar-prefill/native-sweep-reconciliation.json")
    ap.add_argument("--native-out-dir", default="artifacts/m2/dwarfstar-prefill/native")
    args = ap.parse_args()

    native = load_native_prefill_library(compile_native_prefill_library(Path(args.native_out_dir)))
    fixtures = [
        ("short_non_wide", dict(ctx=4096, remaining=2048)),
        ("wide_8k", dict(ctx=32768, remaining=8192)),
        ("wide_16k_defer_candidate", dict(ctx=32768, remaining=32768)),
        ("wide_8k_encoder_only", dict(ctx=32768, remaining=8192, encoder_only=True)),
        ("wide_8k_resume_decoder", dict(ctx=32768, remaining=8192, resume_encoder=True)),
    ]
    plans = {}
    for name, kwargs in fixtures:
        cfg = native_sweep_config_static(**kwargs)
        plans[name] = native.build_sweep_plan(cfg).to_json()

    wide8 = plans["wide_8k"]
    short = plans["short_non_wide"]
    enc_only = plans["wide_8k_encoder_only"]
    roles = {a["semantic_role"]: a for a in wide8["allocations"]}
    record = {
        "schema": "ds41f.m2.native-v41-sweep-reconciliation.v1",
        "purpose": "native C planner topology/lifetime reconciliation to pinned ds41_graph_prefill_sweep; no Metal/model math",
        "native_version": native.version(),
        "ds4_authority": {"remote": DS4_AUTHORITY_REMOTE, "commit": DS4_AUTHORITY_SHA},
        "fixtures": {name: {"summary": summarize(plan), "plan": plan} for name, plan in plans.items()},
    }
    record["gates"] = {
        "authority_sha_pinned": record["ds4_authority"]["commit"] == DS4_AUTHORITY_SHA,
        "short_non_wide_no_suffix": short["decoder_suffix"] is False and short["total_row_layer_work"] == 2048 * 40,
        "wide_8k_suffix_rows_match_formula": wide8["encoder_row_layer_work"] == 8192 * 20 and wide8["decoder_suffix_row_layer_work"] == 24150 and wide8["total_row_layer_work"] == 187990,
        "encoder_chunk_policy_8k_uses_4k": wide8["encoder_chunk"] == 4096,
        "structured_carry_rows_explicit": all(k in roles for k in ["carry.residual", "carry.pre", "carry.ffn_split", "carry.selected_comp", "carry.block_mask"]),
        "persistence_classes_explicit": {a["persistence_class"] for a in wide8["allocations"]} >= {"sweep-persistent", "layer-persistent", "deferred-decoder/suffix", "stage-local reusable scratch"},
        "encoder_only_stays_invalid": enc_only["checkpoint_valid_after_sweep"] is False and enc_only["command_counts"].get("encoder_only_complete_still_invalid") == 1,
        "full_sweep_commit_after_logits": wide8["commands"][-3]["kind_name"] == "encode_output_head" and wide8["commands"][-2]["kind_name"] == "read_logits" and wide8["commands"][-1]["kind_name"] == "checkpoint_may_commit",
        "prefetch_and_readahead_recorded": wide8["command_counts"].get("prefetch_engram_table0") == 1 and wide8["command_counts"].get("prefetch_engram_table1") == 1 and wide8["command_counts"].get("ssd_read_ahead", 0) > 0,
        "per_layer_hc_swap_not_per_chunk": wide8["command_counts"].get("swap_hc_after_layer") == 40,
        "deferred_decoder_planning_represented": plans["wide_16k_defer_candidate"]["defer_decoder_candidate"] is True and plans["wide_16k_defer_candidate"]["command_counts"].get("decoder_pending_still_invalid") == 1,
    }
    record["ok"] = all(record["gates"].values())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
