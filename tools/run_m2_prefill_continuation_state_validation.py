#!/usr/bin/env python3
"""Validate that Milestone 2 handoff contains live continuation state.

This is a no-recompute state-boundary validation: it runs the bounded production
prefill once, detaches the committed PrefillContinuationState from executor
scratch, and verifies required first-decode persistent state categories from the
committed state object rather than from prompt re-execution or JSON digests.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ds41f_mlx.dwarfstar_prefill_slice import DwarfStarPrefillVerticalSliceExecutor  # noqa: E402

DEFAULT_CHECKPOINT = "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", DEFAULT_CHECKPOINT))
    ap.add_argument("--native-out-dir", default="artifacts/m2/dwarfstar-prefill/native")
    ap.add_argument("--out", default="artifacts/m2/dwarfstar-prefill/prefill-continuation-state-validation.json")
    args = ap.parse_args()

    executor = DwarfStarPrefillVerticalSliceExecutor.with_compiled_native(Path(args.checkpoint), Path(args.native_out_dir))
    result = executor.run(tokens=[0, 3], layers=40)
    state = result.continuation_state
    if state is None:
        raise RuntimeError("executor did not return live continuation state")
    # Detach ordinary executor/prefill scratch.  The remaining checks use only
    # the committed live state plus its own inventory method.
    del executor
    artifact = result.artifact
    del result

    from ds41f_mlx.official_model_math import OfficialModelMath  # local after prefill to avoid implying recompute
    math = OfficialModelMath(Path(args.checkpoint))
    inventory = state.inventory(math.digest)
    forked = state.fork()
    fork_inventory = forked.inventory(math.digest)
    forked.reset()

    expected_sources = {2, 8, 14, 20}
    expected_topk = {2, 8, 14, 20, 24, 28, 32, 36}
    gates = {
        "prefill_artifact_ok": artifact.get("ok") is True,
        "state_committed": state.committed is True,
        "token_frontier_present": state.token_frontier == 2 and list(state.token_ids.shape) == [1, 2],
        "all_layer_window_kv_present": set(state.window_kv_by_layer) == set(range(40)),
        "compressed_sources_present": set(state.compressed_kv_by_source) == expected_sources,
        "index_sources_present": set(state.index_k_by_source) == expected_sources,
        "candidate_source20_present": set(state.candidates_by_source) == {20},
        "all_topk_generations_present": expected_topk.issubset(set(state.topk_by_generation)),
        "final_field_ownership_correct": state.field_ownership == {"compress_kv": 20, "index_k": 20, "candidates": 20, "topk_idxs": 36},
        "engram_hash_state_present": all(k in state.ngram_hashes for k in ("full_hash", "layer1_hash", "layer14_hash")),
        "shared_publications_are_live_arrays": all(k in state.shared_publications for k in ("compress_kv", "index_k", "candidates", "topk_idxs")) and all(v is not None for v in state.shared_publications.values()),
        "inventory_has_no_missing": inventory["missing"] == [],
        "fork_preserves_inventory_without_recompute": fork_inventory["missing"] == [] and fork_inventory["field_ownership"] == inventory["field_ownership"],
        "reset_clears_fork_only": forked.token_frontier == 0 and state.token_frontier == 2 and state.committed is True,
        "artifact_declares_full_model_prefill": artifact.get("scope", {}).get("full_model_prefill") is True,
        "artifact_handoff_declares_live_state": artifact.get("session_handoff", {}).get("live_state_available") is True,
        "artifact_requires_no_prefill_recompute": artifact.get("session_handoff", {}).get("requires_no_prefill_recompute") is True,
    }
    rec = {
        "schema": "ds41f.m2-prefill-continuation-state-validation.v1",
        "ok": all(gates.values()),
        "classification": "implementation_scoped_no_recompute_prefill_handoff_validation",
        "checkpoint": str(args.checkpoint),
        "validation_policy": "after one bounded full prefill, discard executor scratch and inspect committed PrefillContinuationState only; no prompt/prefill recomputation is used to satisfy inventory gates",
        "continuation_state_type": "ds41f_mlx.prefill_session.PrefillContinuationState",
        "inventory": inventory,
        "fork_inventory": fork_inventory,
        "gates": gates,
        "non_claims": [
            "does not select DwarfStar decode",
            "does not select oMLX decode",
            "does not execute production decode",
            "does not benchmark long-context performance",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out} ok={rec['ok']}")
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
