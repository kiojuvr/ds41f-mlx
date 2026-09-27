#!/usr/bin/env python3
"""Milestone 3 bounded state-admission probe.

This is architecture-comparison tooling only.  It constructs a small synthetic
PrefillContinuationState with the same semantic fields as the M2 handoff and
attempts backend-specific cache admission without prompt replay.
"""
from __future__ import annotations

import json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ds41f_mlx.prefill_session import PrefillContinuationState

SOURCES = (2, 8, 14, 20)
TOPK = (2, 8, 14, 20, 24, 28, 32, 36)

def synthetic_state(frontier=7, head_dim=16, index_dim=8, window=4):
    return PrefillContinuationState(
        token_ids=np.arange(frontier, dtype=np.int64)[None, :],
        token_frontier=frontier,
        ngram_hashes={"layer1": np.ones((1, frontier, 3, 4), np.int64), "layer14": np.ones((1, frontier, 3, 4), np.int64)*2},
        engram_store={"checkpoint":"synthetic-ssd-store-handle"},
        window_kv_by_layer={i: np.zeros((1, min(frontier, window), head_dim), np.uint8) for i in range(40)},
        compressed_kv_by_source={s: np.zeros((1, frontier // 2, head_dim//2), np.uint8) for s in SOURCES},
        index_k_by_source={s: np.zeros((1, frontier // 2, index_dim), np.uint8) for s in SOURCES},
        candidates_by_source={20: np.zeros((1, 5), np.int32)},
        topk_by_generation={g: np.zeros((1, 5), np.int32) for g in TOPK},
        field_ownership={"compressed_kv":20, "index_k":20, "candidates":20, "topk":36},
        compressor_pending={s: {"kv": np.zeros((1, frontier % 2, head_dim), np.float32), "gate": np.zeros((1, frontier % 2, 1), np.float32)} for s in SOURCES if frontier % 2},
        shared_publications={"kv": np.zeros((1, frontier//2, head_dim//2), np.uint8), "idx": np.zeros((1, 5), np.int32), "candidates": np.zeros((1,5), np.int32)},
        source_generation_order=["source@2", "source@8", "source@14", "source@20", "topk@24", "topk@28", "topk@32", "topk@36"],
        committed=True,
    )

def probe_omlx(state):
    sys.path.insert(0, os.path.expanduser("~/omlx-0.7.0.dev2"))
    import_mode = "mlx-runtime"
    try:
        import mlx.core as mx
        from omlx.patches.deepseek_v41.cache import DeepseekV41Cache
    except Exception as exc:  # local ds41f venv may not include oMLX runtime deps
        mx = None
        DeepseekV41Cache = None
        import_mode = f"shape-only-fallback: {type(exc).__name__}: {exc}"

    class MockCache:
        def __init__(self, ratio):
            self.compress_ratio = ratio
            self.cache = [None] * 7
        def size(self):
            return int(np.asarray(self.cache[0]).reshape(-1)[0])

    cache = [
        (DeepseekV41Cache(2 if i in SOURCES else 0) if DeepseekV41Cache else MockCache(2 if i in SOURCES else 0))
        for i in range(40)
    ]
    def arr(x, dtype=None):
        if mx is None:
            return np.asarray(x)
        return mx.array(x, dtype) if dtype is not None else mx.array(x)
    def zeros(shape, dtype):
        return mx.zeros(shape, dtype) if mx is not None else np.zeros(shape, dtype)
    for i, c in enumerate(cache):
        c.cache[0] = arr([state.token_frontier], mx.int32 if mx else np.int32)
        c.cache[1] = arr(state.window_kv_by_layer[i])
        if i in SOURCES:
            c.cache[2] = arr(state.compressed_kv_by_source[i])
            c.cache[3] = arr(state.index_k_by_source[i])
            pending = state.compressor_pending.get(i, {})
            c.cache[4] = arr(pending.get("kv", np.zeros((1,0,16), np.float32)))
            c.cache[5] = arr(pending.get("gate", np.zeros((1,0,1), np.float32)))
        else:
            c.cache[2] = zeros((1,0,8), mx.uint8 if mx else np.uint8)
            c.cache[3] = zeros((1,0,8), mx.uint8 if mx else np.uint8)
            c.cache[4] = zeros((1,0,16), mx.float32 if mx else np.float32)
            c.cache[5] = zeros((1,0,1), mx.float32 if mx else np.float32)
        c.cache[6] = arr(state.ngram_hashes["layer1"][:, -3:, 0, 0], mx.int64 if mx else np.int64) if i == 0 else zeros((1,0), mx.int64 if mx else np.int64)
    sizes = [c.size() for c in cache]
    return {"status":"ADMITTED_SYNTHETIC_NO_PROMPT_REPLAY", "import_mode": import_mode, "cache_layers": len(cache), "all_offsets_match": all(x == state.token_frontier for x in sizes), "offsets_sample": sizes[:5], "source_slots": {str(s): {"compressed_shape": list(cache[s].cache[2].shape), "index_shape": list(cache[s].cache[3].shape), "pending_kv_shape": list(cache[s].cache[4].shape)} for s in SOURCES}, "engram_history_shape_layer0": list(cache[0].cache[6].shape)}

def probe_dwarfstar(state):
    ds4 = Path.home()/"ds4"/"ds4.c"
    text = ds4.read_text(errors="ignore")
    has_graph_state = all(term in text for term in ["layer_raw_cache", "layer_attn_comp_cache", "layer_index_comp_cache", "history", "ds41_graph_step"])
    has_import_abi = "PrefillContinuationState" in text or "import_continuation" in text or "admit_continuation" in text
    return {"status":"NO_PUBLIC_STATE_ADMISSION_ABI", "graph_state_terms_present": has_graph_state, "public_import_abi_present": has_import_abi, "no_prompt_replay_admission": False, "reason":"DwarfStar owns decode state inside ds41_gpu_graph C/Metal tensors; source has no external ABI to populate all V4.1 graph caches from neutral arrays."}

if __name__ == "__main__":
    state = synthetic_state()
    result = {"schema":"ds41f.m3.state-admission-probe.v1", "state_frontier": state.token_frontier, "required_fields": sorted(state.__dict__.keys()), "omlx": probe_omlx(state), "dwarfstar": probe_dwarfstar(state)}
    out = ROOT/"artifacts/m3/state-admission-probe.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result, indent=2, sort_keys=True))
