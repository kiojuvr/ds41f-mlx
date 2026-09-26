#!/usr/bin/env python3
"""Tiny M2 oMLX-compatibility fixture for explicit V4.1 state publication.

This is not a benchmark and not official DeepSeek correctness evidence.  It
loads the known-good oMLX model, runs a small text-only prompt as two prefill
chunks, and compares:

1. accepted ``LanguageModel._forward`` logits/cache;
2. a copied oMLX-compatibility replay using the existing implicit ``dict`` shared
   state;
3. the same replay using ``ExplicitStatePublication``.

The replay intentionally reuses current oMLX layer, Engram, attention, HC, MoE,
cache, packing, and logits operations.  It only makes the shared-state
publication frontiers observable.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ds41f_mlx.m2_state_publication import (  # noqa: E402
    ExplicitStatePublication,
    build_publication_frontiers,
    cache_digest,
    records_to_json,
    tensor_digest,
)
from ds41f_mlx.runtime.omlx_core import OmlxRuntime, OmlxRuntimeConfig  # noqa: E402


class RecordingDict(dict):
    def __init__(self, frontiers: dict[int, Any]):
        super().__init__()
        self.frontiers = frontiers
        self.records = []

    def publish(self, layer: int, cache: Any, *, has_engram: bool = False) -> None:
        frontier = self.frontiers.get(layer)
        if frontier is None and not has_engram:
            return
        names = frontier.publishes if frontier is not None else ()
        self.records.append(
            {
                "layer": layer,
                "frontier": list(names),
                "shared_keys": sorted(self.keys()),
                "shared": {name: tensor_digest(self.get(name)) for name in names},
                "cache": cache_digest(cache),
                "engram": {"history_slot_6": cache_digest(cache).get("slots", [None] * 7)[6]} if has_engram else None,
            }
        )


def explicit_replay_forward(lm: Any, input_ids: Any, cache: Any, shared_cls: Any) -> tuple[Any, list[Any]]:
    """Single-row text-only replay of oMLX DeepSeek-V4.1 LanguageModel._forward."""

    import mlx.core as mx  # type: ignore
    from contextlib import nullcontext

    lang = importlib.import_module(type(lm).__module__)
    DeepseekV41Cache = importlib.import_module("omlx.patches.deepseek_v41.cache").DeepseekV41Cache
    pack_activation = lang.pack_activation
    hc_pre = lang.hc_pre
    project_logits = lang.project_logits

    c = lm._config
    if input_ids.shape[0] != 1:
        raise ValueError("fixture only supports batch size 1")
    if len(cache) != len(lm.layers):
        raise ValueError("DeepSeek V4.1 cache layer count mismatch")
    for i, item in enumerate(cache):
        ratio = c.compress_ratios[i] if i in c.kv_source_layers else 0
        if item.compress_ratio is None:
            item.compress_ratio = ratio
        elif item.compress_ratio != ratio:
            raise ValueError("DeepSeek V4.1 cache compression layout mismatch")
    if c.engram_layer_ids and lm._hasher is None:
        raise ValueError("DeepSeek V4.1 requires tokenizer-derived Engram token map")

    masks = cache[0].make_mask(input_ids.shape[1])
    valid = np.ones(input_ids.shape[1], bool) if masks is None else np.asarray(masks[0])
    positions = np.flatnonzero(valid)
    if not len(positions):
        raise ValueError("empty fixture chunk")
    begin, end = int(positions[0]), int(positions[-1]) + 1
    if begin != 0 or end != input_ids.shape[1] or end - begin != len(positions):
        raise ValueError("fixture expects one unpadded contiguous row")

    rc = [item.extract(0) for item in cache]
    start = rc[0].size()
    if any(item.size() != start for item in rc):
        raise ValueError("DeepSeek V4.1 cache offsets diverged across layers")

    ids = input_ids[:, begin:end]
    image_mask = None
    h = lm.embed(ids)
    hashes, history = (None, None)
    if lm._hasher is not None:
        hashes, history = lm._hasher(ids, rc[0][6], image_mask)
    h = mx.repeat(h[..., None, :], c.hc_mult, -2)
    pre = mx.broadcast_to((mx.arange(c.hc_mult) == 0).astype(mx.float32), h.shape[:-1])

    frontiers = build_publication_frontiers(c)
    shared = shared_cls(frontiers)
    prefetch = getattr(lm, "_engram_prefetch", None)
    with prefetch.forward() if prefetch is not None else nullcontext():
        if prefetch is not None and c.engram_layer_ids:
            first = lm.layers[c.engram_layer_ids[0]].engram.embed
            prefetch.submit(first, hashes[:, :, 0])
        for i, layer in enumerate(lm.layers):
            has_engram = "engram" in layer
            if has_engram:
                ix = list(c.engram_layer_ids).index(i)
                if prefetch is not None:
                    mx.async_eval(h, pre)
                h = layer.engram(h, hashes[:, :, ix], image_mask)
                if prefetch is not None and ix + 1 < len(c.engram_layer_ids):
                    next_layer = lm.layers[c.engram_layer_ids[ix + 1]]
                    prefetch.submit(next_layer.engram.embed, hashes[:, :, ix + 1])
            h, pre = layer(h, pre, rc[i], shared, start, image_mask)
            if prefetch is not None and has_engram:
                mx.async_eval(h, pre)
            rc[i][0] = mx.array([start + end - begin], mx.int32)
            if history is not None and i == 0:
                rc[i][6] = mx.array(history, mx.int64)
            for slot in range(1, 7):
                if rc[i][slot] is None:
                    width = c.index_head_dim if slot == 3 else c.head_dim
                    empty = mx.zeros((1, 0, width), h.dtype)
                    if slot == 1:
                        empty = pack_activation(empty)
                    elif slot == 2:
                        empty = pack_activation(empty, 4, 16, True)
                    elif slot == 3:
                        empty = pack_activation(empty, 4)
                    rc[i][slot] = mx.zeros((1, 0), mx.int64) if slot == 6 else empty
            shared.publish(i, rc[i], has_engram=has_engram)

    logits = project_logits(lm.norm(hc_pre(h, pre)), lm.head.weight)
    for i, item in enumerate(cache):
        merged = DeepseekV41Cache.merge([rc[i]])
        item.cache = merged.cache
        item.advance(input_ids.shape[1])
    return logits, shared.records


def greedy_id(logits: Any) -> int:
    import mlx.core as mx  # type: ignore

    last = logits[:, -1, :]
    mx.eval(last)
    return int(np.asarray(mx.argmax(last, axis=-1)).reshape(-1)[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="ping\nstate publication fixture\n")
    ap.add_argument("--target-tokens", type=int, default=16)
    ap.add_argument("--chunk-tokens", type=int, default=8)
    ap.add_argument("--out", default="artifacts/m2/state-publication/fixture-2chunk.json")
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"))
    ap.add_argument("--omlx", default=os.environ.get("DS41F_OMLX", str(Path.home() / "omlx-0.7.0.dev2")))
    args = ap.parse_args()

    import mlx.core as mx  # type: ignore

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "schema": "ds41f.m2.state-publication-fixture.v1",
        "purpose": "historical oMLX-compatibility explicit shared-state publication skeleton; not performance; not official qualification",
        "classification": "historical_only_omlx_compatibility_reference_not_official_qualification",
        "checkpoint": args.checkpoint,
        "omlx": args.omlx,
        "target_tokens": args.target_tokens,
        "chunk_tokens": args.chunk_tokens,
    }
    t0 = time.perf_counter()
    rt = OmlxRuntime(OmlxRuntimeConfig(omlx_path=Path(args.omlx), checkpoint_path=Path(args.checkpoint), engram_ssd_offload=True, preserve_mtp=True))
    try:
        model, processor = rt.load_model()
        lm = model.language_model
        record["load_seconds"] = time.perf_counter() - t0
        tokenizer = processor.tokenizer
        text = args.prompt
        ids = tokenizer.encode(text, add_special_tokens=False)
        while len(ids) < args.target_tokens:
            text += args.prompt
            ids = tokenizer.encode(text, add_special_tokens=False)
        ids = ids[: args.target_tokens]
        chunks = [ids[i : i + args.chunk_tokens] for i in range(0, len(ids), args.chunk_tokens)]
        if len(chunks) < 2:
            raise ValueError("fixture requires at least two chunks")
        record["input_token_count"] = len(ids)
        record["chunk_lengths"] = [len(c) for c in chunks]
        record["frontiers"] = {str(k): list(v.publishes) for k, v in build_publication_frontiers(lm._config).items()}

        accepted_cache = lm.make_cache()
        implicit_cache = lm.make_cache()
        explicit_cache = lm.make_cache()
        accepted_logits = implicit_logits = explicit_logits = None
        implicit_records: list[Any] = []
        explicit_records: list[Any] = []
        for chunk in chunks:
            x = mx.array(chunk)[None]
            accepted_logits = lm._forward(x, cache=accepted_cache)
            implicit_logits, rec_i = explicit_replay_forward(lm, x, implicit_cache, RecordingDict)
            explicit_logits, rec_e = explicit_replay_forward(lm, x, explicit_cache, ExplicitStatePublication)
            mx.eval(accepted_logits, implicit_logits, explicit_logits)
            implicit_records.append(rec_i)
            explicit_records.append(records_to_json(rec_e))

        record["accepted_final_greedy"] = greedy_id(accepted_logits)
        record["implicit_final_greedy"] = greedy_id(implicit_logits)
        record["explicit_final_greedy"] = greedy_id(explicit_logits)
        record["final_logits_digest"] = {
            "accepted": tensor_digest(accepted_logits),
            "implicit": tensor_digest(implicit_logits),
            "explicit": tensor_digest(explicit_logits),
        }
        record["final_cache_digest"] = {
            "accepted": [cache_digest(c) for c in accepted_cache],
            "implicit": [cache_digest(c) for c in implicit_cache],
            "explicit": [cache_digest(c) for c in explicit_cache],
        }
        record["publication_records"] = {"implicit": implicit_records, "explicit": explicit_records}
        record["gates"] = {
            "implicit_matches_accepted_token": record["implicit_final_greedy"] == record["accepted_final_greedy"],
            "explicit_matches_accepted_token": record["explicit_final_greedy"] == record["accepted_final_greedy"],
            "implicit_logits_exact": record["final_logits_digest"]["implicit"] == record["final_logits_digest"]["accepted"],
            "explicit_logits_exact": record["final_logits_digest"]["explicit"] == record["final_logits_digest"]["accepted"],
            "explicit_matches_implicit_frontiers": explicit_records == implicit_records,
            "two_chunks_exercised": len(chunks) >= 2,
            "chunk1_to_chunk2_cache_exact": [cache_digest(c) for c in explicit_cache] == [cache_digest(c) for c in implicit_cache],
            "explicit_final_cache_exact": [cache_digest(c) for c in explicit_cache] == [cache_digest(c) for c in accepted_cache],
        }
        record["ok"] = all(record["gates"].values())
    except BaseException as exc:
        record.update(ok=False, error_type=type(exc).__name__, error=str(exc))
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        raise
    finally:
        rt.close()
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
