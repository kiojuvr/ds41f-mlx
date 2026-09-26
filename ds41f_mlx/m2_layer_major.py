"""M2 DeepSeek-V4.1 layer-major oMLX-compatibility prototype.

This module contains a historical loop-inversion replay.  It is not a production
scheduler, not a performance implementation, and not official DeepSeek
correctness evidence.  It reuses existing oMLX operations and cache objects
while changing only traversal order for tiny fixtures:

    layer 0: chunk 0, chunk 1, ...
    layer 1: chunk 0, chunk 1, ...

The implementation intentionally keeps per-chunk hidden/pre tensors so that the
state-publication contract can be tested before any carry-buffer or scratch
optimization exists.
"""

from __future__ import annotations

import importlib
from contextlib import nullcontext
from typing import Any

from ds41f_mlx.m2_state_publication import (
    ExplicitStatePublication,
    build_publication_frontiers,
    records_to_json,
)


def fill_deepseek_v41_empty_slots(rc: Any, h: Any, config: Any, pack_activation: Any) -> None:
    """Mirror oMLX V4.1 empty-cache-slot materialization."""

    import mlx.core as mx  # type: ignore

    for slot in range(1, 7):
        if rc[slot] is None:
            width = config.index_head_dim if slot == 3 else config.head_dim
            empty = mx.zeros((1, 0, width), h.dtype)
            if slot == 1:
                empty = pack_activation(empty)
            elif slot == 2:
                empty = pack_activation(empty, 4, 16, True)
            elif slot == 3:
                empty = pack_activation(empty, 4)
            rc[slot] = mx.zeros((1, 0), mx.int64) if slot == 6 else empty


def _eval_layer_state(h_values: list[Any], pre_values: list[Any], layer_cache: Any) -> None:
    """Materialize retained per-chunk state to cut long MLX graph chains."""

    import mlx.core as mx  # type: ignore

    tensors = [*h_values, *pre_values]
    tensors.extend(slot for slot in getattr(layer_cache, "cache", []) if slot is not None)
    if tensors:
        mx.eval(*tensors)


def layer_major_forward(
    lm: Any,
    chunk_ids: list[list[int]],
    cache: Any,
    *,
    final_logits_only: bool = False,
    record_publications: bool = True,
    eval_boundary: str = "none",
    output_mode: str = "concat",
) -> tuple[Any, list[list[dict[str, Any]]]]:
    """Run a tiny text-only layer-major replay over pre-split chunks.

    Returns logits and per-chunk publication records.  Inputs are token-id
    chunks for one row only.  The replay is an oMLX-compatibility diagnostic,
    not official model correctness evidence.  When ``final_logits_only`` is
    true, only the final prompt-position logits are projected; this is
    diagnostic-only because tiny fixtures showed it is not exact against the
    historical accepted oMLX full-chunk projection.  When
    ``record_publications`` is false, frontier digest materialization is skipped
    for bounded timing while preserving the explicit shared-state object and
    model/cache math.  ``eval_boundary`` may be ``none``, ``chunk``, or ``layer``;
    it materializes retained h/pre/cache state to bound MLX graph lifetime
    without changing math.  ``output_mode`` may be ``concat`` or
    ``full_chunks_last``; the latter still projects every chunk at full length
    but returns only the final chunk logits to avoid retaining a full-prompt
    logits tensor.  The function starts from an empty cache because this M2 unit
    tests prefill topology rather than decode/append.
    """

    import mlx.core as mx  # type: ignore

    lang = importlib.import_module(type(lm).__module__)
    DeepseekV41Cache = importlib.import_module("omlx.patches.deepseek_v41.cache").DeepseekV41Cache
    pack_activation = lang.pack_activation
    hc_pre = lang.hc_pre
    project_logits = lang.project_logits

    if eval_boundary not in {"none", "chunk", "layer"}:
        raise ValueError(f"unknown eval_boundary: {eval_boundary}")
    if output_mode not in {"concat", "full_chunks_last"}:
        raise ValueError(f"unknown output_mode: {output_mode}")

    c = lm._config
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

    rc = [item.extract(0) for item in cache]
    if any(item.size() != rc[0].size() for item in rc):
        raise ValueError("DeepSeek V4.1 cache offsets diverged across layers")
    if rc[0].size() != 0:
        raise ValueError("fixture starts from an empty cache")

    ids_by_chunk = [mx.array(ids)[None] for ids in chunk_ids]
    h_by_chunk = []
    pre_by_chunk = []
    for ids in ids_by_chunk:
        h = lm.embed(ids)
        h = mx.repeat(h[..., None, :], c.hc_mult, -2)
        pre = mx.broadcast_to((mx.arange(c.hc_mult) == 0).astype(mx.float32), h.shape[:-1])
        h_by_chunk.append(h)
        pre_by_chunk.append(pre)

    frontiers = build_publication_frontiers(c)
    shared_by_chunk = [ExplicitStatePublication(frontiers) for _ in ids_by_chunk]
    hashes_by_chunk: list[Any] = [None for _ in ids_by_chunk]

    prefetch = getattr(lm, "_engram_prefetch", None)
    with prefetch.forward() if prefetch is not None else nullcontext():
        for layer_index, layer in enumerate(lm.layers):
            has_engram = "engram" in layer
            if prefetch is not None and has_engram:
                ix = list(c.engram_layer_ids).index(layer_index)
                for hashes in hashes_by_chunk:
                    prefetch.submit(layer.engram.embed, hashes[:, :, ix])
            for chunk_index, ids in enumerate(ids_by_chunk):
                h = h_by_chunk[chunk_index]
                pre = pre_by_chunk[chunk_index]
                image_mask = None
                if layer_index == 0 and lm._hasher is not None:
                    hashes, history = lm._hasher(ids, rc[0][6], image_mask)
                    hashes_by_chunk[chunk_index] = hashes
                else:
                    history = None
                if has_engram:
                    ix = list(c.engram_layer_ids).index(layer_index)
                    if prefetch is not None:
                        mx.async_eval(h, pre)
                    h = layer.engram(h, hashes_by_chunk[chunk_index][:, :, ix], image_mask)
                start = rc[layer_index].size()
                h, pre = layer(h, pre, rc[layer_index], shared_by_chunk[chunk_index], start, image_mask)
                if prefetch is not None and has_engram:
                    mx.async_eval(h, pre)
                rc[layer_index][0] = mx.array([start + ids.shape[1]], mx.int32)
                if history is not None and layer_index == 0:
                    rc[layer_index][6] = mx.array(history, mx.int64)
                fill_deepseek_v41_empty_slots(rc[layer_index], h, c, pack_activation)
                if record_publications:
                    shared_by_chunk[chunk_index].publish(layer_index, rc[layer_index], has_engram=has_engram)
                h_by_chunk[chunk_index] = h
                pre_by_chunk[chunk_index] = pre
                if eval_boundary == "chunk":
                    _eval_layer_state([h], [pre], rc[layer_index])
            if eval_boundary == "layer":
                _eval_layer_state(h_by_chunk, pre_by_chunk, rc[layer_index])

    if final_logits_only:
        h = h_by_chunk[-1][:, -1:, ...]
        pre = pre_by_chunk[-1][:, -1:, ...]
        logits = project_logits(lm.norm(hc_pre(h, pre)), lm.head.weight)
    elif output_mode == "full_chunks_last":
        logits = None
        for h, pre in zip(h_by_chunk, pre_by_chunk):
            logits = project_logits(lm.norm(hc_pre(h, pre)), lm.head.weight)
            mx.eval(logits)
        if logits is None:
            raise ValueError("no chunks to project")
    else:
        logits = mx.concatenate(
            [project_logits(lm.norm(hc_pre(h, pre)), lm.head.weight) for h, pre in zip(h_by_chunk, pre_by_chunk)],
            axis=1,
        )
    for i, item in enumerate(cache):
        merged = DeepseekV41Cache.merge([rc[i]])
        item.cache = merged.cache
        item.advance(sum(len(ids) for ids in chunk_ids))
    return logits, [records_to_json(shared.records) for shared in shared_by_chunk]
