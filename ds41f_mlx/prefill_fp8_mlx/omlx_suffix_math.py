# SPDX-License-Identifier: MIT
# Adapted mathematical portions: Copyright (c) 2023 DeepSeek.
# See OMLX_MATH_LICENSE.
"""Command-local suffix math; no oMLX scheduler or ordinary Block fallback.

Topology: antirez/ds4@0aaea5a238fb41a35106a551e73c8409dfb751ac.
Math: oMLX b390b31 + jundot/omlx@36493634 (CED decomposition).
All projections, norms, HC, packing, sparse attention and MoE use loaded
modules/official operations. Source production and query ranking are separate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import importlib
from functools import wraps


class SuffixMathError(RuntimeError):
    """Loaded semantics cannot be expressed by this reviewed adapter."""


def _fail_closed(operation):
    @wraps(operation)
    def checked(*args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except (AttributeError, KeyError, TypeError, IndexError) as exc:
            raise SuffixMathError(f"{operation.__name__}: unsupported loaded semantics") from exc
    return checked


@dataclass
class OmlxV41SuffixMath:
    language_model: Any
    full_source_generation_count: int = 0
    query_old_lengths: list[tuple[int, int, int]] = field(default_factory=list)

    def _ops(self, layer):
        # Unknown layers must opt in with explicit fake hooks, never __call__.
        if type(layer).__module__ != "omlx.patches.deepseek_v41.language":
            raise SuffixMathError(f"unsupported suffix layer semantics: {type(layer).__module__}")
        try:
            return importlib.import_module(type(layer).__module__), importlib.import_module("mlx.core")
        except ImportError as exc:
            raise SuffixMathError("official oMLX/MLX operations unavailable") from exc

    @_fail_closed
    def prepare_local_window(self, *, layer_id, h_rows, pre_rows, cache, absolute_start, rows):
        layer = self.language_model.layers[layer_id]
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "prepare_decoder_local_window_math"):
            attn.prepare_decoder_local_window_math(h_rows, pre_rows, cache, absolute_start, rows)
            return
        lang, mx = self._ops(layer)
        c = self.language_model._config
        x = lang.hc_pre_norm(h_rows, pre_rows, layer.attn_norm.weight, layer.attn_norm.eps)
        _, kv_input = attn._input_projections(x)
        new = lang.pack_activation(lang.rope(attn.kv_norm(kv_input), mx.arange(absolute_start, absolute_start + rows), c, bool(c.compress_ratios[layer_id])))
        # Rebuild, not append: previous-sweep rows are not contiguous with this cone.
        cache[1] = new[:, -c.window_size:]

    @_fail_closed
    def publish_full_source(self, *, layer_id, h_full, pre_full, cache, shared, absolute_start, rows):
        layer = self.language_model.layers[layer_id]
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "publish_full_encoder_source_math"):
            attn.publish_full_encoder_source_math(h_full, pre_full, cache, shared, absolute_start, rows)
            self.full_source_generation_count += 1
            return
        lang, mx = self._ops(layer)
        c = self.language_model._config
        ratio = c.compress_ratios[layer_id]
        if layer_id not in c.kv_source_layers or not ratio:
            raise SuffixMathError("full source prepare requires a compressed KV source")
        x = lang.hc_pre_norm(h_full, pre_full, layer.attn_norm.weight, layer.attn_norm.eps)
        latent = attn.compressor(x, cache, absolute_start)
        # ratio=1 compressor leaves pending slots 4/5 untouched, exactly as upstream.
        previous = cache[2]
        if previous is None:
            previous = lang.pack_activation(mx.zeros((1, 0, c.head_dim), x.dtype), bits=4, group_size=16, e4m3_scale=True)
        previous = previous[:, :absolute_start // ratio]
        if latent is not None:
            pos = mx.arange(absolute_start // ratio, (absolute_start + rows) // ratio) * ratio
            compressed = lang.pack_activation(lang.rope(latent, pos, c, True), bits=4, group_size=16, e4m3_scale=True)
            previous = mx.concatenate([previous, compressed], 1)
        shared["kv"] = cache[2] = previous
        indexer = attn.indexer
        previous = cache[3]
        if previous is None:
            previous = lang.pack_activation(mx.zeros((1, 0, c.index_head_dim), x.dtype), bits=4)
        previous = previous[:, :absolute_start // ratio]
        if latent is not None:
            key = lang.pack_activation(lang.rope(indexer.k_norm(indexer.wk(latent)), pos, c, True), bits=4)
            previous = mx.concatenate([previous, key], 1)
        shared["index_k"] = cache[3] = previous
        # No q_norm, index queries, top-k, candidates or full-prompt idx here.
        self.full_source_generation_count += 1

    @_fail_closed
    def execute_suffix_query(self, *, layer_id, h_chunk, pre_chunk, cache, shared, absolute_start, image_mask):
        layer = self.language_model.layers[layer_id]
        if hasattr(layer, "execute_suffix_query_math"):
            return layer.execute_suffix_query_math(h_chunk, pre_chunk, cache, shared, absolute_start, image_mask)
        lang, mx = self._ops(layer)
        c = self.language_model._config
        try:
            ap, ao, ac = lang.hc_mixes(h_chunk, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, c)
            x = lang.hc_pre_norm(h_chunk, pre_chunk, layer.attn_norm.weight, layer.attn_norm.eps)
            out = self._attention(layer.attn, x, cache, shared, absolute_start, layer_id, lang, mx)
            h = lang.hc_post(out, h_chunk, ao, ac)
            fp, fo, fc = lang.hc_mixes(h, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, c)
            h = lang.hc_post(layer.ffn(lang.hc_pre_norm(h, ap, layer.ffn_norm.weight, layer.ffn_norm.eps), image_mask), h, fo, fc)
            return h, fp
        except (AttributeError, KeyError, TypeError) as exc:
            raise SuffixMathError("loaded layer does not express reviewed suffix semantics") from exc

    def _index_query(self, indexer, x, qr, shared, start, ratio, layer, lang, mx):
        # Query half of reviewed Indexer.__call__; deliberately no wk/cache writes.
        c = self.language_model._config
        key = shared.get("index_k")
        if key is None:
            raise SuffixMathError("suffix index query requires published/private index K")
        end = start + x.shape[1]
        q = indexer.wq_b(qr).reshape(1, x.shape[1], c.index_n_heads, c.index_head_dim)
        q = lang.quantize_activation(lang.rope(q, mx.arange(start, end), c, True), bits=4)
        weights = indexer.weights_proj(x).astype(mx.float32) * (c.index_head_dim**-0.5 * c.index_n_heads**-0.5)
        if not (0 <= c.candidate_source_layer < layer):
            idx, blocks = lang.packed_index_topk(q, key, weights, start, ratio, c.index_topk, block_count=c.candidate_topk_blocks if layer == c.candidate_source_layer else 0, block_size=c.candidate_block_size)
            if layer == c.candidate_source_layer:
                shared["candidates"] = blocks
            return idx
        blocks = shared["candidates"]
        candidates = blocks[..., None] * c.candidate_block_size + mx.arange(c.candidate_block_size)
        candidates = mx.where(blocks[..., None] >= 0, candidates, -1).reshape(1, x.shape[1], blocks.shape[-1] * c.candidate_block_size)
        scores = lang.packed_index_scores(q, key, weights, start, ratio, candidates)
        count = min(c.index_topk, scores.shape[-1])
        order = mx.argsort(-scores, axis=-1)[..., :count].astype(mx.int32)
        valid = mx.take_along_axis(scores, order, axis=-1) > -float("inf")
        idx = mx.take_along_axis(candidates, order, -1)
        return mx.sort(mx.where(valid, idx, -1), axis=-1)

    def _attention(self, attn, x, cache, shared, start, layer, lang, mx):
        c = self.language_model._config
        ratio, length = c.compress_ratios[layer], x.shape[1]
        positions = mx.arange(start, start + length)
        query, kv_input = attn._input_projections(x)
        qr = attn.q_norm(query)
        q = lang.rope(attn.wq_b(qr).reshape(1, length, c.n_heads, c.head_dim), positions, c, bool(ratio))
        new = lang.pack_activation(lang.rope(attn.kv_norm(kv_input), positions, c, bool(ratio)))
        old = cache[1]
        old_len = min(start, c.window_size, 0 if old is None else int(old.shape[1]))
        self.query_old_lengths.append((layer, start, old_len))
        kv = mx.concatenate([old[:, :old_len], new], 1) if old_len else new
        cache[1] = kv[:, -c.window_size:]
        if start == 0:
            local = mx.maximum(positions[:, None] - c.window_size + 1, 0) + mx.arange(min(length, c.window_size))
        else:
            local = positions[:, None] - c.window_size + 1 + mx.arange(c.window_size)
        valid = (local >= max(0, start - old_len)) & (local <= positions[:, None])
        idx = mx.where(valid, local - (start - old_len), -1)[None]
        pooled = mx.zeros((1, 0, c.head_dim // 2 + c.head_dim // 16), mx.uint8)
        ci = mx.zeros((1, length, 0), mx.int32)
        if ratio:
            if shared.get("kv") is None:
                raise SuffixMathError("suffix query requires published/private compressed KV")
            if layer in c.index_source_layers:
                shared["idx"] = self._index_query(attn.indexer, x, qr, shared, start, ratio, layer, lang, mx)
            if shared.get("idx") is None:
                raise SuffixMathError("suffix query requires row-span idx")
            ci, pooled = shared["idx"], shared["kv"]
        # Official sparse implementation consumes explicit local indices. The fused
        # long-prefill kernel assumes a full old window and is not used here.
        out = lang.packed_sparse_attention(q, kv, pooled, idx, ci, attn.attn_sink, c.head_dim**-0.5)
        out = lang.rope(out, positions, c, bool(ratio), inverse=True)
        grouped = out.reshape(1, length, c.o_groups, -1)
        weight = attn.wo_a.weight.reshape(c.o_groups, c.o_lora_rank, -1)
        return attn.wo_b(mx.einsum("bsgd,grd->bsgr", grouped, weight).flatten(-2))
