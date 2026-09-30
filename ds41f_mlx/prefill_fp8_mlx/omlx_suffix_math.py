"""ds41f-owned oMLX DeepSeek-V4.1 decoder-suffix math adapter.

This adapter is the production seam for DwarfStar wide-prefill suffix math.  It
uses the already-loaded oMLX layer objects and their reviewed MLX operations; it
does not require patching the external oMLX package and does not call ds41f
validation/reference helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import importlib


class SuffixMathError(RuntimeError):
    """Suffix math operation could not be expressed with loaded oMLX objects."""


@dataclass
class OmlxV41SuffixMath:
    language_model: Any

    def prepare_local_window(self, *, layer_id: int, h_rows: Any, pre_rows: Any, cache: Any, absolute_start: int, rows: int) -> None:
        """Populate cache slot 1 for preceding local-window rows without advancing slot 0."""

        layer = self.language_model.layers[layer_id]
        attn = getattr(layer, "attn", None)
        if attn is None:
            raise SuffixMathError("loaded oMLX layer has no attention module")
        if hasattr(attn, "prepare_decoder_local_window_math"):
            attn.prepare_decoder_local_window_math(h_rows, pre_rows, cache, absolute_start, rows)
            return
        lang = importlib.import_module(type(layer).__module__)
        mx = importlib.import_module("mlx.core")
        c = self.language_model._config
        normed = lang.hc_pre_norm(h_rows, pre_rows, layer.attn_norm.weight, layer.attn_norm.eps)
        _query, kv_input = attn._input_projections(normed)
        kv = lang.pack_activation(lang.rope(attn.kv_norm(kv_input), mx.arange(absolute_start, absolute_start + rows), c, bool(c.compress_ratios[layer_id])))
        old = _cache_get(cache, 1)
        old_len = min(absolute_start, c.window_size)
        if old is not None and old_len:
            kv = mx.concatenate([old[:, :old_len], kv], 1)
        _cache_set(cache, 1, kv[:, -c.window_size:])

    def publish_full_source(self, *, layer_id: int, h_full: Any, pre_full: Any, cache: Any, shared: dict[str, Any], absolute_start: int, rows: int) -> None:
        """Publish layer-20 full encoder-final compressed/index source state once."""

        layer = self.language_model.layers[layer_id]
        attn = getattr(layer, "attn", None)
        if attn is None:
            raise SuffixMathError("loaded oMLX layer has no attention module")
        if hasattr(attn, "publish_full_encoder_source_math"):
            attn.publish_full_encoder_source_math(h_full, pre_full, cache, shared, absolute_start, rows)
            return
        lang = importlib.import_module(type(layer).__module__)
        mx = importlib.import_module("mlx.core")
        c = self.language_model._config
        normed = lang.hc_pre_norm(h_full, pre_full, layer.attn_norm.weight, layer.attn_norm.eps)
        query, kv_input = attn._input_projections(normed)
        qr = attn.q_norm(query)
        latent = attn.compressor(normed, cache, absolute_start) if hasattr(attn, "compressor") else None
        if latent is not None:
            ratio = c.compress_ratios[layer_id]
            compressed = lang.rope(latent, mx.arange(absolute_start // ratio, (absolute_start + rows) // ratio) * ratio, c, True)
            compressed = lang.pack_activation(compressed, bits=4, group_size=16, e4m3_scale=True)
            prev = _cache_get(cache, 2)
            shared["kv"] = _cache_set_return(cache, 2, compressed if prev is None else mx.concatenate([prev[:, :absolute_start // ratio], compressed], 1))
        if hasattr(attn, "indexer"):
            shared["idx"] = attn.indexer(normed, qr, latent, cache, shared, absolute_start, c.compress_ratios[layer_id])
            if _cache_get(cache, 3) is not None:
                shared["index_k"] = _cache_get(cache, 3)

    def execute_suffix_query(self, *, layer_id: int, h_chunk: Any, pre_chunk: Any, cache: Any, shared: dict[str, Any], absolute_start: int, image_mask: Any) -> tuple[Any, Any]:
        """Execute official oMLX Block semantics for the planned suffix query rows."""

        layer = self.language_model.layers[layer_id]
        if hasattr(layer, "execute_suffix_query_math"):
            return layer.execute_suffix_query_math(h_chunk, pre_chunk, cache, shared, absolute_start, image_mask)
        return layer(h_chunk, pre_chunk, cache, shared, absolute_start, image_mask)


def _cache_get(cache: Any, slot: int) -> Any:
    try:
        return cache[slot]
    except Exception:
        return cache.cache[slot]


def _cache_set(cache: Any, slot: int, value: Any) -> None:
    try:
        cache[slot] = value
        return
    except Exception:
        cache.cache[slot] = value


def _cache_set_return(cache: Any, slot: int, value: Any) -> Any:
    _cache_set(cache, slot, value)
    return value
