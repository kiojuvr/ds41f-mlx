"""Production MLX prefill substrate with DwarfStar-compatible seams.

This module is intentionally separate from:

* ``ds41f_mlx.dwarfstar_prefill_slice``: bounded NumPy/reference correctness
  vertical slice;
* ``ds41f_mlx.dwarfstar_prefill`` / ``m2_layer_major``: historical diagnostic
  oMLX-compatible topology prototype.

The implementation below uses the already-loaded oMLX/MLX LanguageModel and a
fresh request-local ``DeepseekV41Cache``.  It executes the transformer prefix and
keeps the live cache as the production handoff object.  It does not export or
repack through ``PrefillContinuationState`` and it deliberately skips final
prefix logits because serving consumes the final recipe token as the first decode
input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Sequence
import importlib
import sys

import numpy as np

from ds41f_mlx.runtime.omlx_core import DEFAULT_OMLX

REFERENCE_VERTICAL_SLICE_CLASSIFICATION = "PRODUCTION_PREFILL_REGRESSED_TO_REFERENCE_VERTICAL_SLICE"
FORBIDDEN_HOT_PATH_MODULES = (
    "ds41f_mlx.dwarfstar_prefill_slice",
    "ds41f_mlx.official_model_math",
    "tools.run_m4_omlx_base_decode_qualification",
    "tools.run_native_layer0_25_transformer_entry_validation",
)


@dataclass(frozen=True)
class PracticalPrefillResult:
    """Live-cache result for same-backend production handoff."""

    token_ids: np.ndarray
    live_cache: list[Any]
    seconds: float
    phase_timings_s: dict[str, float]
    cache_frontier: int
    cache_offsets: tuple[int, ...]
    execution_substrate: str
    skipped_final_logits: bool = True
    portable_state_exported: bool = False
    active_dwarfstar_mechanisms: tuple[str, ...] = field(default_factory=tuple)
    missing_dwarfstar_performance_mechanisms: tuple[str, ...] = field(default_factory=tuple)

    def to_json(self) -> dict[str, Any]:
        return {
            "token_count": int(self.token_ids.size),
            "seconds": float(self.seconds),
            "tok_per_s": (float(self.token_ids.size) / float(self.seconds)) if self.seconds > 0 else None,
            "phase_timings_s": {k: float(v) for k, v in self.phase_timings_s.items()},
            "cache_frontier": int(self.cache_frontier),
            "cache_offsets_head": list(self.cache_offsets[:8]),
            "all_cache_offsets_equal_frontier": all(o == self.cache_frontier for o in self.cache_offsets),
            "execution_substrate": self.execution_substrate,
            "skipped_final_logits": bool(self.skipped_final_logits),
            "portable_state_exported": bool(self.portable_state_exported),
            "active_dwarfstar_mechanisms": list(self.active_dwarfstar_mechanisms),
            "missing_dwarfstar_performance_mechanisms": list(self.missing_dwarfstar_performance_mechanisms),
        }


class DwarfStarMLXPrefillSession:
    """One-chunk practical prefill using real MLX/oMLX model operations.

    The serving/decode interface is intentionally future-compatible with a later
    DwarfStar sweep implementation: the request owns explicit state, execution
    order is explicit, cache/publication ownership is explicit, and the result is
    a live request-local cache ready for zero-repack handoff.
    """

    def __init__(self, model: Any, *, omlx_path: Any = DEFAULT_OMLX, eval_group_size: int = 8, stream: Any | None = None):
        root = str(omlx_path or DEFAULT_OMLX)
        if root not in sys.path:
            sys.path.insert(0, root)
        self.mx = importlib.import_module("mlx.core")
        self.model = model
        self.language_model = getattr(model, "language_model", model)
        if hasattr(self.language_model, "configure_mtp"):
            self.language_model.configure_mtp(False, 1)
        self.eval_group_size = int(eval_group_size)
        if self.eval_group_size <= 0:
            raise ValueError("eval_group_size must be positive")
        self.stream = stream

    def prefill(self, token_ids: Sequence[int]) -> PracticalPrefillResult:
        self._assert_no_reference_modules_loaded_at_entry()
        ids_list = [int(t) for t in token_ids]
        if not ids_list:
            raise ValueError("production prefill requires at least one prefix token")

        mx = self.mx
        lm = self.language_model
        c = lm._config
        phase_timings: dict[str, float] = {}
        t_total = perf_counter()

        t0 = perf_counter()
        cache = lm.make_cache()
        self._validate_fresh_cache_layout(cache)
        input_ids = mx.array(ids_list, dtype=mx.int64)[None]
        rc = [item.extract(0) for item in cache]
        start = rc[0].size()
        if start != 0 or any(item.size() != start for item in rc):
            raise RuntimeError("fresh request-local cache is not empty/aligned")
        if c.engram_layer_ids and getattr(lm, "_hasher", None) is None:
            raise RuntimeError("DeepSeek V4.1 requires loaded tokenizer-derived Engram token map")
        image_mask = None
        h = lm.embed(input_ids)
        hashes, history = (None, None)
        if getattr(lm, "_hasher", None) is not None:
            hashes, history = lm._hasher(input_ids, rc[0][6], image_mask)
        h = mx.repeat(h[..., None, :], c.hc_mult, -2)
        pre = mx.broadcast_to((mx.arange(c.hc_mult) == 0).astype(mx.float32), h.shape[:-1])
        shared: dict[str, Any] = {}
        # Materialize setup once so timing does not hide tokenizer/Engram hash work
        # inside the first layer group.
        eval_items = [h, pre]
        if hashes is not None:
            eval_items.append(hashes)
        if history is not None:
            eval_items.append(history)
        mx.eval(*eval_items)
        phase_timings["embedding/setup"] = perf_counter() - t0

        prefetch = getattr(lm, "_engram_prefetch", None)
        engram_ids = list(c.engram_layer_ids)
        group_start = 0
        group_t0 = perf_counter()
        with prefetch.forward() if prefetch is not None else _NullContext():
            if prefetch is not None and engram_ids:
                first = lm.layers[engram_ids[0]].engram.embed
                prefetch.submit(first, hashes[:, :, 0])
            for i, layer in enumerate(lm.layers):
                if "engram" in layer:
                    ix = engram_ids.index(i)
                    if prefetch is not None:
                        mx.async_eval(h, pre)
                    h = layer.engram(h, hashes[:, :, ix], image_mask)
                    if prefetch is not None and ix + 1 < len(engram_ids):
                        next_layer = lm.layers[engram_ids[ix + 1]]
                        prefetch.submit(next_layer.engram.embed, hashes[:, :, ix + 1])
                h, pre = layer(h, pre, rc[i], shared, start, image_mask)
                if prefetch is not None and "engram" in layer:
                    mx.async_eval(h, pre)
                rc[i][0] = mx.array([len(ids_list)], mx.int32)
                if history is not None and i == 0:
                    rc[i][6] = mx.array(history, mx.int64)
                self._fill_empty_slots(rc[i], h, c)

                end_group = (i + 1) % self.eval_group_size == 0 or i + 1 == len(lm.layers)
                if end_group:
                    self._eval_live_state(h, pre, rc[i])
                    label = f"layers {group_start}-{i}"
                    phase_timings[label] = perf_counter() - group_t0
                    group_start = i + 1
                    group_t0 = perf_counter()

        t0 = perf_counter()
        for i, item in enumerate(cache):
            merged = type(item).merge([rc[i]])
            item.cache = merged.cache
            item.advance(len(ids_list))
        offsets = self.cache_offsets(cache)
        if any(o != len(ids_list) for o in offsets):
            raise RuntimeError(f"prefill cache offsets {offsets[:8]} do not match prefix length {len(ids_list)}")
        # Ensure all cache publications are real before handing to GenerationBatch.
        live_arrays: list[Any] = []
        for item in cache:
            live_arrays.extend(slot for slot in getattr(item, "cache", []) if slot is not None)
        if live_arrays:
            mx.eval(*live_arrays)
        if self.stream is not None:
            mx.synchronize(self.stream)
        else:
            mx.synchronize()
        phase_timings["handoff"] = perf_counter() - t0

        self._assert_no_reference_modules_loaded_at_entry()
        seconds = perf_counter() - t_total
        return PracticalPrefillResult(
            token_ids=np.asarray(ids_list, dtype=np.int64).reshape(1, -1),
            live_cache=cache,
            seconds=seconds,
            phase_timings_s=phase_timings,
            cache_frontier=len(ids_list),
            cache_offsets=offsets,
            execution_substrate="oMLX-operation based real MLX LanguageModel layers + request-local DeepseekV41Cache",
            active_dwarfstar_mechanisms=(
                "explicit_request_owned_prefill_state",
                "explicit_layer_execution_order",
                "explicit_cache_publication_ownership",
                "future_chunk_sweep_boundary",
                "zero_prompt_replay_handoff_contract",
            ),
            missing_dwarfstar_performance_mechanisms=(
                "static Metal carry arena",
                "DwarfStar command graph ownership",
                "decoder suffix/deferred decoder",
                "long-context chunk geometry",
                "weight/expert scheduling and cache seeding",
                "Engram read-ahead overlap beyond oMLX prefetch",
            ),
        )

    def _validate_fresh_cache_layout(self, cache: list[Any]) -> None:
        lm = self.language_model
        c = lm._config
        if len(cache) != len(lm.layers):
            raise ValueError("DeepSeek V4.1 cache layer count mismatch")
        for i, item in enumerate(cache):
            ratio = c.compress_ratios[i] if i in c.kv_source_layers else 0
            if item.compress_ratio is None:
                item.compress_ratio = ratio
            elif item.compress_ratio != ratio:
                raise ValueError("DeepSeek V4.1 cache compression layout mismatch")

    def _fill_empty_slots(self, row_cache: Any, h: Any, config: Any) -> None:
        lang = importlib.import_module(type(self.language_model).__module__)
        pack_activation = lang.pack_activation
        mx = self.mx
        for slot in range(1, 7):
            if row_cache[slot] is None:
                width = config.index_head_dim if slot == 3 else config.head_dim
                empty = mx.zeros((1, 0, width), h.dtype)
                if slot == 1:
                    empty = pack_activation(empty)
                elif slot == 2:
                    empty = pack_activation(empty, 4, 16, True)
                elif slot == 3:
                    empty = pack_activation(empty, 4)
                row_cache[slot] = mx.zeros((1, 0), mx.int64) if slot == 6 else empty

    def _eval_live_state(self, h: Any, pre: Any, row_cache: Any) -> None:
        tensors = [h, pre]
        tensors.extend(slot for slot in row_cache if slot is not None)
        self.mx.eval(*tensors)

    @staticmethod
    def cache_offsets(cache: list[Any]) -> tuple[int, ...]:
        return tuple(int(item.size()) for item in cache)

    @staticmethod
    def _assert_no_reference_modules_loaded_at_entry() -> None:
        loaded = [name for name in FORBIDDEN_HOT_PATH_MODULES if name in sys.modules]
        if loaded:
            raise RuntimeError(f"{REFERENCE_VERTICAL_SLICE_CLASSIFICATION}: forbidden reference modules loaded on production prefill hot path: {loaded}")


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False
