"""Production MLX prefill selector facade.

``DwarfStarMLXPrefillSession`` is the runtime-facing compatibility API and now
selects the M6-qualified dense P0-P7 path:

    DeferredPrefillAppend -> LivePrefillResult -> P5 handoff

The old one-chunk oMLX layer-loop implementation is retained only as
``LegacyOneChunkMLXPrefillSession`` for diagnostics/evidence.  Production serving
must not select the reference vertical slice, export ``PrefillContinuationState``,
repack cache tensors, replay the prompt, or enable rejected P8 tile-native carry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Sequence
import importlib
import os
import sys

import numpy as np

from ds41f_mlx.runtime.omlx_core import DEFAULT_OMLX
from ds41f_mlx.prefill_fp8_mlx import DeferredPrefillAppend, LivePrefillResult
from ds41f_mlx.prefill_fp8_mlx.p7_scheduling import P7_ENGRAM_TILE
from ds41f_mlx.prefill_fp8_mlx.telemetry import summarize_p7_engram_events

PRODUCTION_PREFILL_SELECTOR = "DENSE_P0_P7"
ONE_CHUNK_SUBSTRATE = "DIAGNOSTIC_LEGACY"
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


class LegacyOneChunkMLXPrefillSession:
    """Diagnostic-only one-chunk practical prefill using real MLX/oMLX model operations.

    This legacy substrate is retained as evidence/tooling only.  It is not the
    production selector and must not be used by serving or qualification.
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


@dataclass(frozen=True)
class DenseP0P7PrefillResult:
    """Runtime-facing production prefill result backed by a live P6/P7 cache."""

    prefix_token_ids: tuple[int, ...]
    frontier: int
    live_result: LivePrefillResult
    seconds: float
    phase_timings_s: dict[str, float]
    segment_metadata: tuple[dict[str, Any], ...]
    p7_scheduling_evidence: dict[str, Any]
    production_prefill_selector: str = PRODUCTION_PREFILL_SELECTOR
    portable_state_exported: bool = False
    full_cache_repack_count: int = 0

    @property
    def token_ids(self) -> np.ndarray:
        return np.asarray(self.prefix_token_ids, dtype=np.int64).reshape(1, -1)

    @property
    def live_cache(self) -> list[Any]:
        return self.live_result.live_cache

    @property
    def cache_frontier(self) -> int:
        return self.frontier

    def to_json(self) -> dict[str, Any]:
        return {
            "production_prefill_selector": self.production_prefill_selector,
            "token_count": len(self.prefix_token_ids),
            "frontier": self.frontier,
            "seconds": self.seconds,
            "phase_timings_s": dict(self.phase_timings_s),
            "segment_count": len(self.segment_metadata),
            "segment_metadata": list(self.segment_metadata),
            "p7_scheduling_evidence": dict(self.p7_scheduling_evidence),
            "portable_state_exported": self.portable_state_exported,
            "full_cache_repack_count": self.full_cache_repack_count,
            "handoff_count": self.live_result.handoff_count,
        }


class DenseP0P7PrefillSession:
    """M6-qualified dense P0-P7 production prefill selector implementation."""

    def __init__(self, model: Any, *, omlx_path: Any = DEFAULT_OMLX, mx: Any | None = None, stream: Any | None = None):
        root = str(omlx_path or DEFAULT_OMLX)
        if root not in sys.path:
            sys.path.insert(0, root)
        self.mx = mx if mx is not None else importlib.import_module("mlx.core")
        self.model = model
        self.language_model = getattr(model, "language_model", model)
        if hasattr(self.language_model, "configure_mtp"):
            self.language_model.configure_mtp(False, 1)
        self.stream = stream
        self._require_production_configuration()

    def _require_production_configuration(self) -> None:
        if os.environ.get("DS41F_P8_TILE_NATIVE_CARRY", "").strip() in {"1", "true", "TRUE", "yes", "on"}:
            raise RuntimeError("P8 tile-native carry was rejected and is forbidden for production serving")
        setattr(self.language_model, "_p7_enable_overlap", True)
        if getattr(self.language_model, "_p7_enable_overlap", False) is not True:
            raise RuntimeError("production DENSE_P0_P7 requires P7 overlap enabled")

    def prefill(self, token_ids: Sequence[int]) -> DenseP0P7PrefillResult:
        LegacyOneChunkMLXPrefillSession._assert_no_reference_modules_loaded_at_entry()
        ids = tuple(int(t) for t in token_ids)
        if not ids:
            raise ValueError("production prefill requires at least one prefix token")
        make_cache = getattr(self.language_model, "make_cache", None)
        if make_cache is None:
            raise RuntimeError("language model does not expose make_cache() for live-cache prefill")
        t0 = perf_counter()
        cache = make_cache()
        app = DeferredPrefillAppend.create(
            self.language_model,
            cache,
            ids,
            committed_frontier=0,
            mx=self.mx,
        )
        app.execute_all()
        if app.commit_certificate is None:
            raise RuntimeError("DENSE_P0_P7 prefill did not produce a commit certificate")
        live = LivePrefillResult.from_committed(app.commit_certificate, prefix_token_ids=ids)
        seconds = perf_counter() - t0
        segment_metadata = tuple(
            {
                "mode": getattr(record.mode, "value", str(record.mode)),
                "start": int(record.start),
                "end": int(record.end),
                "command_count": int(record.command_count),
                "E_after": int(record.E_after),
                "D_after": int(record.D_after),
                "cone_rows": None if record.cone_rows is None else int(record.cone_rows),
                "scheduling_event_count": len(record.scheduling_events),
            }
            for record in app.segment_records
        )
        p7_events = [event for record in app.segment_records for event in record.scheduling_events]
        p7_summary = summarize_p7_engram_events(p7_events)
        return DenseP0P7PrefillResult(
            prefix_token_ids=ids,
            frontier=live.frontier,
            live_result=live,
            seconds=seconds,
            phase_timings_s={"dense_p0_p7_append": seconds},
            segment_metadata=segment_metadata,
            p7_scheduling_evidence={
                "enabled": bool(getattr(self.language_model, "_p7_enable_overlap", False)),
                "policy": "FULL_RESIDENT_BACKBONE_SSD_ENGRAM",
                "p7_engram_tile": int(P7_ENGRAM_TILE),
                "foreground_fallback": int(p7_summary["foreground_engram_fallback"]),
                "background_reads": int(p7_summary["prefetch_submissions"]),
                **p7_summary,
            },
        )


class DwarfStarMLXPrefillSession:
    """Compatibility facade for the production DENSE_P0_P7 selector."""

    def __init__(self, model: Any, *, omlx_path: Any = DEFAULT_OMLX, mx: Any | None = None, stream: Any | None = None, **_: Any):
        self._delegate = DenseP0P7PrefillSession(model, omlx_path=omlx_path, mx=mx, stream=stream)

    def prefill(self, token_ids: Sequence[int]) -> DenseP0P7PrefillResult:
        return self._delegate.prefill(token_ids)
