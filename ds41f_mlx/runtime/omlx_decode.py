"""Production oMLX-derived base target decode session.

Milestone 4 selects oMLX DeepSeek-V4.1 target decode as the production decode
architecture after DwarfStar-derived prefill.  This module owns the narrow
runtime-facing seam:

    PrefillContinuationState -> DeepseekV41Cache[40] -> one-token target decode

It deliberately uses oMLX model-core/cache objects and does not use oMLX HTTP,
prompt parsing, scheduler prefill, prompt replay, or DSpark/MTP proposal paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable
import importlib
import sys

import numpy as np

from ds41f_mlx.prefill_session import PrefillContinuationState
from ds41f_mlx.runtime.omlx_core import DEFAULT_OMLX, OmlxRuntime, OmlxRuntimeConfig

SOURCE_LAYERS = (2, 8, 14, 20)
INDEX_GENERATIONS = (2, 8, 14, 20, 24, 28, 32, 36)


class OMLXDecodeAdmissionError(ValueError):
    """A committed prefill state cannot be admitted into oMLX decode caches."""


@dataclass(frozen=True)
class OMLXDecodeConfig:
    """Configuration for base target decode.

    Speculation is intentionally disabled for Milestone 4 base qualification.
    """

    omlx_path: Path = DEFAULT_OMLX
    checkpoint_path: Path | None = None
    engram_ssd_offload: bool = True
    preserve_mtp: bool | None = False
    moe_expert_offload_resident_fraction: float | None = None
    speculation_enabled: bool = False
    eval_logits: bool = True


@dataclass(frozen=True)
class CacheSlotSummary:
    layer: int
    compress_ratio: int
    offset: int
    slot_shapes: dict[int, tuple[int, ...] | None]
    slot_dtypes: dict[int, str | None]


@dataclass(frozen=True)
class OMLXAdmissionReport:
    schema: str
    token_frontier: int
    cache_layers: int
    no_prompt_replay: bool
    slot_summaries: tuple[CacheSlotSummary, ...]
    transient_state_policy: dict[str, str]
    engram_history_shape: tuple[int, ...] | None
    dtype_layout_conversions: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "token_frontier": self.token_frontier,
            "cache_layers": self.cache_layers,
            "no_prompt_replay": self.no_prompt_replay,
            "slot_summaries": [
                {
                    "layer": s.layer,
                    "compress_ratio": s.compress_ratio,
                    "offset": s.offset,
                    "slot_shapes": {str(k): (None if v is None else list(v)) for k, v in s.slot_shapes.items()},
                    "slot_dtypes": {str(k): v for k, v in s.slot_dtypes.items()},
                }
                for s in self.slot_summaries
            ],
            "transient_state_policy": dict(self.transient_state_policy),
            "engram_history_shape": None if self.engram_history_shape is None else list(self.engram_history_shape),
            "dtype_layout_conversions": list(self.dtype_layout_conversions),
        }


@dataclass(frozen=True)
class DecodeStepReport:
    token: int
    frontier_before: int
    frontier_after: int
    logits_shape: tuple[int, ...]
    logits_dtype: str
    elapsed_s: float
    cache_offsets: tuple[int, ...]
    state_changed: dict[str, Any]
    committed: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "frontier_before": self.frontier_before,
            "frontier_after": self.frontier_after,
            "logits_shape": list(self.logits_shape),
            "logits_dtype": self.logits_dtype,
            "elapsed_s": self.elapsed_s,
            "cache_offsets": list(self.cache_offsets),
            "state_changed": self.state_changed,
            "committed": self.committed,
        }


@dataclass
class _TransactionSnapshot:
    cache_refs: list[list[Any]]
    left_padding: list[Any]
    lengths: list[Any]
    token_ids: np.ndarray
    token_frontier: int
    committed_steps: int


@dataclass
class OMLXDecodeStateAdapter:
    """Admit a committed M2 `PrefillContinuationState` into real oMLX caches."""

    model: Any
    omlx_path: Path = DEFAULT_OMLX
    _mx: Any = field(init=False, repr=False)
    _cache_cls: Any = field(init=False, repr=False)
    _pack_activation: Any = field(init=False, repr=False)
    _conversions: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        root = str(self.omlx_path)
        if root not in sys.path:
            sys.path.insert(0, root)
        object.__setattr__(self, "_mx", importlib.import_module("mlx.core"))
        cache_mod = importlib.import_module("omlx.patches.deepseek_v41.cache")
        quant_mod = importlib.import_module("omlx.patches.deepseek_v41.quantization")
        object.__setattr__(self, "_cache_cls", cache_mod.DeepseekV41Cache)
        object.__setattr__(self, "_pack_activation", quant_mod.pack_activation)

    @property
    def language_model(self) -> Any:
        return getattr(self.model, "language_model", self.model)

    @property
    def config(self) -> Any:
        return self.language_model._config

    def admit(self, state: PrefillContinuationState) -> tuple[list[Any], OMLXAdmissionReport]:
        if not state.committed:
            raise OMLXDecodeAdmissionError("PrefillContinuationState must be committed before decode admission")
        c = self.config
        n_layers = int(c.n_layers)
        if n_layers != 40:
            raise OMLXDecodeAdmissionError(f"expected 40 V4.1 layers, got {n_layers}")
        if state.token_frontier <= 0 or state.token_ids.shape[-1] != state.token_frontier:
            raise OMLXDecodeAdmissionError("token_ids/token_frontier mismatch")
        missing_window = [i for i in range(n_layers) if i not in state.window_kv_by_layer]
        if missing_window:
            raise OMLXDecodeAdmissionError(f"missing window KV layers: {missing_window[:8]}")
        missing_sources = [s for s in c.kv_source_layers if s not in state.compressed_kv_by_source or s not in state.index_k_by_source]
        if missing_sources:
            raise OMLXDecodeAdmissionError(f"missing source compressed/index state: {missing_sources}")
        if 20 not in state.candidates_by_source:
            raise OMLXDecodeAdmissionError("candidate source@20 evidence missing; cannot validate transient regeneration policy")
        missing_topk = [g for g in INDEX_GENERATIONS if g not in state.topk_by_generation]
        if missing_topk:
            raise OMLXDecodeAdmissionError(f"top-k generation evidence missing: {missing_topk}")

        cache = [
            self._cache_cls(c.compress_ratios[i] if i in c.kv_source_layers else 0)
            for i in range(n_layers)
        ]
        history = self._engram_history_from_state(state)
        for layer, item in enumerate(cache):
            ratio = c.compress_ratios[layer] if layer in c.kv_source_layers else 0
            item.cache[0] = self._mx.array([state.token_frontier], self._mx.int32)
            item.cache[1] = self._pack_cache_array(
                state.window_kv_by_layer[layer],
                role=f"layer{layer}.window_kv",
                bits=8,
                group_size=32,
                e4m3_scale=False,
                expected_unpacked_width=int(c.head_dim),
            )
            if layer in c.kv_source_layers:
                item.cache[2] = self._pack_cache_array(
                    state.compressed_kv_by_source[layer],
                    role=f"source{layer}.compressed_kv",
                    bits=4,
                    group_size=16,
                    e4m3_scale=True,
                    expected_unpacked_width=int(c.head_dim),
                )
                item.cache[3] = self._pack_cache_array(
                    state.index_k_by_source[layer],
                    role=f"source{layer}.index_k",
                    bits=4,
                    group_size=32,
                    e4m3_scale=False,
                    expected_unpacked_width=int(c.index_head_dim),
                )
                pending = state.compressor_pending.get(layer, {})
                item.cache[4] = self._pending_array(
                    pending.get("kv"), f"source{layer}.pending_kv", int(c.head_dim), self._mx.bfloat16
                )
                item.cache[5] = self._pending_array(
                    pending.get("gate", pending.get("score")), f"source{layer}.pending_gate", int(c.head_dim), self._mx.bfloat16
                )
            else:
                item.cache[2] = self._empty_packed(int(c.head_dim), bits=4, group_size=16, e4m3_scale=True)
                item.cache[3] = self._empty_packed(int(c.index_head_dim), bits=4, group_size=32, e4m3_scale=False)
                item.cache[4] = self._mx.zeros((1, 0, int(c.head_dim)), self._mx.bfloat16)
                item.cache[5] = self._mx.zeros((1, 0, int(c.head_dim)), self._mx.bfloat16)
            item.cache[6] = history if layer == 0 and history is not None else self._mx.zeros((1, 0), self._mx.int64)
            self._validate_layer_cache(layer, item, state.token_frontier)

        summaries = tuple(self._summarize_cache(i, item) for i, item in enumerate(cache))
        report = OMLXAdmissionReport(
            schema="ds41f.m4.omlx-admission.v1",
            token_frontier=state.token_frontier,
            cache_layers=len(cache),
            no_prompt_replay=True,
            slot_summaries=summaries,
            transient_state_policy={
                "candidates_by_source[20]": "validated-present; oMLX regenerates candidates transiently inside Attention/Indexer target forward from admitted source@20 index state",
                "topk_by_generation": "validated-present for M2 handoff; oMLX regenerates top-k/index selections transiently from admitted compressed/index state",
                "shared_publications": "not persisted as cache authority; per-forward oMLX shared dict is rebuilt from admitted cache slots",
                "field_ownership/source_generation_order": "validated at admission; layer/source order is fixed by oMLX model config",
            },
            engram_history_shape=None if history is None else tuple(history.shape),
            dtype_layout_conversions=tuple(self._conversions),
        )
        return cache, report

    def _empty_packed(self, width: int, *, bits: int, group_size: int, e4m3_scale: bool):
        return self._pack_activation(
            self._mx.zeros((1, 0, width), self._mx.bfloat16),
            bits=bits,
            group_size=group_size,
            e4m3_scale=e4m3_scale,
        )

    def _as_mx_bf16(self, arr: np.ndarray, role: str):
        a = np.asarray(arr)
        if a.dtype == np.uint16:
            self._conversions.append(f"{role}: uint16 BF16 bits -> mx.bfloat16")
            return self._mx.array(a).view(self._mx.bfloat16)
        mx_arr = self._mx.array(a)
        if str(mx_arr.dtype) != "bfloat16":
            self._conversions.append(f"{role}: {a.dtype} -> mx.bfloat16")
            mx_arr = mx_arr.astype(self._mx.bfloat16)
        return mx_arr

    def _packed_width(self, width: int, *, bits: int, group_size: int) -> int:
        value_bytes = width if bits == 8 else width // 2
        scale_bytes = width // group_size
        return value_bytes + scale_bytes

    def _pack_cache_array(self, arr: np.ndarray, *, role: str, bits: int, group_size: int, e4m3_scale: bool, expected_unpacked_width: int):
        a = np.asarray(arr)
        if a.ndim != 3 or a.shape[0] != 1:
            raise OMLXDecodeAdmissionError(f"{role} expected shape [1, rows, width], got {a.shape}")
        packed_width = self._packed_width(expected_unpacked_width, bits=bits, group_size=group_size)
        if a.dtype == np.uint8 and a.shape[-1] == packed_width:
            self._conversions.append(f"{role}: already packed uint8 width {packed_width}")
            return self._mx.array(a)
        if a.shape[-1] != expected_unpacked_width:
            raise OMLXDecodeAdmissionError(
                f"{role} expected unpacked width {expected_unpacked_width} or packed width {packed_width}, got {a.shape[-1]}"
            )
        bf16 = self._as_mx_bf16(a, role)
        self._conversions.append(
            f"{role}: pack_activation(bits={bits}, group_size={group_size}, e4m3_scale={e4m3_scale})"
        )
        return self._pack_activation(bf16, bits=bits, group_size=group_size, e4m3_scale=e4m3_scale)

    def _pending_array(self, arr: np.ndarray | None, role: str, width: int, dtype: Any):
        if arr is None:
            return self._mx.zeros((1, 0, width), dtype)
        a = np.asarray(arr)
        if a.ndim != 3 or a.shape[0] != 1 or a.shape[-1] not in (1, width):
            raise OMLXDecodeAdmissionError(f"{role} invalid pending shape {a.shape}")
        mx_arr = self._as_mx_bf16(a, role) if a.dtype == np.uint16 else self._mx.array(a).astype(dtype)
        if a.shape[-1] == 1 and width != 1:
            mx_arr = self._mx.broadcast_to(mx_arr, (a.shape[0], a.shape[1], width))
            self._conversions.append(f"{role}: broadcast scalar pending gate to head_dim {width}")
        return mx_arr

    def _engram_history_from_state(self, state: PrefillContinuationState):
        lm = self.language_model
        hasher = getattr(lm, "_hasher", None)
        if hasher is None:
            if getattr(self.config, "engram_layer_ids", None):
                raise OMLXDecodeAdmissionError("oMLX model has no Engram hasher; tokenizer was not installed")
            return None
        ids = np.asarray(state.token_ids, dtype=np.int64)
        _hashes, history = hasher(ids, None, None)
        self._conversions.append(
            "Engram slot6 history regenerated from committed token_ids using oMLX NgramHash only; no model/prompt forward replay"
        )
        return self._mx.array(history, self._mx.int64)

    def _validate_layer_cache(self, layer: int, item: Any, frontier: int) -> None:
        if item.size() != frontier:
            raise OMLXDecodeAdmissionError(f"layer {layer} offset {item.size()} != frontier {frontier}")
        for slot in range(7):
            if item.cache[slot] is None:
                raise OMLXDecodeAdmissionError(f"layer {layer} slot {slot} is None after admission")

    def _summarize_cache(self, layer: int, item: Any) -> CacheSlotSummary:
        return CacheSlotSummary(
            layer=layer,
            compress_ratio=int(item.compress_ratio or 0),
            offset=int(item.size()),
            slot_shapes={i: (None if item.cache[i] is None else tuple(item.cache[i].shape)) for i in range(7)},
            slot_dtypes={i: (None if item.cache[i] is None else str(item.cache[i].dtype)) for i in range(7)},
        )


class OMLXDecodeSession:
    """Production base target decode session after M2 state admission."""

    def __init__(self, model: Any, cache: list[Any], token_ids: np.ndarray, config: OMLXDecodeConfig | None = None, admission_report: OMLXAdmissionReport | None = None):
        self.model = model
        self.language_model = getattr(model, "language_model", model)
        self.cache = cache
        self.config = config or OMLXDecodeConfig()
        self.admission_report = admission_report
        self.token_ids = np.asarray(token_ids, dtype=np.int64).copy()
        self.token_frontier = int(self.token_ids.shape[-1])
        self.committed_steps = 0
        self.speculation_enabled = False
        if hasattr(self.language_model, "configure_mtp"):
            self.language_model.configure_mtp(False, 1)

    @classmethod
    def from_prefill_state(cls, model: Any, state: PrefillContinuationState, config: OMLXDecodeConfig | None = None) -> "OMLXDecodeSession":
        cfg = config or OMLXDecodeConfig()
        adapter = OMLXDecodeStateAdapter(model=model, omlx_path=cfg.omlx_path)
        cache, report = adapter.admit(state)
        return cls(model=model, cache=cache, token_ids=state.token_ids, config=cfg, admission_report=report)

    @classmethod
    def load_model_and_admit(cls, state: PrefillContinuationState, config: OMLXDecodeConfig) -> "OMLXDecodeSession":
        rt_cfg = OmlxRuntimeConfig(
            omlx_path=config.omlx_path,
            checkpoint_path=config.checkpoint_path or OmlxRuntimeConfig().checkpoint_path,
            engram_ssd_offload=config.engram_ssd_offload,
            preserve_mtp=False if config.preserve_mtp is None else config.preserve_mtp,
            moe_expert_offload_resident_fraction=config.moe_expert_offload_resident_fraction,
        )
        runtime = OmlxRuntime(rt_cfg)
        model, _processor = runtime.load_model()
        session = cls.from_prefill_state(model, state, config)
        session._runtime = runtime  # keep static model resources alive and closeable
        return session

    def decode_one(self, token: int, *, inject_failure: str | None = None) -> tuple[Any, DecodeStepReport]:
        if self.speculation_enabled:
            raise RuntimeError("base Milestone 4 decode requires speculation disabled")
        mx = importlib.import_module("mlx.core")
        before = self._snapshot()
        before_offsets = self.cache_offsets()
        t0 = perf_counter()
        try:
            if inject_failure == "before_forward":
                raise RuntimeError("injected failure before oMLX target forward")
            input_ids = mx.array([[int(token)]], mx.int64)
            # Milestone 3 selected the model-core LanguageModel._forward target path.
            # Avoid the multimodal wrapper/protocol layer and its image-token checks.
            logits = self.language_model._forward(input_ids, cache=self.cache)
            if inject_failure == "after_forward_before_eval":
                raise RuntimeError("injected failure after oMLX target forward before eval")
            if self.config.eval_logits:
                mx.eval(logits)
            if inject_failure == "after_eval_before_commit":
                raise RuntimeError("injected failure after logits eval before commit")
            elapsed = perf_counter() - t0
            self.token_ids = np.concatenate([self.token_ids, np.array([[int(token)]], dtype=np.int64)], axis=1)
            self.token_frontier += 1
            self.committed_steps += 1
            offsets = self.cache_offsets()
            report = DecodeStepReport(
                token=int(token),
                frontier_before=before.token_frontier,
                frontier_after=self.token_frontier,
                logits_shape=tuple(logits.shape),
                logits_dtype=str(logits.dtype),
                elapsed_s=elapsed,
                cache_offsets=offsets,
                state_changed=self._state_change_summary(before_offsets, offsets),
                committed=True,
            )
            return logits, report
        except BaseException:
            self._restore(before)
            raise

    def decode_many(self, tokens: Iterable[int]) -> list[DecodeStepReport]:
        reports: list[DecodeStepReport] = []
        for token in tokens:
            _logits, report = self.decode_one(int(token))
            reports.append(report)
        return reports

    def fork(self) -> "OMLXDecodeSession":
        cloned_cache = []
        for item in self.cache:
            other = type(item)(item.compress_ratio)
            other.cache = list(item.cache)  # MLX arrays are immutable/lazy refs; decode replaces refs on mutation.
            other.left_padding = item.left_padding
            other.lengths = item.lengths
            cloned_cache.append(other)
        child = OMLXDecodeSession(self.model, cloned_cache, self.token_ids.copy(), self.config, self.admission_report)
        child.token_frontier = self.token_frontier
        child.committed_steps = self.committed_steps
        return child

    def reset(self) -> None:
        self.cache = self.language_model.make_cache()
        self.token_ids = np.empty((1, 0), dtype=np.int64)
        self.token_frontier = 0
        self.committed_steps = 0
        self.speculation_enabled = False

    def close(self) -> None:
        runtime = getattr(self, "_runtime", None)
        if runtime is not None:
            runtime.close()

    def cache_offsets(self) -> tuple[int, ...]:
        return tuple(int(item.size()) for item in self.cache)

    def _snapshot(self) -> _TransactionSnapshot:
        return _TransactionSnapshot(
            cache_refs=[list(item.cache) for item in self.cache],
            left_padding=[item.left_padding for item in self.cache],
            lengths=[item.lengths for item in self.cache],
            token_ids=self.token_ids.copy(),
            token_frontier=self.token_frontier,
            committed_steps=self.committed_steps,
        )

    def _restore(self, snap: _TransactionSnapshot) -> None:
        for item, refs, left_padding, lengths in zip(self.cache, snap.cache_refs, snap.left_padding, snap.lengths):
            item.cache = list(refs)
            item.left_padding = left_padding
            item.lengths = lengths
        self.token_ids = snap.token_ids.copy()
        self.token_frontier = snap.token_frontier
        self.committed_steps = snap.committed_steps

    def _state_change_summary(self, before: tuple[int, ...], after: tuple[int, ...]) -> dict[str, Any]:
        changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
        source_shapes = {}
        for source in SOURCE_LAYERS:
            item = self.cache[source]
            source_shapes[str(source)] = {
                "offset": int(item.size()),
                "compressed": list(item.cache[2].shape),
                "index": list(item.cache[3].shape),
                "pending_kv": list(item.cache[4].shape),
                "pending_gate": list(item.cache[5].shape),
            }
        return {
            "all_layer_offsets_advanced_by_one": all((b - a) == 1 for a, b in zip(before, after)),
            "changed_layers": changed,
            "source_state_shapes": source_shapes,
            "engram_history_shape": list(self.cache[0].cache[6].shape),
            "transient_candidate_topk_regenerated_by_forward": True,
        }
