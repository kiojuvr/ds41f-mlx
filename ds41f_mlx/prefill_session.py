"""Architecture-neutral prefill continuation handoff contract.

Milestone 2 produces a live continuation state after a successful
DwarfStar-derived prefill transaction.  The live state contains runtime tensor
objects/handles required for continuation; JSON artifacts contain only digests,
provenance, ownership, and inventory evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

StateStatus = Literal["PRESENT", "NOT_REQUIRED_ACROSS_DECODE", "STATIC_MODEL_STATE", "MISSING"]


@dataclass(frozen=True)
class StateInventoryEntry:
    name: str
    status: StateStatus
    owner: str
    position: int | None = None
    shape: list[int] | None = None
    dtype: str | None = None
    provenance: str | None = None
    digest: str | None = None
    note: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {k: v for k, v in {
            "name": self.name,
            "status": self.status,
            "owner": self.owner,
            "position": self.position,
            "shape": self.shape,
            "dtype": self.dtype,
            "provenance": self.provenance,
            "digest": self.digest,
            "note": self.note,
        }.items() if v is not None}


@dataclass
class PrefillContinuationState:
    """Live model-semantic continuation state for a future decode backend.

    The object is intentionally architecture-neutral: arrays are keyed by their
    official/model-semantic role instead of by a DwarfStar or oMLX cache layout.
    A future decode runtime may consume these arrays directly or adapt them after
    Milestone 3 selects the decode architecture.
    """

    token_ids: np.ndarray
    token_frontier: int
    ngram_hashes: dict[str, np.ndarray]
    engram_store: dict[str, Any]
    window_kv_by_layer: dict[int, np.ndarray]
    compressed_kv_by_source: dict[int, np.ndarray]
    index_k_by_source: dict[int, np.ndarray]
    candidates_by_source: dict[int, np.ndarray]
    topk_by_generation: dict[int, np.ndarray]
    field_ownership: dict[str, int]
    compressor_pending: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    shared_publications: dict[str, np.ndarray | None] = field(default_factory=dict)
    source_generation_order: list[str] = field(default_factory=list)
    committed: bool = False

    def commit(self) -> "PrefillContinuationState":
        self.committed = True
        # Detach from executor scratch.  ndarray copies also make fork/reset
        # semantics explicit for later decode candidates.
        self.token_ids = np.array(self.token_ids, copy=True)
        self.ngram_hashes = {k: np.array(v, copy=True) for k, v in self.ngram_hashes.items()}
        self.window_kv_by_layer = {k: np.array(v, copy=True) for k, v in self.window_kv_by_layer.items()}
        self.compressed_kv_by_source = {k: np.array(v, copy=True) for k, v in self.compressed_kv_by_source.items()}
        self.index_k_by_source = {k: np.array(v, copy=True) for k, v in self.index_k_by_source.items()}
        self.candidates_by_source = {k: np.array(v, copy=True) for k, v in self.candidates_by_source.items()}
        self.topk_by_generation = {k: np.array(v, copy=True) for k, v in self.topk_by_generation.items()}
        self.compressor_pending = {
            k: {name: np.array(value, copy=True) for name, value in pending.items()}
            for k, pending in self.compressor_pending.items()
        }
        self.shared_publications = {
            k: (None if v is None else np.array(v, copy=True)) for k, v in self.shared_publications.items()
        }
        return self

    def fork(self) -> "PrefillContinuationState":
        return PrefillContinuationState(
            token_ids=np.array(self.token_ids, copy=True),
            token_frontier=self.token_frontier,
            ngram_hashes={k: np.array(v, copy=True) for k, v in self.ngram_hashes.items()},
            engram_store=dict(self.engram_store),
            window_kv_by_layer={k: np.array(v, copy=True) for k, v in self.window_kv_by_layer.items()},
            compressed_kv_by_source={k: np.array(v, copy=True) for k, v in self.compressed_kv_by_source.items()},
            index_k_by_source={k: np.array(v, copy=True) for k, v in self.index_k_by_source.items()},
            candidates_by_source={k: np.array(v, copy=True) for k, v in self.candidates_by_source.items()},
            topk_by_generation={k: np.array(v, copy=True) for k, v in self.topk_by_generation.items()},
            field_ownership=dict(self.field_ownership),
            compressor_pending={k: {n: np.array(v, copy=True) for n, v in p.items()} for k, p in self.compressor_pending.items()},
            shared_publications={k: (None if v is None else np.array(v, copy=True)) for k, v in self.shared_publications.items()},
            source_generation_order=list(self.source_generation_order),
            committed=self.committed,
        )

    def reset(self) -> None:
        self.token_ids = np.empty((1, 0), dtype=np.int64)
        self.token_frontier = 0
        self.ngram_hashes.clear()
        self.window_kv_by_layer.clear()
        self.compressed_kv_by_source.clear()
        self.index_k_by_source.clear()
        self.candidates_by_source.clear()
        self.topk_by_generation.clear()
        self.compressor_pending.clear()
        self.shared_publications.clear()
        self.field_ownership.clear()
        self.source_generation_order.clear()
        self.committed = False

    def inventory(self, digest_fn) -> dict[str, Any]:
        entries: list[StateInventoryEntry] = []
        entries.append(StateInventoryEntry("token/ngram history", "PRESENT", "PrefillContinuationState.token_ids/ngram_hashes", self.token_frontier, list(self.token_ids.shape), str(self.token_ids.dtype), "prefill input tokens plus regenerated Engram hashes", digest_fn(self.token_ids)))
        entries.append(StateInventoryEntry("Engram SSD-backed store", "STATIC_MODEL_STATE", "model/checkpoint Engram store handles", self.token_frontier, provenance=str(self.engram_store.get("checkpoint")), note="session stores history/hash state; static table backing is not duplicated"))
        for layer in range(40):
            arr = self.window_kv_by_layer.get(layer)
            entries.append(StateInventoryEntry(f"layer-{layer} window KV", "PRESENT" if arr is not None else "MISSING", f"window_kv_by_layer[{layer}]", self.token_frontier, None if arr is None else list(arr.shape), None if arr is None else str(arr.dtype), "Block attention window publication", None if arr is None else digest_fn(arr)))
        for source in (2, 8, 14, 20):
            ckv = self.compressed_kv_by_source.get(source)
            idx = self.index_k_by_source.get(source)
            entries.append(StateInventoryEntry(f"source@{source} compressed KV", "PRESENT" if ckv is not None else "MISSING", f"compressed_kv_by_source[{source}]", self.token_frontier, None if ckv is None else list(ckv.shape), None if ckv is None else str(ckv.dtype), f"full source generation @{source}", None if ckv is None else digest_fn(ckv)))
            entries.append(StateInventoryEntry(f"source@{source} index K", "PRESENT" if idx is not None else "MISSING", f"index_k_by_source[{source}]", self.token_frontier, None if idx is None else list(idx.shape), None if idx is None else str(idx.dtype), f"full source generation @{source}", None if idx is None else digest_fn(idx)))
            pending = self.compressor_pending.get(source, {})
            entries.append(StateInventoryEntry(f"source@{source} compressor pending KV/score", "PRESENT" if pending else "NOT_REQUIRED_ACROSS_DECODE", f"compressor_pending[{source}]", self.token_frontier, provenance=f"compress ratio group completed for bounded prefill source@{source}", note="empty means no partial compression group remains to carry"))
        cand = self.candidates_by_source.get(20)
        entries.append(StateInventoryEntry("candidate state source@20", "PRESENT" if cand is not None else "MISSING", "candidates_by_source[20]", self.token_frontier, None if cand is None else list(cand.shape), None if cand is None else str(cand.dtype), "candidate source layer 20", None if cand is None else digest_fn(cand)))
        for generation in (2, 8, 14, 20, 24, 28, 32, 36):
            topk = self.topk_by_generation.get(generation)
            entries.append(StateInventoryEntry(f"top-k/index generation@{generation}", "PRESENT" if topk is not None else "MISSING", f"topk_by_generation[{generation}]", self.token_frontier, None if topk is None else list(topk.shape), None if topk is None else str(topk.dtype), "full source or index-refresh generation", None if topk is None else digest_fn(topk)))
        entries.append(StateInventoryEntry("HC residual/pre-mix", "NOT_REQUIRED_ACROSS_DECODE", "qualification evidence only", self.token_frontier, note="HC pre_mix is call-local across block/subblock boundaries; final residual/pre_mix digests remain artifact evidence, not decoder state"))
        return {
            "schema": "ds41f.prefill-continuation-inventory.v1",
            "committed": self.committed,
            "token_frontier": self.token_frontier,
            "field_ownership": dict(self.field_ownership),
            "source_generation_order": list(self.source_generation_order),
            "entries": [entry.to_json() for entry in entries],
            "missing": [entry.name for entry in entries if entry.status == "MISSING"],
        }


@dataclass(frozen=True)
class PrefillSessionHandoff:
    schema: str
    token_frontier: int
    tokens_digest: str
    last_logits_digest: str
    committed_shared_state: dict[str, str | None]
    continuation_state: PrefillContinuationState
    current_hc_digest: str
    pre_mix_digest: str
    engram_history: dict[str, str | None]
    decode_architecture_selected: bool = False
    requires_no_prefill_recompute: bool = True
    extra_state: dict[str, Any] | None = None

    def to_artifact(self, digest_fn) -> dict[str, Any]:
        inventory = self.continuation_state.inventory(digest_fn)
        return {
            "schema": self.schema,
            "classification": "architecture_neutral_prefill_to_decode_contract",
            "token_frontier": self.token_frontier,
            "tokens_digest": self.tokens_digest,
            "last_logits_digest": self.last_logits_digest,
            "committed_shared_state": self.committed_shared_state,
            "current_hc_digest": self.current_hc_digest,
            "pre_mix_digest": self.pre_mix_digest,
            "hc_pre_mix_classification": "qualification_evidence_not_persistent_decode_state",
            "engram_history": self.engram_history,
            "continuation_state_inventory": inventory,
            "live_state_available": self.continuation_state.committed,
            "decode_architecture_selected": self.decode_architecture_selected,
            "requires_no_prefill_recompute": self.requires_no_prefill_recompute,
            "extra_state": self.extra_state or {},
        }
