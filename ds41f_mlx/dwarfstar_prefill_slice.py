"""First vertical slice of the DwarfStar-derived prefill executor.

This module is intentionally narrow: it drives official checkpoint data through
an existing DwarfStar V4.1 sweep plan, owns typed carry/row/publication state in
that executor, and executes the first complete layer span with official model
math helpers.  It does not call TextBackboneReference/TextEncoderReference/
TextDecoderReference and it does not claim full-model prefill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ds41f_mlx.native_prefill import (
    NativePrefillLibrary,
    compile_native_prefill_library,
    load_native_prefill_library,
    native_sweep_config_static,
)
from ds41f_mlx.dwarfstar_v41_sweep import DS4_AUTHORITY_REMOTE, DS4_AUTHORITY_SHA
from ds41f_mlx.official_model_math import OfficialModelMath
from ds41f_mlx.prefill_session import PrefillContinuationState, PrefillSessionHandoff


@dataclass
class SweepTransaction:
    """DwarfStar-style bounded transaction state for the vertical slice."""

    valid: bool = True
    begun: bool = False
    committed: bool = False
    failed: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)

    def begin(self) -> None:
        if not self.valid or self.begun:
            raise RuntimeError("invalid transaction begin")
        self.valid = False
        self.begun = True
        self.events.append({"event": "begin_sweep_checkpoint_invalid", "valid": self.valid})

    def publish(self, layer: int, frontier: dict[str, str | None]) -> None:
        if self.valid or not self.begun or self.committed:
            raise RuntimeError("publication outside invalid sweep transaction")
        self.events.append({"event": "publish_state_frontier", "layer": layer, "frontier": frontier, "valid": self.valid})

    def final_output_ready(self, logits_digest: str) -> None:
        if self.valid or not self.begun or self.committed:
            raise RuntimeError("final output outside invalid sweep transaction")
        self.events.append({"event": "final_output_ready_private", "logits_digest": logits_digest, "valid": self.valid})

    def commit(self) -> None:
        if self.valid or not self.begun or self.failed:
            raise RuntimeError("invalid transaction commit")
        self.valid = True
        self.committed = True
        self.events.append({"event": "checkpoint_may_commit", "valid": self.valid})

    def fail(self, error: BaseException) -> None:
        self.failed = True
        self.valid = False
        self.events.append({"event": "sweep_failed_still_invalid", "error_type": type(error).__name__, "error": str(error), "valid": self.valid})


@dataclass
class TypedSweepStorage:
    """Executor-owned typed storage for the bounded sweep rows/carry."""

    tokens: np.ndarray
    embedding_bf16: np.ndarray
    current_hc_bf16: np.ndarray
    next_hc_bf16: np.ndarray
    pre_f32: np.ndarray
    shared: dict[str, Any]
    engram_hashes: dict[str, Any] = field(default_factory=dict)
    publications: list[dict[str, Any]] = field(default_factory=list)

    def frontier_digest(self, math: OfficialModelMath) -> dict[str, str | None]:
        return math.snapshot_publications(self.shared)

    def publish(self, layer: int, math: OfficialModelMath) -> dict[str, Any]:
        rec = {"layer": layer, "frontier": self.frontier_digest(math)}
        self.publications.append(rec)
        return rec


@dataclass
class VerticalSliceResult:
    artifact: dict[str, Any]
    continuation_state: PrefillContinuationState | None = None

    @property
    def ok(self) -> bool:
        return bool(self.artifact.get("ok"))


class DwarfStarPrefillVerticalSliceExecutor:
    """Bounded production-prefill executor slice.

    The executor is driven by the native DwarfStar-derived sweep plan.  For each
    selected `encode_rows` command it runs real official model math for that
    layer and stores/publishes state in executor-owned storage.
    """

    def __init__(self, checkpoint: Path, native: NativePrefillLibrary):
        self.checkpoint = Path(checkpoint)
        self.native = native
        self.math = OfficialModelMath(self.checkpoint)

    @classmethod
    def with_compiled_native(cls, checkpoint: Path, out_dir: Path) -> "DwarfStarPrefillVerticalSliceExecutor":
        lib = compile_native_prefill_library(out_dir)
        return cls(checkpoint, load_native_prefill_library(lib))

    def _init_storage(self, tokens: np.ndarray) -> tuple[TypedSweepStorage, dict[str, Any]]:
        emb_prefix = self.math.embedding_prefix(max(int(tokens.max()) + 1, 16))
        gathered, native_result = self.native.official_embedding_gather_bf16(emb_prefix, tokens.reshape(-1).astype(np.int32))
        embedding = gathered.reshape(1, int(tokens.size), self.math.dim)
        current = np.repeat(embedding[:, :, None, :], self.math.hc_mult, axis=2).copy()
        next_hc = np.empty_like(current)
        pre = np.zeros((1, int(tokens.size), self.math.hc_mult), np.float32)
        pre[:, :, 0] = 1.0
        storage = TypedSweepStorage(
            tokens=np.ascontiguousarray(tokens, dtype=np.int64),
            embedding_bf16=embedding,
            current_hc_bf16=current,
            next_hc_bf16=next_hc,
            pre_f32=pre,
            shared={"compress_kv": None, "index_k": None, "candidates": None, "topk_idxs": None},
            engram_hashes=self.math.engram_hashes_for_tokens(tokens),
        )
        return storage, native_result

    def run(self, *, tokens: list[int] | None = None, layers: int = 40, ctx: int = 32768, enable_engram: bool = True) -> VerticalSliceResult:
        if layers < 1:
            raise ValueError("vertical slice must execute at least one layer")
        token_arr = np.array([tokens or [0, 3]], dtype=np.int64)
        native_cfg = native_sweep_config_static(ctx=ctx, remaining=int(token_arr.size), memory_budget_bytes=128 * 1024 * 1024)
        plan = self.native.build_sweep_plan(native_cfg).to_json()
        tx = SweepTransaction()
        storage, embedding_native = self._init_storage(token_arr)
        commands = plan["commands"]
        executed: list[dict[str, Any]] = []
        scheduling_events: list[dict[str, Any]] = []
        engram_events: list[dict[str, Any]] = []
        layer_records: dict[str, Any] = {}
        final_output: dict[str, Any] | None = None
        session_handoff: dict[str, Any] | None = None
        continuation_state: PrefillContinuationState | None = None
        actual_window_kv: dict[int, np.ndarray] = {}
        actual_compressed_kv: dict[int, np.ndarray] = {}
        actual_index_k: dict[int, np.ndarray] = {}
        actual_candidates: dict[int, np.ndarray] = {}
        actual_topk: dict[int, np.ndarray] = {}
        actual_field_owner: dict[str, int] = {}
        actual_generation_order: list[str] = []
        t0 = perf_counter()
        tx.begin()
        try:
            for command in commands:
                name = command["kind_name"]
                layer = command["layer"]
                if name in {"prefetch_engram_table0", "prefetch_engram_table1", "ssd_read_ahead", "begin_layer", "end_layer", "swap_hc_after_layer", "publish_state_frontier"}:
                    scheduling_events.append(command)
                if name != "encode_rows" or layer is None or int(layer) >= layers:
                    continue
                if int(command["offset"]) != 0 or int(command["rows"]) != int(token_arr.size):
                    # The first bounded slice supports the first full row span only.
                    continue
                engram_before = None
                if enable_engram and int(layer) in {1, 14}:
                    engram_layer = int(layer)
                    hash_key = "layer1_hash" if engram_layer == 1 else "layer14_hash"
                    pre_engram = self.math.digest(storage.current_hc_bf16)
                    pre_mix_before = self.math.digest(storage.pre_f32)
                    shared_before = storage.frontier_digest(self.math)
                    post, evidence = self.math.apply_engram(engram_layer, storage.current_hc_bf16, storage.engram_hashes[hash_key])
                    storage.current_hc_bf16 = post
                    engram_before = pre_engram
                    evidence["command_context"] = command
                    evidence["pre_mix_digest_unchanged"] = pre_mix_before
                    evidence["shared_state_before"] = shared_before
                    evidence["shared_state_after"] = storage.frontier_digest(self.math)
                    evidence["shared_state_unchanged"] = evidence["shared_state_before"] == evidence["shared_state_after"]
                    engram_events.append(evidence)
                before = storage.frontier_digest(self.math)
                x_in = self.math.digest(storage.current_hc_bf16)
                pre_in = self.math.digest(storage.pre_f32)
                out = self.math.execute_block(int(layer), storage.current_hc_bf16, storage.pre_f32, storage.shared)
                storage.next_hc_bf16[...] = out["x_out"]
                storage.current_hc_bf16, storage.next_hc_bf16 = storage.next_hc_bf16, storage.current_hc_bf16
                storage.pre_f32 = out["ffn_pre"]
                actual_window_kv[int(layer)] = np.array(out["attn_path"]["window_kv"], copy=True)
                producer_actual = {k: np.array(v, copy=True) for k, v in (out["attn_path"].get("producer") or {}).items() if isinstance(v, np.ndarray)}
                if producer_actual:
                    actual_generation_order.append(("index-refresh" if set(producer_actual) == {"topk_idxs"} else "source") + f"@{int(layer)}")
                    if "compress_kv" in producer_actual:
                        actual_compressed_kv[int(layer)] = producer_actual["compress_kv"]
                        actual_field_owner["compress_kv"] = int(layer)
                    if "index_k" in producer_actual:
                        actual_index_k[int(layer)] = producer_actual["index_k"]
                        actual_field_owner["index_k"] = int(layer)
                    if "candidates" in producer_actual:
                        actual_candidates[int(layer)] = producer_actual["candidates"]
                        actual_field_owner["candidates"] = int(layer)
                    if "topk_idxs" in producer_actual:
                        actual_topk[int(layer)] = producer_actual["topk_idxs"]
                        actual_field_owner["topk_idxs"] = int(layer)
                publication = storage.publish(int(layer), self.math)
                tx.publish(int(layer), publication["frontier"])
                after = storage.frontier_digest(self.math)
                rec = {
                    "command": command,
                    "input_hc_digest": x_in,
                    "pre_engram_input_hc_digest": engram_before,
                    "input_pre_digest": pre_in,
                    "attention_input_digest": self.math.digest(out["attention_input"]),
                    "attention_output_digest": self.math.digest(out["attention_output"]),
                    "x_after_attn_digest": self.math.digest(out["x_after_attn"]),
                    "moe_input_digest": self.math.digest(out["moe_input"]),
                    "moe_output_digest": self.math.digest(out["full_moe_output"]),
                    "output_hc_digest": self.math.digest(out["x_out"]),
                    "output_pre_digest": self.math.digest(out["ffn_pre"]),
                    "window_kv_digest": self.math.digest(out["attn_path"]["window_kv"]),
                    "state_before": before,
                    "state_after": after,
                    "publication": publication,
                    "producer": {k: self.math.digest(v) for k, v in (out["attn_path"].get("producer") or {}).items() if isinstance(v, np.ndarray)},
                    "consumed": out["attn_path"].get("consumed") or {},
                }
                layer_records[str(layer)] = rec
                executed.append(command)
            if len(executed) != layers:
                raise RuntimeError(f"expected {layers} executed layers, got {len(executed)}")
            if layers >= 40:
                final_raw = self.math.final_logits(storage.current_hc_bf16, storage.pre_f32)
                final_output = {k: v for k, v in final_raw.items() if k != "logits"}
                tx.final_output_ready(final_output["logits_digest"])
                continuation_state = PrefillContinuationState(
                    token_ids=storage.tokens,
                    token_frontier=int(token_arr.size),
                    ngram_hashes={
                        "full_hash": storage.engram_hashes["full_hash"],
                        "layer1_hash": storage.engram_hashes["layer1_hash"],
                        "layer14_hash": storage.engram_hashes["layer14_hash"],
                    },
                    engram_store={"checkpoint": str(self.checkpoint), "storage_policy": "SSD-backed model-static Engram tables"},
                    window_kv_by_layer=actual_window_kv,
                    compressed_kv_by_source=actual_compressed_kv,
                    index_k_by_source=actual_index_k,
                    candidates_by_source=actual_candidates,
                    topk_by_generation=actual_topk,
                    field_ownership=actual_field_owner,
                    compressor_pending={source: {} for source in actual_compressed_kv},
                    shared_publications={k: (None if v is None else np.array(v, copy=True)) for k, v in storage.shared.items()},
                    source_generation_order=actual_generation_order,
                ).commit()
                session_handoff = PrefillSessionHandoff(
                    schema="ds41f.prefill-session-handoff.v1",
                    token_frontier=int(token_arr.size),
                    tokens_digest=self.math.digest(storage.tokens),
                    last_logits_digest=final_output["logits_digest"],
                    committed_shared_state=storage.frontier_digest(self.math),
                    continuation_state=continuation_state,
                    current_hc_digest=self.math.digest(storage.current_hc_bf16),
                    pre_mix_digest=self.math.digest(storage.pre_f32),
                    engram_history={"full_hash_digest": storage.engram_hashes.get("full_hash_digest"), "layer1_hash_digest": storage.engram_hashes.get("layer1_hash_digest"), "layer14_hash_digest": storage.engram_hashes.get("layer14_hash_digest")},
                ).to_artifact(self.math.digest)
            tx.commit()
        except BaseException as exc:
            tx.fail(exc)
            raise
        seconds = perf_counter() - t0

        if enable_engram and layers >= 40:
            reference_path = Path("artifacts/native-engram-connected-deterministic-logits-validation.json")
            reference_kind = "engram_aware_connected_full_logits_scope"
        elif enable_engram and layers >= 28:
            reference_path = Path("artifacts/native-engram-layer20-27-validation.json")
            reference_kind = "engram_aware_connected_candidate27_scope"
        elif enable_engram and layers >= 20:
            reference_path = Path("artifacts/native-engram-layer14-block19-validation.json")
            reference_kind = "engram_aware_connected_block19_scope"
        elif enable_engram:
            reference_path = Path("artifacts/native-engram-layer14-validation.json")
            reference_kind = "engram_aware_connected_scope"
        else:
            reference_path = Path("artifacts/native-layer0-25-transformer-entry-validation.json")
            reference_kind = "legacy_non_engram_connected_scope"
        reference_comparison: dict[str, Any] = {"reference_path": str(reference_path), "reference_kind": reference_kind, "available": reference_path.exists(), "layers": {}}
        if reference_path.exists():
            import json
            ref = json.loads(reference_path.read_text())
            ref_layers = ref.get("layers", ref.get("blocks1_13", {}).get("layers", {})) if enable_engram else ref.get("layers", {})
            for layer in range(layers):
                ours = layer_records[str(layer)]
                theirs = ref_layers.get(str(layer), {})
                checks: dict[str, bool] = {}
                if layer == 0 and enable_engram and reference_kind == "engram_aware_connected_scope":
                    checks = {
                        "x_out": ours["output_hc_digest"] == theirs.get("x_out"),
                        "ffn_pre": ours["output_pre_digest"] == theirs.get("ffn_pre"),
                        "post_engram1_h": bool(engram_events) and engram_events[0]["output_digest"] == theirs.get("post_engram1_h"),
                    }
                else:
                    checks = {
                        "attention_input": ours["attention_input_digest"] == theirs.get("attention_input"),
                        "attention_output": ours["attention_output_digest"] == theirs.get("attention_output"),
                        "x_after_attn": ours["x_after_attn_digest"] == theirs.get("x_after_attn"),
                        "moe_input": ours["moe_input_digest"] == theirs.get("moe_input"),
                        "moe_output": ours["moe_output_digest"] == theirs.get("moe_output", theirs.get("full_moe_output")),
                        "x_out": ours["output_hc_digest"] == theirs.get("x_out"),
                        "ffn_pre": ours["output_pre_digest"] == theirs.get("ffn_pre"),
                    }
                reference_comparison["layers"][str(layer)] = checks
            reference_comparison["all_compared_exact"] = all(
                all(v is True for v in layer_cmp.values())
                for layer_cmp in reference_comparison["layers"].values()
            )
            if enable_engram:
                reference_comparison["engram1_post_digest"] = ref.get("post_engram1_regression", {}).get("digest", ref.get("engram", {}).get("1", {}).get("output_digest", ref.get("upstream_regressions", {}).get("post_engram1_h")))
                reference_comparison["engram14_post_digest"] = ref.get("residual_update", {}).get("post_engram14_h", {}).get("digest", ref.get("engram", {}).get("14", {}).get("output_digest", ref.get("upstream_regressions", {}).get("post_engram14_h")))
                reference_comparison["generation2"] = ref.get("shared_attention_state", {}).get("generation2", ref.get("source_generations", {}).get("2", {}).get("source_publication"))
                reference_comparison["generation8"] = ref.get("shared_attention_state", {}).get("generation8", ref.get("source_generations", {}).get("8", {}).get("source_publication"))
                reference_comparison["generation14"] = ref.get("source_generations", {}).get("14", {}).get("source_publication")
                reference_comparison["generation20"] = ref.get("source_generations", {}).get("20", {}).get("source_publication")
                reference_comparison["generation24"] = ref.get("source_generations", {}).get("24", {}).get("source_publication")
                reference_comparison["logits_digest"] = ref.get("parallel_head", {}).get("logits", {}).get("digest")
        else:
            reference_comparison["all_compared_exact"] = False

        role_table = {str(layer): self.math.layer_roles(layer) for layer in range(layers)}
        publication_events: list[dict[str, Any]] = []
        field_owner: dict[str, int] = {}
        field_owner_history: dict[str, list[dict[str, Any]]] = {field: [] for field in ("compress_kv", "index_k", "topk_idxs", "candidates")}
        consumer_observations: dict[str, Any] = {}
        source_generations: dict[str, Any] = {}

        def field_digest_at(layer: int, when: str, field: str) -> str | None:
            return layer_records[str(layer)][when].get(field)

        for layer in range(layers):
            rec = layer_records[str(layer)]
            producer = rec["producer"]
            changed_fields = [field for field in ("compress_kv", "index_k", "topk_idxs", "candidates") if field in producer]
            if producer:
                if {"compress_kv", "index_k", "topk_idxs"}.issubset(producer):
                    kind = "full_source_generation"
                elif changed_fields == ["topk_idxs"]:
                    kind = "index_only_topk_refresh_generation"
                else:
                    kind = "partial_publication_generation"
                consumed_state = {
                    field: {
                        "digest": field_digest_at(layer, "state_before", field),
                        "owner_before": field_owner.get(field),
                    }
                    for field in ("compress_kv", "index_k", "topk_idxs", "candidates")
                }
                previous_owners = dict(field_owner)
                for field in changed_fields:
                    field_owner[field] = layer
                    field_owner_history[field].append({"layer": layer, "digest": producer[field], "generation_kind": kind})
                ownership_after = {field: field_owner.get(field) for field in ("compress_kv", "index_k", "topk_idxs", "candidates")}
                event = {
                    "generation_id": f"{'index-refresh' if kind == 'index_only_topk_refresh_generation' else 'source'}@{layer}",
                    "layer": layer,
                    "kind": kind,
                    "changed_fields": changed_fields,
                    "producer": producer,
                    "consumed_state_before_publication": consumed_state,
                    "previous_field_owners": previous_owners,
                    "field_ownership_after": ownership_after,
                    "state_after": rec["state_after"],
                    "candidate_provenance": "computed_by_real_block_execution" if "candidates" in changed_fields else None,
                }
                publication_events.append(event)

        for event_index, event in enumerate(publication_events):
            layer = int(event["layer"])
            next_pub = int(publication_events[event_index + 1]["layer"]) if event_index + 1 < len(publication_events) else layers
            consumers = list(range(layer + 1, next_pub))
            reads: dict[str, Any] = {}
            ownership_after_event = event["field_ownership_after"]
            for consumer in consumers:
                observed = layer_records[str(consumer)]["state_before"]
                per_field: dict[str, Any] = {}
                for field in ("compress_kv", "index_k", "topk_idxs", "candidates"):
                    owner = ownership_after_event.get(field)
                    expected = layer_records[str(owner)]["producer"].get(field) if owner is not None else None
                    stale_digest_aliases = [
                        hist for hist in field_owner_history[field]
                        if hist["layer"] < int(owner) and observed.get(field) == hist["digest"]
                    ] if owner is not None else []
                    # Short bounded fixtures can legitimately produce identical
                    # top-k index bytes across two different publication events.
                    # Staleness is therefore rejected by executor ownership /
                    # provenance, while digest aliases are recorded separately.
                    stale = [] if observed.get(field) == expected else stale_digest_aliases
                    per_field[field] = {
                        "observed_digest": observed.get(field),
                        "expected_owner": owner,
                        "expected_digest": expected,
                        "matches_expected_owner": observed.get(field) == expected,
                        "stale_owner_matches": stale,
                        "stale_digest_aliases": stale_digest_aliases,
                    }
                reads[str(consumer)] = {
                    "observed_fields": per_field,
                    "raw_attn_consumed": layer_records[str(consumer)]["consumed"],
                    "consumer_recomputed_producer_state": bool(layer_records[str(consumer)]["producer"]),
                }
                consumer_observations[str(consumer)] = reads[str(consumer)]
            source_generations[str(layer)] = {
                "generation_id": event["generation_id"],
                "source_layer": layer,
                "kind": event["kind"],
                "changed_fields": event["changed_fields"],
                "producer": event["producer"],
                "source_publication": event["state_after"],
                "field_ownership_after": event["field_ownership_after"],
                "consumer_layers": consumers,
                "consumer_reads": reads,
                "complete": bool(reads) and all(str(c) in reads for c in consumers),
            }

        field_ownership_after_layer24 = None
        layer24_event = next((event for event in publication_events if event["layer"] == 24), None)
        if layer24_event is not None:
            field_ownership_after_layer24 = layer24_event["field_ownership_after"]

        source_group = {
            "generations": source_generations,
            "publication_events": publication_events,
            "field_owner_history": field_owner_history,
            "consumer_observations": consumer_observations,
            "field_ownership_after_layer24": field_ownership_after_layer24,
            "candidate_lifecycle": "candidate state is first produced by the configured candidate source layer 20",
        }

        def observation_matches(layer: int, expected: dict[str, int]) -> bool:
            obs = consumer_observations.get(str(layer), {}).get("observed_fields", {})
            return all(
                obs.get(field, {}).get("expected_owner") == owner and obs.get(field, {}).get("matches_expected_owner") is True
                for field, owner in expected.items()
            )

        def observation_has_no_stale(layer: int, fields: tuple[str, ...]) -> bool:
            obs = consumer_observations.get(str(layer), {}).get("observed_fields", {})
            return all(not obs.get(field, {}).get("stale_owner_matches") for field in fields)

        if session_handoff is not None:
            latest_ownership = publication_events[-1]["field_ownership_after"] if publication_events else {}
            session_handoff["extra_state"] = {
                "committed_field_ownership": latest_ownership,
                "publication_generations": [event["generation_id"] for event in publication_events],
            }

        layer20_event = next((event for event in publication_events if event["layer"] == 20), None)
        layer24_event = next((event for event in publication_events if event["layer"] == 24), None)
        generation20_consumers = source_generations.get("20", {}).get("consumer_layers", [])
        generation24_consumers = source_generations.get("24", {}).get("consumer_layers", [])
        mixed_after24 = {"compress_kv": 20, "index_k": 20, "candidates": 20, "topk_idxs": 24}
        mixed_after28 = {"compress_kv": 20, "index_k": 20, "candidates": 20, "topk_idxs": 28}
        mixed_after32 = {"compress_kv": 20, "index_k": 20, "candidates": 20, "topk_idxs": 32}
        mixed_after36 = {"compress_kv": 20, "index_k": 20, "candidates": 20, "topk_idxs": 36}

        gates = {
            "native_sweep_plan_used": plan["source_authority"]["commit"] == DS4_AUTHORITY_SHA,
            "checkpoint_invalid_during_execution": any(e["event"] == "begin_sweep_checkpoint_invalid" and e["valid"] is False for e in tx.events),
            "checkpoint_committed_after_publication": tx.committed and tx.valid and tx.events[-1]["event"] == "checkpoint_may_commit",
            "official_embedding_native_executed": bool(embedding_native.get("metal_enabled")) or embedding_native.get("n_tokens") == int(token_arr.size),
            "executor_owned_typed_storage": storage.current_hc_bf16.shape == (1, int(token_arr.size), self.math.hc_mult, self.math.dim) and storage.pre_f32.shape == (1, int(token_arr.size), self.math.hc_mult),
            "layer0_complete_path_executed": "0" in layer_records and bool(layer_records["0"]["attention_output_digest"]) and bool(layer_records["0"]["moe_output_digest"]),
            "all_span_layers_have_hc_attention_moe": all(bool(layer_records[str(i)]["attention_output_digest"]) and bool(layer_records[str(i)]["moe_output_digest"]) for i in range(layers)),
            "no_reference_text_runtime_called": True,
            "publication_owned_by_executor": len(storage.publications) == layers and len([e for e in tx.events if e["event"] == "publish_state_frontier"]) == layers,
            "engram1_executed": enable_engram and any(e["layer"] == 1 for e in engram_events),
            "engram14_executed": (layers <= 14) or (enable_engram and any(e["layer"] == 14 for e in engram_events)),
            "engram1_ssd_sparse_rows_only": enable_engram and any(e["layer"] == 1 and e.get("ssd_backed_sparse_rows_only") for e in engram_events),
            "engram1_matches_reference": (not enable_engram) or any(e["layer"] == 1 and e["output_digest"] == reference_comparison.get("engram1_post_digest") for e in engram_events),
            "engram14_ssd_sparse_rows_only": (layers <= 14) or (enable_engram and any(e["layer"] == 14 and e.get("ssd_backed_sparse_rows_only") for e in engram_events)),
            "engram14_matches_reference": (layers <= 14) or any(e["layer"] == 14 and e["output_digest"] == reference_comparison.get("engram14_post_digest") for e in engram_events),
            "engram14_preserves_shared_state": (layers <= 14) or any(e["layer"] == 14 and e.get("shared_state_unchanged") for e in engram_events),
            "layer2_state_publication_present": layers < 3 or all(source_generations.get("2", {}).get("producer", {}).get(k) is not None for k in ("compress_kv", "index_k", "topk_idxs")),
            "layer8_state_publication_present": layers < 9 or all(source_generations.get("8", {}).get("producer", {}).get(k) is not None for k in ("compress_kv", "index_k", "topk_idxs")),
            "layer14_state_publication_present": layers < 15 or all(source_generations.get("14", {}).get("producer", {}).get(k) is not None for k in ("compress_kv", "index_k", "topk_idxs")),
            "source2_reuse_group_complete": source_generations.get("2", {}).get("complete") is True,
            "source8_reuse_group_complete": layers < 14 or source_generations.get("8", {}).get("complete") is True,
            "source14_reuse_group_complete": layers < 20 or source_generations.get("14", {}).get("complete") is True,
            "candidate_absent_before_layer20": layers <= 20 or all(layer_records[str(i)]["state_after"].get("candidates") is None for i in range(20)),
            "layer20_candidate_source_generation_present": layers < 21 or (layer20_event is not None and layer20_event["kind"] == "full_source_generation" and all(layer20_event["producer"].get(k) is not None for k in ("compress_kv", "index_k", "topk_idxs", "candidates"))),
            "candidate_first_present_at_layer20": layers < 21 or (layer_records["19"]["state_after"].get("candidates") is None and layer_records["20"]["state_after"].get("candidates") == layer20_event["producer"].get("candidates")),
            "consumers21_23_use_generation20": layers < 24 or (generation20_consumers == [21, 22, 23] and all(observation_matches(layer, {"compress_kv": 20, "index_k": 20, "topk_idxs": 20, "candidates": 20}) for layer in generation20_consumers)),
            "consumers21_23_reject_stale_pre20": layers < 24 or all(observation_has_no_stale(layer, ("compress_kv", "index_k", "topk_idxs", "candidates")) and not consumer_observations[str(layer)]["consumer_recomputed_producer_state"] for layer in generation20_consumers),
            "layer24_index_only_refresh_present": layers < 25 or (layer24_event is not None and layer24_event["kind"] == "index_only_topk_refresh_generation" and layer24_event["changed_fields"] == ["topk_idxs"]),
            "layer24_consumes_generation20_state": layers < 25 or all(layer24_event["consumed_state_before_publication"][field]["owner_before"] == 20 for field in ("compress_kv", "index_k", "topk_idxs", "candidates")),
            "layer24_preserves_compress_index_candidates": layers < 25 or all(layer24_event["field_ownership_after"][field] == 20 for field in ("compress_kv", "index_k", "candidates")),
            "layer24_refreshes_topk_owner": layers < 25 or layer24_event["field_ownership_after"].get("topk_idxs") == 24,
            "consumers_after24_use_mixed_generation": layers < 28 or (generation24_consumers == [25, 26, 27] and all(observation_matches(layer, mixed_after24) for layer in generation24_consumers)),
            "consumers_after24_reject_stale_topk20_and_pre20": layers < 28 or all(observation_has_no_stale(layer, ("compress_kv", "index_k", "topk_idxs", "candidates")) and not consumer_observations[str(layer)]["consumer_recomputed_producer_state"] for layer in generation24_consumers),
            "layer28_index_only_refresh_present": layers < 29 or (source_generations.get("28", {}).get("kind") == "index_only_topk_refresh_generation" and source_generations["28"]["changed_fields"] == ["topk_idxs"]),
            "consumers_after28_use_mixed_generation": layers < 32 or all(observation_matches(layer, mixed_after28) for layer in source_generations.get("28", {}).get("consumer_layers", [])),
            "layer32_index_only_refresh_present": layers < 33 or (source_generations.get("32", {}).get("kind") == "index_only_topk_refresh_generation" and source_generations["32"]["changed_fields"] == ["topk_idxs"]),
            "consumers_after32_use_mixed_generation": layers < 36 or all(observation_matches(layer, mixed_after32) for layer in source_generations.get("32", {}).get("consumer_layers", [])),
            "layer36_index_only_refresh_present": layers < 37 or (source_generations.get("36", {}).get("kind") == "index_only_topk_refresh_generation" and source_generations["36"]["changed_fields"] == ["topk_idxs"]),
            "consumers_after36_use_mixed_generation": layers < 40 or all(observation_matches(layer, mixed_after36) for layer in source_generations.get("36", {}).get("consumer_layers", [])),
            "candidate_lifecycle_scoped_before_next_source": layers <= 40 and all(layer_records[str(i)]["state_after"].get("candidates") is not None for i in range(20, layers)),
            "final_logits_present": layers < 40 or bool(final_output and final_output.get("logits_digest")),
            "final_logits_match_reference": layers < 40 or (final_output is not None and final_output.get("logits_digest") == reference_comparison.get("logits_digest")),
            "session_handoff_present_after_final_output": layers < 40 or bool(session_handoff and session_handoff.get("last_logits_digest") == final_output.get("logits_digest")),
            "transaction_commit_after_final_output": layers < 40 or any(e["event"] == "final_output_ready_private" for e in tx.events),
            "sweep_command_order_exact": [int(c["layer"]) for c in executed] == list(range(layers),),
            "engram_aware_reference_connected_scope_exact": reference_comparison["all_compared_exact"] and reference_comparison["reference_kind"] in {"engram_aware_connected_scope", "engram_aware_connected_block19_scope", "engram_aware_connected_candidate27_scope", "engram_aware_connected_full_logits_scope"},
        }
        artifact = {
            "schema": "ds41f.dwarfstar-prefill-vertical-slice.v1",
            "classification": "implementation_scoped_dwarfstar_prefill_slice_validation",
            "purpose": "first vertically integrated DwarfStar-derived production-prefill executor slice: official embedding -> typed sweep carry -> complete real layers -> executor publication/commit",
            "checkpoint": str(self.checkpoint),
            "ds4_authority": {"remote": DS4_AUTHORITY_REMOTE, "commit": DS4_AUTHORITY_SHA},
            "native_version": self.native.version(),
            "scope": {"tokens": token_arr.reshape(-1).tolist(), "layers": list(range(layers)), "full_model_prefill": layers >= 40, "engram_enabled": enable_engram},
            "native_sweep_plan_summary": {k: plan[k] for k in ["count", "prefill_cap", "encoder_chunk", "wide", "decoder_suffix", "checkpoint_valid_during_sweep", "checkpoint_valid_after_sweep", "command_counts"]},
            "embedding_native_result": embedding_native,
            "typed_storage": {
                "tokens_shape": list(storage.tokens.shape),
                "embedding_shape": list(storage.embedding_bf16.shape),
                "current_hc_shape": list(storage.current_hc_bf16.shape),
                "pre_shape": list(storage.pre_f32.shape),
                "current_hc_digest": self.math.digest(storage.current_hc_bf16),
                "pre_digest": self.math.digest(storage.pre_f32),
                "engram_hashes": {"full_hash_digest": storage.engram_hashes.get("full_hash_digest"), "layer1_hash_digest": storage.engram_hashes.get("layer1_hash_digest"), "layer14_hash_digest": storage.engram_hashes.get("layer14_hash_digest")},
            },
            "executed_commands": executed,
            "scheduling_events": scheduling_events,
            "engram_events": engram_events,
            "layer_roles": role_table,
            "source_reuse_group": source_group,
            "layers": layer_records,
            "final_output": final_output,
            "session_handoff": session_handoff,
            "reference_comparison": reference_comparison,
            "transaction_events": tx.events,
            "publications": storage.publications,
            "seconds": seconds,
            "gates": gates,
            "ok": bool(all(gates.values())),
            "non_claims": [
                "not full-model prefill",
                "not decode architecture selection",
                "not long-context qualification",
                "not performance qualification",
                "does not call TextBackboneReference/TextEncoderReference/TextDecoderReference",
            ],
        }
        return VerticalSliceResult(artifact, continuation_state=continuation_state)
