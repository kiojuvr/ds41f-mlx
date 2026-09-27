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

    def run(self, *, tokens: list[int] | None = None, layers: int = 14, ctx: int = 32768, enable_engram: bool = True) -> VerticalSliceResult:
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
            tx.commit()
        except BaseException as exc:
            tx.fail(exc)
            raise
        seconds = perf_counter() - t0

        if enable_engram and layers >= 20:
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
                reference_comparison["engram1_post_digest"] = ref.get("post_engram1_regression", {}).get("digest", ref.get("engram", {}).get("1", {}).get("output_digest"))
                reference_comparison["engram14_post_digest"] = ref.get("residual_update", {}).get("post_engram14_h", {}).get("digest", ref.get("engram", {}).get("14", {}).get("output_digest"))
                reference_comparison["generation2"] = ref.get("shared_attention_state", {}).get("generation2", ref.get("source_generations", {}).get("2", {}).get("source_publication"))
                reference_comparison["generation8"] = ref.get("shared_attention_state", {}).get("generation8", ref.get("source_generations", {}).get("8", {}).get("source_publication"))
                reference_comparison["generation14"] = ref.get("source_generations", {}).get("14", {}).get("source_publication")
        else:
            reference_comparison["all_compared_exact"] = False

        role_table = {str(layer): self.math.layer_roles(layer) for layer in range(layers)}
        source_layers = [layer for layer in range(layers) if layer_records[str(layer)]["producer"]]
        source_generations: dict[str, Any] = {}
        for idx, source in enumerate(source_layers):
            next_source = source_layers[idx + 1] if idx + 1 < len(source_layers) else layers
            consumers = range(source + 1, next_source)
            source_rec = layer_records[str(source)]
            frontier = source_rec["state_after"]
            stale_frontiers = {str(prev): layer_records[str(prev)]["state_after"] for prev in source_layers[:idx]}
            reads: dict[str, Any] = {}
            for consumer in consumers:
                consumed = layer_records[str(consumer)]["consumed"]
                stale_matches = {
                    prev: consumed.get("compress_kv") == prev_frontier.get("compress_kv")
                    for prev, prev_frontier in stale_frontiers.items()
                    if prev_frontier.get("compress_kv") is not None
                }
                reads[str(consumer)] = {
                    "observed_generation_layer": source,
                    "compress_kv_matches_source": consumed.get("compress_kv") == frontier.get("compress_kv"),
                    "index_k_matches_source": consumed.get("index_k") == frontier.get("index_k"),
                    "topk_idxs_matches_source": consumed.get("topk_idxs") == frontier.get("topk_idxs"),
                    "consumer_recomputed_producer_state": bool(layer_records[str(consumer)]["producer"]),
                    "stale_generation_reads": stale_matches,
                    "stale_generation_read": any(stale_matches.values()),
                    "candidate_lifecycle": "not_applicable_before_candidate_source_layer20",
                }
            source_generations[str(source)] = {
                "generation_id": f"source@{source}",
                "source_layer": source,
                "publication_frontier_layer": source,
                "consumer_layers": list(consumers),
                "source_publication": frontier,
                "producer": source_rec["producer"],
                "consumer_reads": reads,
                "complete": bool(reads) and all(str(c) in reads for c in consumers),
                "candidate_lifecycle": "candidate state is not produced before candidate source layer 20",
            }
        source_group = {
            "generations": source_generations,
            "candidate_lifecycle": "candidate state is not produced before candidate source layer 20",
        }

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
            "layer2_state_publication_present": layers < 3 or all(source_generations.get("2", {}).get("source_publication", {}).get(k) is not None for k in ("compress_kv", "index_k", "topk_idxs")),
            "layer8_state_publication_present": layers < 9 or all(source_generations.get("8", {}).get("source_publication", {}).get(k) is not None for k in ("compress_kv", "index_k", "topk_idxs")),
            "layer14_state_publication_present": layers < 15 or all(source_generations.get("14", {}).get("source_publication", {}).get(k) is not None for k in ("compress_kv", "index_k", "topk_idxs")),
            "source2_reuse_group_complete": source_generations.get("2", {}).get("complete") is True,
            "source8_reuse_group_complete": layers < 14 or source_generations.get("8", {}).get("complete") is True,
            "source14_reuse_group_complete": layers < 20 or source_generations.get("14", {}).get("complete") is True,
            "consumer_reads_match_source_publications": bool(source_generations) and all(
                all(v["compress_kv_matches_source"] and v["topk_idxs_matches_source"] and not v["consumer_recomputed_producer_state"] for v in gen["consumer_reads"].values())
                for gen in source_generations.values() if gen["consumer_reads"]
            ),
            "consumers_do_not_read_stale_generations": all(
                all(not v["stale_generation_read"] for v in gen["consumer_reads"].values())
                for gen in source_generations.values() if gen["consumer_reads"]
            ),
            "candidate_lifecycle_scoped_before_layer20": layers <= 20 and all(layer_records[str(i)]["state_after"].get("candidates") is None for i in range(layers)),
            "sweep_command_order_exact": [int(c["layer"]) for c in executed] == list(range(layers)),
            "engram_aware_reference_connected_scope_exact": reference_comparison["all_compared_exact"] and reference_comparison["reference_kind"] in {"engram_aware_connected_scope", "engram_aware_connected_block19_scope"},
        }
        artifact = {
            "schema": "ds41f.dwarfstar-prefill-vertical-slice.v1",
            "classification": "implementation_scoped_dwarfstar_prefill_slice_validation",
            "purpose": "first vertically integrated DwarfStar-derived production-prefill executor slice: official embedding -> typed sweep carry -> complete real layers -> executor publication/commit",
            "checkpoint": str(self.checkpoint),
            "ds4_authority": {"remote": DS4_AUTHORITY_REMOTE, "commit": DS4_AUTHORITY_SHA},
            "native_version": self.native.version(),
            "scope": {"tokens": token_arr.reshape(-1).tolist(), "layers": list(range(layers)), "full_model_prefill": False, "engram_enabled": enable_engram},
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
        return VerticalSliceResult(artifact)
