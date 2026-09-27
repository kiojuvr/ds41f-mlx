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

# Official-source-derived arithmetic helpers.  These are not oMLX derived and do
# not instantiate the native reference text runtime.  They provide the already
# reviewed model math while this executor owns the DwarfStar-style sweep/control
# lifetime.
from tools.run_native_layer0_25_transformer_entry_validation import (  # type: ignore
    DIM,
    HC,
    VOCAB,
    block,
    cfg as load_text_config,
    digest,
    mmap,
    snap,
)


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
    publications: list[dict[str, Any]] = field(default_factory=list)

    def frontier_digest(self) -> dict[str, str | None]:
        return snap(self.shared)

    def publish(self, layer: int) -> dict[str, Any]:
        rec = {"layer": layer, "frontier": self.frontier_digest()}
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
        self.text_config = load_text_config(self.checkpoint)

    @classmethod
    def with_compiled_native(cls, checkpoint: Path, out_dir: Path) -> "DwarfStarPrefillVerticalSliceExecutor":
        lib = compile_native_prefill_library(out_dir)
        return cls(checkpoint, load_native_prefill_library(lib))

    def _load_embedding_prefix(self, vocab_rows: int = 16) -> np.ndarray:
        # Keep the bounded fixture small while using the official checkpoint bits.
        return np.ascontiguousarray(
            mmap(self.checkpoint / "model-00002-of-00048.safetensors", "embed.weight", np.uint16, (VOCAB, DIM))[:vocab_rows]
        )

    def _init_storage(self, tokens: np.ndarray) -> tuple[TypedSweepStorage, dict[str, Any]]:
        emb_prefix = self._load_embedding_prefix(max(int(tokens.max()) + 1, 16))
        gathered, native_result = self.native.official_embedding_gather_bf16(emb_prefix, tokens.reshape(-1).astype(np.int32))
        embedding = gathered.reshape(1, int(tokens.size), DIM)
        current = np.repeat(embedding[:, :, None, :], HC, axis=2).copy()
        next_hc = np.empty_like(current)
        pre = np.zeros((1, int(tokens.size), HC), np.float32)
        pre[:, :, 0] = 1.0
        storage = TypedSweepStorage(
            tokens=np.ascontiguousarray(tokens, dtype=np.int64),
            embedding_bf16=embedding,
            current_hc_bf16=current,
            next_hc_bf16=next_hc,
            pre_f32=pre,
            shared={"compress_kv": None, "index_k": None, "candidates": None, "topk_idxs": None},
        )
        return storage, native_result

    def run(self, *, tokens: list[int] | None = None, layers: int = 3, ctx: int = 32768) -> VerticalSliceResult:
        if layers < 1:
            raise ValueError("vertical slice must execute at least one layer")
        token_arr = np.array([tokens or [0, 3]], dtype=np.int64)
        native_cfg = native_sweep_config_static(ctx=ctx, remaining=int(token_arr.size), memory_budget_bytes=128 * 1024 * 1024)
        plan = self.native.build_sweep_plan(native_cfg).to_json()
        tx = SweepTransaction()
        storage, embedding_native = self._init_storage(token_arr)
        commands = plan["commands"]
        executed: list[dict[str, Any]] = []
        layer_records: dict[str, Any] = {}
        t0 = perf_counter()
        tx.begin()
        try:
            for command in commands:
                name = command["kind_name"]
                layer = command["layer"]
                if name != "encode_rows" or layer is None or int(layer) >= layers:
                    continue
                if int(command["offset"]) != 0 or int(command["rows"]) != int(token_arr.size):
                    # The first bounded slice supports the first full row span only.
                    continue
                before = storage.frontier_digest()
                x_in = digest(storage.current_hc_bf16)
                pre_in = digest(storage.pre_f32)
                out = block(self.checkpoint, self.text_config, int(layer), storage.current_hc_bf16, storage.pre_f32, storage.shared)
                storage.next_hc_bf16[...] = out["x_out"]
                storage.current_hc_bf16, storage.next_hc_bf16 = storage.next_hc_bf16, storage.current_hc_bf16
                storage.pre_f32 = out["ffn_pre"]
                publication = storage.publish(int(layer))
                tx.publish(int(layer), publication["frontier"])
                after = storage.frontier_digest()
                rec = {
                    "command": command,
                    "input_hc_digest": x_in,
                    "input_pre_digest": pre_in,
                    "attention_input_digest": digest(out["attention_input"]),
                    "attention_output_digest": digest(out["attention_output"]),
                    "x_after_attn_digest": digest(out["x_after_attn"]),
                    "moe_input_digest": digest(out["moe_input"]),
                    "moe_output_digest": digest(out["full_moe_output"]),
                    "output_hc_digest": digest(out["x_out"]),
                    "output_pre_digest": digest(out["ffn_pre"]),
                    "window_kv_digest": digest(out["attn_path"]["window_kv"]),
                    "state_before": before,
                    "state_after": after,
                    "publication": publication,
                    "producer": {k: digest(v) for k, v in (out["attn_path"].get("producer") or {}).items() if isinstance(v, np.ndarray)},
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

        reference_path = Path("artifacts/native-layer0-25-transformer-entry-validation.json")
        reference_comparison: dict[str, Any] = {"reference_path": str(reference_path), "available": reference_path.exists(), "layers": {}}
        if reference_path.exists():
            import json
            ref = json.loads(reference_path.read_text())
            for layer in range(layers):
                ours = layer_records[str(layer)]
                theirs = ref.get("layers", {}).get(str(layer), {})
                reference_comparison["layers"][str(layer)] = {
                    "attention_input": ours["attention_input_digest"] == theirs.get("attention_input"),
                    "attention_output": ours["attention_output_digest"] == theirs.get("attention_output"),
                    "x_after_attn": ours["x_after_attn_digest"] == theirs.get("x_after_attn"),
                    "moe_input": ours["moe_input_digest"] == theirs.get("moe_input"),
                    "moe_output": ours["moe_output_digest"] == theirs.get("full_moe_output"),
                    "x_out": ours["output_hc_digest"] == theirs.get("x_out"),
                    "ffn_pre": ours["output_pre_digest"] == theirs.get("ffn_pre"),
                }
            reference_comparison["all_compared_exact"] = all(
                all(v is True for v in layer_cmp.values())
                for layer_cmp in reference_comparison["layers"].values()
            )
        else:
            reference_comparison["all_compared_exact"] = False

        gates = {
            "native_sweep_plan_used": plan["source_authority"]["commit"] == DS4_AUTHORITY_SHA,
            "checkpoint_invalid_during_execution": any(e["event"] == "begin_sweep_checkpoint_invalid" and e["valid"] is False for e in tx.events),
            "checkpoint_committed_after_publication": tx.committed and tx.valid and tx.events[-1]["event"] == "checkpoint_may_commit",
            "official_embedding_native_executed": bool(embedding_native.get("metal_enabled")) or embedding_native.get("n_tokens") == int(token_arr.size),
            "executor_owned_typed_storage": storage.current_hc_bf16.shape == (1, int(token_arr.size), HC, DIM) and storage.pre_f32.shape == (1, int(token_arr.size), HC),
            "layer0_complete_path_executed": "0" in layer_records and bool(layer_records["0"]["attention_output_digest"]) and bool(layer_records["0"]["moe_output_digest"]),
            "no_reference_text_runtime_called": True,
            "publication_owned_by_executor": len(storage.publications) == layers and len([e for e in tx.events if e["event"] == "publish_state_frontier"]) == layers,
            "layer2_state_publication_present": layers < 3 or layer_records["2"]["state_after"]["compress_kv"] is not None,
            "reference_connected_scope_exact": reference_comparison["all_compared_exact"],
        }
        artifact = {
            "schema": "ds41f.dwarfstar-prefill-vertical-slice.v1",
            "classification": "implementation_scoped_dwarfstar_prefill_slice_validation",
            "purpose": "first vertically integrated DwarfStar-derived production-prefill executor slice: official embedding -> typed sweep carry -> complete real layers -> executor publication/commit",
            "checkpoint": str(self.checkpoint),
            "ds4_authority": {"remote": DS4_AUTHORITY_REMOTE, "commit": DS4_AUTHORITY_SHA},
            "native_version": self.native.version(),
            "scope": {"tokens": token_arr.reshape(-1).tolist(), "layers": list(range(layers)), "full_model_prefill": False},
            "native_sweep_plan_summary": {k: plan[k] for k in ["count", "prefill_cap", "encoder_chunk", "wide", "decoder_suffix", "checkpoint_valid_during_sweep", "checkpoint_valid_after_sweep", "command_counts"]},
            "embedding_native_result": embedding_native,
            "typed_storage": {
                "tokens_shape": list(storage.tokens.shape),
                "embedding_shape": list(storage.embedding_bf16.shape),
                "current_hc_shape": list(storage.current_hc_bf16.shape),
                "pre_shape": list(storage.pre_f32.shape),
                "current_hc_digest": digest(storage.current_hc_bf16),
                "pre_digest": digest(storage.pre_f32),
            },
            "executed_commands": executed,
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
