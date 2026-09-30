"""Command-driven official FP8/MLX block execution seam.

P3 scope: consume DwarfStar sweep commands and invoke reviewed oMLX/MLX
operation semantics for individual layer chunks.  This module deliberately does
not provide a whole-prefix `for layer in layers` execution loop, does not export
`PrefillContinuationState`, and does not implement final decode handoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ds41f_mlx.prefill_fp8_mlx.arena import RequestArena
from ds41f_mlx.prefill_fp8_mlx.guards import assert_no_reference_hot_path
from ds41f_mlx.prefill_fp8_mlx.planner import SweepCommand, SweepCommandKind
from ds41f_mlx.prefill_fp8_mlx.publications import PublicationManager


class BlockExecutionError(RuntimeError):
    """Invalid command-driven block execution transition."""


@dataclass(frozen=True)
class ServingLogitsPolicy:
    """Final-prefix-logits policy for prefill execution."""

    compute_final_prefix_logits: bool = False


@dataclass(frozen=True)
class MlxEvaluationPolicy:
    """Executor-owned MLX materialization policy over command batches.

    The default records no forced materialization.  Future executors may provide
    an `mx` module and choose explicit batch boundaries without tying sync to
    semantic publication/checkpoint boundaries.
    """

    evaluate_after_batch: bool = False
    synchronize_after_batch: bool = False

    def maybe_eval_batch(self, values: Iterable[Any], *, mx: Any | None = None) -> bool:
        if not self.evaluate_after_batch:
            return False
        if mx is None:
            raise BlockExecutionError("MLX evaluation requested but no mx module was supplied")
        tensors = [v for v in values if v is not None]
        if tensors:
            mx.eval(*tensors)
        if self.synchronize_after_batch:
            mx.synchronize()
        return True


@dataclass
class CommandExecutionRecord:
    command_index: int
    kind: str
    layer: int | None
    offset: int
    rows: int
    skipped_final_logits: bool = False
    invoked_block: bool = False
    invoked_engram: bool = False

    def to_json(self) -> dict[str, object]:
        return {
            "command_index": self.command_index,
            "kind": self.kind,
            "layer": self.layer,
            "offset": self.offset,
            "rows": self.rows,
            "skipped_final_logits": self.skipped_final_logits,
            "invoked_block": self.invoked_block,
            "invoked_engram": self.invoked_engram,
        }


@dataclass
class OfficialFP8MLXBlockRunner:
    """DwarfStar-command-owned execution adapter over oMLX layer operations."""

    language_model: Any
    publication_manager: PublicationManager
    logits_policy: ServingLogitsPolicy = field(default_factory=ServingLogitsPolicy)
    eval_policy: MlxEvaluationPolicy = field(default_factory=MlxEvaluationPolicy)
    image_mask: Any = None
    working_cache: list[Any] | None = None
    records: list[CommandExecutionRecord] = field(default_factory=list)
    final_logits_suppressed: bool = False
    final_logits: Any = None
    prefill_continuation_exported: bool = False
    full_cache_repack_count: int = 0

    def execute_batch(self, commands: Iterable[SweepCommand], arena: RequestArena) -> None:
        assert_no_reference_hot_path()
        batch_values: list[Any] = []
        try:
            for command in commands:
                result = self.execute_command(command, arena)
                if result is not None:
                    batch_values.append(result)
            self.eval_policy.maybe_eval_batch(batch_values, mx=getattr(self, "mx", None))
        finally:
            assert_no_reference_hot_path()

    def execute_command(self, command: SweepCommand, arena: RequestArena) -> Any | None:
        assert_no_reference_hot_path()
        record = CommandExecutionRecord(command.index, command.kind.value, command.layer, command.offset, command.rows)
        self.records.append(record)
        if command.kind is SweepCommandKind.BEGIN_INVALIDATE:
            arena.apply_command(command)
            self.publication_manager.begin_transaction()
            return None
        if command.kind is SweepCommandKind.ENCODE_ROWS:
            result = self._execute_encode_rows(command, arena, record)
            return result
        if command.kind is SweepCommandKind.PUBLISH_FRONTIER:
            arena.apply_command(command)
            self.publication_manager.publish_frontier(command)
            return None
        if command.kind is SweepCommandKind.SWAP_HC_AFTER_LAYER:
            arena.apply_command(command)
            return None
        if command.kind is SweepCommandKind.DECODER_PREPARE_SUFFIX:
            arena.apply_command(command)
            return None
        if command.kind is SweepCommandKind.ENCODE_OUTPUT_HEAD:
            return self._handle_output_head(command, arena, record)
        if command.kind is SweepCommandKind.READ_LOGITS:
            return self._handle_read_logits(command, arena, record)
        if command.kind is SweepCommandKind.CHECKPOINT_MAY_COMMIT:
            self._assert_commit_ready()
            arena.apply_command(command)
            self.publication_manager.commit()
            return None
        arena.apply_command(command)
        return None

    def _execute_encode_rows(self, command: SweepCommand, arena: RequestArena, record: CommandExecutionRecord) -> Any:
        if command.layer is None:
            raise BlockExecutionError("encode_rows command requires a layer")
        arena.apply_command(command)
        lm = self.language_model
        layer = lm.layers[int(command.layer)]
        h_chunk = _slice_rows(arena.carry.current.value, command.offset, command.rows)
        pre_chunk = _slice_rows(arena.carry.pre.value, command.offset, command.rows)
        shared = self.publication_manager.shared_for_layer(int(command.layer))
        cache = self._cache_for_layer(int(command.layer))
        start = int(command.offset)
        invoked_engram = False
        if _layer_has_engram(layer):
            hashes = _slice_engram_hashes(arena.engram.hashes.value, command.offset, command.rows, int(command.layer), lm)
            if hashes is not None:
                h_chunk = layer.engram(h_chunk, hashes, self.image_mask)
                invoked_engram = True
        if not callable(layer):
            raise BlockExecutionError(f"layer {command.layer} is not callable")
        h_out, pre_out = layer(h_chunk, pre_chunk, cache, shared, start, self.image_mask)
        arena.carry.next.value = _write_rows(arena.carry.next.value, command.offset, command.rows, h_out)
        arena.carry.pre.value = _write_rows(arena.carry.pre.value, command.offset, command.rows, pre_out)
        self.publication_manager.capture_layer_outputs(int(command.layer), shared, command_index=command.index)
        record.invoked_block = True
        record.invoked_engram = invoked_engram
        return h_out

    def _cache_for_layer(self, layer: int) -> Any:
        if self.working_cache is None:
            make_cache = getattr(self.language_model, "make_cache", None)
            if make_cache is None:
                self.working_cache = [None for _ in getattr(self.language_model, "layers")]
            else:
                self.working_cache = make_cache()
        return self.working_cache[layer]

    def _handle_output_head(self, command: SweepCommand, arena: RequestArena, record: CommandExecutionRecord) -> Any | None:
        arena.apply_command(command)
        if not self.logits_policy.compute_final_prefix_logits:
            self.final_logits_suppressed = True
            record.skipped_final_logits = True
            return None
        raise BlockExecutionError("diagnostic final-prefix logits path is not implemented in P3")

    def _handle_read_logits(self, command: SweepCommand, arena: RequestArena, record: CommandExecutionRecord) -> Any | None:
        arena.apply_command(command)
        if not self.logits_policy.compute_final_prefix_logits:
            self.final_logits_suppressed = True
            record.skipped_final_logits = True
            return None
        return self.final_logits

    def _assert_commit_ready(self) -> None:
        if self.logits_policy.compute_final_prefix_logits and self.final_logits is None:
            raise BlockExecutionError("cannot commit diagnostic logits path before final logits are ready")
        # Serving path deliberately suppresses logits; commit readiness is state/publication based.

    def to_json(self) -> dict[str, object]:
        return {
            "records": [r.to_json() for r in self.records],
            "final_logits_suppressed": self.final_logits_suppressed,
            "prefill_continuation_exported": self.prefill_continuation_exported,
            "full_cache_repack_count": self.full_cache_repack_count,
        }


def _layer_has_engram(layer: Any) -> bool:
    try:
        if "engram" in layer:
            return True
    except Exception:
        pass
    return hasattr(layer, "engram")


def _slice_rows(value: Any, offset: int, rows: int) -> Any:
    if value is None:
        return None
    try:
        return value[:, offset:offset + rows]
    except Exception:
        return value


def _write_rows(base: Any, offset: int, rows: int, update: Any) -> Any:
    if base is None:
        return update
    try:
        base[:, offset:offset + rows] = update
        return base
    except Exception:
        return update if offset == 0 else base


def _slice_engram_hashes(hashes: Any, offset: int, rows: int, layer: int, language_model: Any) -> Any:
    if hashes is None:
        return None
    try:
        layer_ids = list(language_model._config.engram_layer_ids)
        ix = layer_ids.index(layer)
        return hashes[:, offset:offset + rows, ix]
    except Exception:
        try:
            return hashes[:, offset:offset + rows]
        except Exception:
            return hashes
