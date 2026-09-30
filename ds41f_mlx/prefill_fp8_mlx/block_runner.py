"""Command-driven official FP8/MLX block execution seam.

P3 scope: consume DwarfStar sweep commands and invoke reviewed oMLX/MLX
operation semantics for individual layer chunks.  This module deliberately does
not provide a whole-prefix `for layer in layers` execution loop, does not export
`PrefillContinuationState`, and does not implement final decode handoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
import importlib

from ds41f_mlx.prefill_fp8_mlx.arena import RequestArena
from ds41f_mlx.prefill_fp8_mlx.guards import assert_no_reference_hot_path
from ds41f_mlx.prefill_fp8_mlx.planner import SweepCommand, SweepCommandKind, SweepPhase
from ds41f_mlx.prefill_fp8_mlx.publications import PublicationManager


class BlockExecutionError(RuntimeError):
    """Invalid command-driven block execution transition."""


@dataclass(frozen=True)
class ServingLogitsPolicy:
    """Final-prefix-logits policy for prefill execution."""

    compute_final_prefix_logits: bool = False


@dataclass(frozen=True)
class MlxEvaluationPolicy:
    """Executor-owned MLX materialization policy over command batches."""

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
    absolute_start: int | None = None
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
            "absolute_start": self.absolute_start,
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
            return self._execute_encode_rows(command, arena, record)
        if command.kind is SweepCommandKind.PUBLISH_FRONTIER:
            arena.apply_command(command)
            self.publication_manager.publish_frontier(command)
            return None
        if command.kind in {SweepCommandKind.SWAP_HC_AFTER_LAYER, SweepCommandKind.DECODER_PREPARE_SUFFIX}:
            arena.apply_command(command)
            return None
        if command.kind is SweepCommandKind.ENCODE_OUTPUT_HEAD:
            return self._handle_output_head(command, arena, record)
        if command.kind is SweepCommandKind.READ_LOGITS:
            return self._handle_read_logits(command, arena, record)
        if command.kind is SweepCommandKind.CHECKPOINT_MAY_COMMIT:
            self._assert_commit_ready()
            self._assert_cache_slots_ready(arena)
            arena.apply_command(command)
            self.publication_manager.commit()
            return None
        arena.apply_command(command)
        return None

    def _execute_encode_rows(self, command: SweepCommand, arena: RequestArena, record: CommandExecutionRecord) -> Any:
        if command.layer is None:
            raise BlockExecutionError("encode_rows command requires a layer")
        arena.apply_command(command)
        arena.require_decoder_prepared(command)
        lm = self.language_model
        layer_id = int(command.layer)
        layer = lm.layers[layer_id]
        absolute_start = int(arena.base_frontier) + int(command.offset)
        record.absolute_start = absolute_start
        h_chunk = _slice_rows(arena.carry.current.value, command.offset, command.rows, role=arena.carry.current.role)
        pre_chunk = _slice_rows(arena.carry.pre.value, command.offset, command.rows, role=arena.carry.pre.role)
        shared = self.publication_manager.shared_for_span(layer_id, command.offset, command.rows, require_keys=self._required_publication_keys(layer_id))
        if command.phase is SweepPhase.DECODER_SUFFIX and layer_id == 20 and arena.encoder_final_source_state is not None:
            shared["__full_encoder_final_source_state__"] = arena.encoder_final_source_state
        cache = self._cache_for_layer(layer_id)
        invoked_engram = False
        if _layer_has_engram(layer):
            hashes = _slice_engram_hashes(arena.engram.hashes.value, command.offset, command.rows, layer_id, lm)
            if hashes is not None:
                h_chunk = layer.engram(h_chunk, hashes, self.image_mask)
                invoked_engram = True
        if not callable(layer):
            raise BlockExecutionError(f"layer {command.layer} is not callable")
        h_out, pre_out = layer(h_chunk, pre_chunk, cache, shared, absolute_start, self.image_mask)
        arena.carry.next.value = _write_rows(arena.carry.next.value, command.offset, command.rows, h_out, role=arena.carry.next.role)
        arena.carry.pre.value = _write_rows(arena.carry.pre.value, command.offset, command.rows, pre_out, role=arena.carry.pre.role)
        self._advance_cache_layer(cache, absolute_start + command.rows, arena=arena, layer=layer_id)
        self.publication_manager.capture_layer_outputs(layer_id, shared, command_index=command.index, offset=command.offset, rows=command.rows)
        record.invoked_block = True
        record.invoked_engram = invoked_engram
        return h_out

    def _required_publication_keys(self, layer: int) -> tuple[str, ...]:
        return self.publication_manager.topology.consumes_by_layer.get(int(layer), ())

    def _cache_for_layer(self, layer: int) -> Any:
        if self.working_cache is None:
            make_cache = getattr(self.language_model, "make_cache", None)
            if make_cache is None:
                self.working_cache = [None for _ in getattr(self.language_model, "layers")]
            else:
                self.working_cache = make_cache()
        return self.working_cache[layer]

    def _advance_cache_layer(self, cache: Any, absolute_end: int, *, arena: RequestArena, layer: int) -> None:
        if cache is None:
            return
        _set_cache_slot(cache, 0, _make_cache_offset(absolute_end, self.language_model))
        if layer == 0 and arena.engram.history.value is not None:
            _set_cache_slot(cache, 6, arena.engram.history.value)
        self._fill_empty_slots(cache, layer)

    def _fill_empty_slots(self, cache: Any, layer: int) -> None:
        if cache is None:
            return
        for slot in range(1, 7):
            if _get_cache_slot(cache, slot) is None:
                _set_cache_slot(cache, slot, _empty_cache_slot(self.language_model, slot))

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

    def _assert_cache_slots_ready(self, arena: RequestArena) -> None:
        if self.working_cache is None:
            raise BlockExecutionError("cannot commit before working cache is initialized")
        for i, cache in enumerate(self.working_cache):
            if cache is None:
                raise BlockExecutionError(f"working cache layer {i} is missing")
            if _get_cache_slot(cache, 0) is None:
                raise BlockExecutionError(f"working cache layer {i} frontier slot is missing")
            for slot in range(1, 7):
                if _get_cache_slot(cache, slot) is None:
                    raise BlockExecutionError(f"working cache layer {i} slot {slot} is missing")

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


def _slice_rows(value: Any, offset: int, rows: int, *, role: str) -> Any:
    if value is None:
        raise BlockExecutionError(f"cannot slice unbound arena slot {role}")
    end = int(offset) + int(rows)
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 1 and int(shape[1]) < end:
        raise BlockExecutionError(f"arena slot {role} has only {shape[1]} rows; requested {offset}:{end}")
    try:
        return value[:, offset:end]
    except Exception as exc:
        raise BlockExecutionError(f"failed to slice arena slot {role} rows {offset}:{end}") from exc


def _write_rows(base: Any, offset: int, rows: int, update: Any, *, role: str) -> Any:
    if base is None:
        raise BlockExecutionError(f"cannot write rows into unallocated arena slot {role}")
    end = int(offset) + int(rows)
    shape = getattr(base, "shape", None)
    if shape is not None and len(shape) > 1 and int(shape[1]) < end:
        raise BlockExecutionError(f"arena slot {role} has only {shape[1]} rows; cannot write {offset}:{end}")
    update_shape = getattr(update, "shape", None)
    if update_shape is not None and len(update_shape) > 1 and int(update_shape[1]) != int(rows):
        raise BlockExecutionError(f"update for {role} has {update_shape[1]} rows; expected {rows}")
    assign = getattr(base, "assign_rows", None)
    if assign is not None:
        assign(offset, rows, update)
        return base
    try:
        base[:, offset:end] = update
        return base
    except Exception as exc:
        raise BlockExecutionError(f"failed to write arena slot {role} rows {offset}:{end}") from exc


def _slice_engram_hashes(hashes: Any, offset: int, rows: int, layer: int, language_model: Any) -> Any:
    if hashes is None:
        return None
    end = offset + rows
    try:
        layer_ids = list(language_model._config.engram_layer_ids)
        ix = layer_ids.index(layer)
        return hashes[:, offset:end, ix]
    except ValueError:
        return None
    except Exception as exc:
        raise BlockExecutionError(f"failed to slice Engram hashes for layer {layer} rows {offset}:{end}") from exc


def _get_cache_slot(cache: Any, slot: int) -> Any:
    try:
        return cache[slot]
    except Exception:
        return getattr(cache, "cache", [None] * 7)[slot]


def _set_cache_slot(cache: Any, slot: int, value: Any) -> None:
    try:
        cache[slot] = value
        return
    except Exception:
        pass
    if not hasattr(cache, "cache"):
        raise BlockExecutionError("cache object does not support slot assignment")
    cache.cache[slot] = value


def _make_cache_offset(value: int, language_model: Any) -> Any:
    factory = getattr(language_model, "cache_offset", None)
    if factory is not None:
        return factory(int(value))
    try:
        mx = importlib.import_module("mlx.core")
    except Exception as exc:
        raise BlockExecutionError("production cache offset construction requires mlx.core") from exc
    try:
        return mx.array([int(value)], mx.int32)
    except Exception as exc:
        raise BlockExecutionError("failed to construct MLX cache offset") from exc


def _empty_cache_slot(language_model: Any, slot: int) -> Any:
    factory = getattr(language_model, "empty_cache_slot", None)
    if factory is not None:
        return factory(slot)
    try:
        mx = importlib.import_module("mlx.core")
        c = language_model._config
        width = c.index_head_dim if slot == 3 else c.head_dim
        empty = mx.zeros((1, 0, width), mx.bfloat16)
        if slot == 6:
            return mx.zeros((1, 0), mx.int64)
        if slot in {1, 2, 3}:
            lang = importlib.import_module(type(language_model).__module__)
            pack_activation = lang.pack_activation
            if slot == 1:
                return pack_activation(empty)
            if slot == 2:
                return pack_activation(empty, 4, 16, True)
            return pack_activation(empty, 4)
        return empty
    except Exception as exc:
        raise BlockExecutionError(f"failed to construct packed empty cache slot {slot}") from exc
