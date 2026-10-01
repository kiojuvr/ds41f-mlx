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
from ds41f_mlx.prefill_fp8_mlx.omlx_suffix_math import OmlxV41SuffixMath
from ds41f_mlx.prefill_fp8_mlx.planner import SweepCommand, SweepCommandKind, SweepPhase
from ds41f_mlx.prefill_fp8_mlx.publications import PublicationManager
from ds41f_mlx.prefill_fp8_mlx.p7_scheduling import SchedulingCoordinator


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
    invoked_decoder_prepare: bool = False
    invoked_full_source_publish: bool = False

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
            "invoked_decoder_prepare": self.invoked_decoder_prepare,
            "invoked_full_source_publish": self.invoked_full_source_publish,
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
    suffix_math: OmlxV41SuffixMath | None = None
    records: list[CommandExecutionRecord] = field(default_factory=list)
    final_logits_suppressed: bool = False
    final_logits: Any = None
    prefill_continuation_exported: bool = False
    full_cache_repack_count: int = 0
    handoff_reserved: bool = False
    handoff_transferred: bool = False
    closed: bool = False
    p6_owner_token: int | None = None
    mx: Any | None = None
    execution_revoked: bool = False
    scheduling_coordinator: SchedulingCoordinator | None = None

    def __post_init__(self) -> None:
        if self.suffix_math is None:
            self.suffix_math = OmlxV41SuffixMath(self.language_model)
        if self.scheduling_coordinator is None and getattr(self.language_model, "_p7_enable_overlap", False):
            self.scheduling_coordinator = SchedulingCoordinator(self.language_model, mx=self.mx)

    def execute_batch(self, commands: Iterable[SweepCommand], arena: RequestArena) -> None:
        assert_no_reference_hot_path()
        command_list = tuple(commands)
        if self.scheduling_coordinator is not None:
            self.scheduling_coordinator.set_command_stream(command_list)
        batch_values: list[Any] = []
        try:
            for command in command_list:
                result = self.execute_command(command, arena)
                if result is not None:
                    batch_values.append(result)
            self.eval_policy.maybe_eval_batch(batch_values, mx=self.mx)
        finally:
            assert_no_reference_hot_path()

    def execute_command(self, command: SweepCommand, arena: RequestArena) -> Any | None:
        if self.closed or self.execution_revoked:
            raise BlockExecutionError('prefill runner capability is closed/revoked')
        self._assert_p6_cache_capability()
        if self.handoff_reserved or self.handoff_transferred:
            raise BlockExecutionError('prefill cache authority reserved/transferred to generation')
        assert_no_reference_hot_path()
        record = CommandExecutionRecord(command.index, command.kind.value, command.layer, command.offset, command.rows)
        self.records.append(record)
        if self.scheduling_coordinator is not None:
            self.scheduling_coordinator.handle_command(command, arena, _slice_engram_hashes)
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
        if command.kind is SweepCommandKind.SWAP_HC_AFTER_LAYER:
            arena.apply_command(command)
            return None
        if command.kind is SweepCommandKind.DECODER_PREPARE_SUFFIX:
            arena.apply_command(command)
            self._prepare_decoder_local_window(command, arena, record)
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
            if self.scheduling_coordinator is not None:
                self.scheduling_coordinator.seal_success()
            return None
        if command.kind is SweepCommandKind.P6_SOURCE_COMPLETE_AND_DETACH_CONE:
            arena.apply_command(command)
            self._p6_materialize_and_detach(command, arena, record)
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
        private_start = getattr(arena, "p6_private_start", None)
        absolute_start = int(arena.base_frontier if private_start is None else private_start) + int(command.offset)
        record.absolute_start = absolute_start
        h_chunk = _slice_rows(arena.carry.current.value, command.offset, command.rows, role=arena.carry.current.role, row_origin=arena.carry.current.row_origin)
        pre_chunk = _slice_rows(arena.carry.pre.value, command.offset, command.rows, role=arena.carry.pre.role, row_origin=arena.carry.pre.row_origin)
        shared_fn = self.publication_manager.producer_shared_for_span if command.phase is SweepPhase.DECODER_SUFFIX and layer_id == 20 else self.publication_manager.shared_for_span
        shared = shared_fn(layer_id, command.offset, command.rows, require_keys=self._required_publication_keys(layer_id))
        cache = self._cache_for_layer(layer_id)
        invoked_engram = False
        if _layer_has_engram(layer):
            hashes = _slice_engram_hashes(arena.engram.hashes.value, command.offset, command.rows, layer_id, lm)
            if hashes is not None:
                if self.scheduling_coordinator is not None:
                    h_chunk = self.scheduling_coordinator.apply_engram_micro_pipeline(command, arena, h_chunk, pre_chunk, layer.engram, self.image_mask, _slice_engram_hashes)
                else:
                    h_chunk = layer.engram(h_chunk, hashes, self.image_mask)
                invoked_engram = True
        if not callable(layer):
            raise BlockExecutionError(f"layer {command.layer} is not callable")
        if command.phase is SweepPhase.DECODER_SUFFIX:
            h_out, pre_out = self.suffix_math.execute_suffix_query(layer_id=layer_id, h_chunk=h_chunk, pre_chunk=pre_chunk, cache=cache, shared=shared, absolute_start=absolute_start, image_mask=self.image_mask)
        else:
            h_out, pre_out = layer(h_chunk, pre_chunk, cache, shared, absolute_start, self.image_mask)
        arena.carry.next.value = _write_rows(arena.carry.next.value, command.offset, command.rows, h_out, role=arena.carry.next.role, row_origin=arena.carry.next.row_origin)
        arena.carry.pre.value = _write_rows(arena.carry.pre.value, command.offset, command.rows, pre_out, role=arena.carry.pre.role, row_origin=arena.carry.pre.row_origin)
        self._advance_cache_layer(cache, absolute_start + command.rows, arena=arena, layer=layer_id)
        capture_keys = ("idx", "candidates") if command.phase is SweepPhase.DECODER_SUFFIX and layer_id == 20 else None
        self.publication_manager.capture_layer_outputs(layer_id, shared, command_index=command.index, offset=command.offset, rows=command.rows, keys=capture_keys)
        record.invoked_block = True
        record.invoked_engram = invoked_engram
        return h_out

    def _required_publication_keys(self, layer: int) -> tuple[str, ...]:
        return self.publication_manager.topology.consumes_by_layer.get(int(layer), ())

    def _prepare_decoder_local_window(self, command: SweepCommand, arena: RequestArena, record: CommandExecutionRecord) -> None:
        if command.layer is None:
            raise BlockExecutionError("decoder_prepare_suffix requires a layer")
        role = arena.decoder_prepare_role(command)
        cache = self._cache_for_layer(int(command.layer))
        private_start = getattr(arena, "p6_private_start", None)
        absolute_start = int(arena.base_frontier if private_start is None else private_start) + int(command.offset)
        record.absolute_start = absolute_start
        if role == "decoder_full_source_publish":
            if arena.encoder_final_h is None or arena.encoder_final_pre is None:
                raise BlockExecutionError("layer20 full-source publication requires immutable encoder-final h/pre")
            shared = self.publication_manager.shared_for_span(int(command.layer), 0, command.rows)
            private_start = getattr(arena, "p6_private_start", None)
            source_start = int(arena.base_frontier if private_start is None else private_start)
            self.suffix_math.publish_full_source(layer_id=int(command.layer), h_full=arena.encoder_final_h, pre_full=arena.encoder_final_pre, cache=cache, shared=shared, absolute_start=source_start, rows=arena.plan.count)
            self.publication_manager.capture_layer_outputs(int(command.layer), shared, command_index=command.index, offset=0, rows=arena.plan.count, keys=("kv", "index_k"))
            record.invoked_full_source_publish = True
            return
        rows = _slice_rows(arena.carry.current.value, command.offset, command.rows, role=arena.carry.current.role, row_origin=arena.carry.current.row_origin)
        pre = _slice_rows(arena.carry.pre.value, command.offset, command.rows, role=arena.carry.pre.role, row_origin=arena.carry.pre.row_origin)
        self.suffix_math.prepare_local_window(layer_id=int(command.layer), h_rows=rows, pre_rows=pre, cache=cache, absolute_start=absolute_start, rows=command.rows)
        record.invoked_decoder_prepare = True

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
        if getattr(arena, "p6_freeze_public_offsets", False):
            arena.p6_private_layer_frontiers[int(layer)] = int(absolute_end)
        else:
            _set_cache_slot(cache, 0, _make_cache_offset(absolute_end, self.language_model))
            if layer == 0 and arena.engram.history.value is not None:
                _set_cache_slot(cache, 6, arena.engram.history.value)
        self._fill_empty_slots(cache, layer)

    def close(self) -> None:
        if self.scheduling_coordinator is not None:
            self.scheduling_coordinator.revoke()
        self.closed = True
        self.working_cache = None

    def _assert_p6_cache_capability(self) -> None:
        cache = self.working_cache
        if not cache:
            return
        invalid = any(getattr(c, "_p6_append_invalid", False) or getattr(c, "_p6_append_pending", False) for c in cache)
        sealed = any(getattr(c, "_p6_append_sealed", False) for c in cache)
        failed = any(getattr(c, "_p6_append_failed", False) for c in cache)
        if sealed:
            raise BlockExecutionError("cache is sealed; prefill execution is revoked")
        if failed:
            raise BlockExecutionError("cache belongs to a failed P6 append")
        if invalid:
            tokens = {getattr(c, "_p6_owner_token", None) for c in cache}
            if len(tokens) != 1 or self.p6_owner_token not in tokens:
                raise BlockExecutionError("runner lacks current P6 append owner capability")

    def _p6_materialize_and_detach(self, command: SweepCommand, arena: RequestArena, record: CommandExecutionRecord) -> None:
        frontier = int(getattr(arena, "p6_private_start", arena.base_frontier) or arena.base_frontier) + int(arena.plan.count)
        evaluated = self._p6_eval_persistent_source_state(arena)
        if command.rows > 0:
            origin = int(command.offset)
            private_start = int(getattr(arena, "p6_private_start", arena.base_frontier) or arena.base_frontier)
            q20_origin = int(arena.plan.count) - 2414
            h = self._owned_row_copy(arena.carry.current.value, origin, int(command.rows), role=arena.carry.current.role)
            nxt = self._owned_row_copy(arena.carry.next.value, q20_origin, 2414, role=arena.carry.next.role)
            pre = self._owned_row_copy(arena.carry.pre.value, origin, int(command.rows), role=arena.carry.pre.role)
            arena.detach_final_decoder_cone(origin=private_start + origin, rows=int(command.rows), h_value=h, next_value=nxt, pre_value=pre, row_origin=origin, next_row_origin=q20_origin)
            arena.input_ids.value = None
            arena.engram.hashes.value = None
            arena.encoder_final_h = None
            arena.encoder_final_pre = None
            arena.active_chunk_views.clear()
        arena.materialize_p6_source_boundary(command_index=command.index, frontier=frontier, evaluated_slots=evaluated)

    def _p6_eval_persistent_source_state(self, arena: RequestArena) -> list[dict[str, int]]:
        if self.working_cache is None:
            raise BlockExecutionError("P6 source materialization requires a live cache")
        mx = self.mx
        if mx is None:
            try:
                mx = importlib.import_module("mlx.core")
            except Exception:
                mx = None
        values: list[Any] = []
        slots: list[dict[str, int]] = []
        cfg = getattr(self.language_model, "_config", None)
        source_layers = set(int(x) for x in getattr(cfg, "kv_source_layers", (2, 8, 14, 20))) | set(int(x) for x in getattr(cfg, "index_source_layers", (2, 8, 14, 20)))
        for layer in range(20):
            v = _get_cache_slot(self.working_cache[layer], 1)
            if _p6_materializable_value(v):
                values.append(v); slots.append({"layer": layer, "slot": 1})
        for layer in sorted(source_layers):
            if 0 <= layer < len(self.working_cache):
                for slot in (2, 3, 4, 5):
                    v = _get_cache_slot(self.working_cache[layer], slot)
                    if _p6_materializable_value(v):
                        values.append(v); slots.append({"layer": layer, "slot": slot})
        if values:
            if mx is None or not hasattr(mx, "eval"):
                if not all(getattr(v, "allow_fake_eval", False) for v in values):
                    raise BlockExecutionError("P6 source materialization requires mx.eval for persistent arrays")
                fake_eval = getattr(values[0], "fake_eval", None)
                if fake_eval is not None:
                    fake_eval(values)
            else:
                mx.eval(*values)
        return slots

    def _owned_row_copy(self, value: Any, offset: int, rows: int, *, role: str) -> Any:
        return _owned_row_copy(value, offset, rows, role=role, mx=self.mx)

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
            frontier = _get_cache_slot(cache, 0)
            if frontier is None:
                raise BlockExecutionError(f"working cache layer {i} frontier slot is missing")
            expected = int(arena.base_frontier) + int(arena.plan.count)
            actual = int(frontier[0]) if hasattr(frontier, "__getitem__") else int(frontier)
            if actual != expected:
                raise BlockExecutionError(f"layer {i} logical frontier {actual} != {expected}")
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


def _p6_materializable_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple)) and not value:
        return False
    return True


def _layer_has_engram(layer: Any) -> bool:
    try:
        if "engram" in layer:
            return True
    except Exception:
        pass
    return hasattr(layer, "engram")


def _slice_rows(value: Any, offset: int, rows: int, *, role: str, row_origin: int = 0) -> Any:
    if value is None:
        raise BlockExecutionError(f"cannot slice unbound arena slot {role}")
    offset = int(offset) - int(row_origin)
    if offset < 0:
        raise BlockExecutionError(f"arena slot {role} row origin {row_origin} cannot serve requested offset")
    end = int(offset) + int(rows)
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 1 and int(shape[1]) < end:
        raise BlockExecutionError(f"arena slot {role} has only {shape[1]} rows; requested {offset}:{end}")
    try:
        return value[:, offset:end]
    except Exception as exc:
        raise BlockExecutionError(f"failed to slice arena slot {role} rows {offset}:{end}") from exc


def _owned_row_copy(value: Any, offset: int, rows: int, *, role: str, mx: Any | None = None) -> Any:
    sliced = _slice_rows(value, offset, rows, role=role)
    detach = getattr(sliced, "detach_rows", None)
    if detach is not None:
        return detach()
    copy = getattr(sliced, "copy", None)
    if copy is not None:
        try:
            return copy()
        except Exception:
            pass
    if mx is not None:
        try:
            copier = getattr(mx, "copy", None)
            array_ctor = getattr(mx, "array", None)
            if copier is not None:
                out = copier(sliced)
            elif array_ctor is not None:
                out = array_ctor(sliced)
            else:
                raise BlockExecutionError("MLX module does not expose a compact copy primitive")
            mx.eval(out)
            return out
        except BlockExecutionError:
            raise
        except Exception as exc:
            raise BlockExecutionError(f"failed to create owned compact MLX rows for {role}") from exc
    if getattr(sliced, "allow_fake_compact", False):
        return _CompactRows(sliced, rows)
    raise BlockExecutionError(f"cannot prove owned compact rows for {role} without MLX copy or fake adapter capability")


class _CompactRows:
    def __init__(self, value: Any, rows: int):
        self._value = value
        shape = getattr(value, "shape", (1, rows))
        self.shape = tuple([shape[0], int(rows)] + list(shape[2:])) if len(shape) > 1 else (int(rows),)
        self.detached_compact_rows = True

    def __getitem__(self, item):
        return self._value[item]

    def assign_rows(self, offset, rows, update):
        assign = getattr(self._value, "assign_rows", None)
        if assign is not None:
            return assign(offset, rows, update)
        self._value[:, offset:offset + rows] = update


def _write_rows(base: Any, offset: int, rows: int, update: Any, *, role: str, row_origin: int = 0) -> Any:
    if base is None:
        raise BlockExecutionError(f"cannot write rows into unallocated arena slot {role}")
    offset = int(offset) - int(row_origin)
    if offset < 0:
        raise BlockExecutionError(f"arena slot {role} row origin {row_origin} cannot accept write")
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
