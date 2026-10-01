"""P7 command-driven overlap scheduling for the qualified ds41f/oMLX target.

Base backend policy is deliberately narrow:
FULL_RESIDENT_BACKBONE_SSD_ENGRAM.  Backbone and MoE weights must already be
resident/materialized by the qualified loader; only SSD Engram is scheduled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

from ds41f_mlx.prefill_fp8_mlx.planner import SweepCommand, SweepCommandKind

P7_ENGRAM_TILE = 2048


class P7SchedulingError(RuntimeError):
    """P7 scheduling/admission failure."""


class P7BackendPolicy(str, Enum):
    FULL_RESIDENT_BACKBONE_SSD_ENGRAM = "FULL_RESIDENT_BACKBONE_SSD_ENGRAM"
    EXPERT_OFFLOAD = "EXPERT_OFFLOAD"


@dataclass(frozen=True)
class LoaderQualification:
    preserve_mtp: bool = False
    engram_ssd_offload: bool = True
    moe_expert_offload_resident_fraction: float | None = None
    ordinary_parameters_materialized: bool = True


@dataclass
class SchedulingTelemetry:
    events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, event: str, **fields: Any) -> None:
        self.events.append({"event": event, **fields})


class ResidencyPolicy:
    """Admission/readiness policy for FULL_RESIDENT_BACKBONE_SSD_ENGRAM."""

    def __init__(self, qualification: LoaderQualification | None = None):
        self.qualification = qualification or LoaderQualification()
        self.admitted = False

    def admit(self, language_model: Any, telemetry: SchedulingTelemetry) -> None:
        q = self.qualification
        if q.preserve_mtp is not False:
            raise P7SchedulingError("P7 base backend requires preserve_mtp=False")
        if q.engram_ssd_offload is not True:
            raise P7SchedulingError("P7 base backend requires engram_ssd_offload=True")
        if q.moe_expert_offload_resident_fraction is not None:
            raise P7SchedulingError("P7 base backend rejects expert offload")
        if not q.ordinary_parameters_materialized:
            raise P7SchedulingError("P7 base backend requires loader-materialized ordinary parameters")
        if getattr(language_model, "_moe_offload_plan", None) is not None:
            raise P7SchedulingError("P7 base backend rejects active _moe_offload_plan")
        for obj in _walk_model_objects(language_model):
            if type(obj).__name__ == "OffloadedExpert":
                raise P7SchedulingError("P7 base backend rejects installed OffloadedExpert modules")
        self._require_disk_engram(language_model)
        self.admitted = True
        telemetry.record("residency_admitted", backend=P7BackendPolicy.FULL_RESIDENT_BACKBONE_SSD_ENGRAM.value)

    def _require_disk_engram(self, language_model: Any) -> None:
        for layer_id in getattr(getattr(language_model, "_config", object()), "engram_layer_ids", (1, 14)):
            embed = _engram_storage_embed(language_model, int(layer_id))
            if type(embed).__name__ != "DiskEngramEmbedding" and not getattr(embed, "_p7_disk_engram", False):
                raise P7SchedulingError(f"Engram layer {layer_id} embed is not DiskEngramEmbedding under SSD mode")

    def verify_layer_ready(self, layer: int, telemetry: SchedulingTelemetry) -> str:
        if not self.admitted:
            raise P7SchedulingError("P7 residency policy was not admitted")
        telemetry.record("layer_resident_ready", layer=int(layer))
        return "RESIDENT_ALREADY_READY"


def _walk_model_objects(root: Any) -> Iterable[Any]:
    stack = [root]
    seen: set[int] = set()
    while stack:
        obj = stack.pop()
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        yield obj
        if isinstance(obj, (str, bytes, int, float, bool, type(None))):
            continue
        if isinstance(obj, dict):
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple, set)):
            stack.extend(obj)
        else:
            for name, value in getattr(obj, "__dict__", {}).items():
                if name.startswith("__"):
                    continue
                stack.append(value)


class ReadAheadPolicy:
    def __init__(self, residency: ResidencyPolicy):
        self.residency = residency

    def ssd_read_ahead(self, layer: int | None, telemetry: SchedulingTelemetry) -> str:
        if layer is None:
            raise P7SchedulingError("SSD_READ_AHEAD requires a target layer")
        self.residency.verify_layer_ready(layer, telemetry)
        telemetry.record("ssd_read_ahead", layer=int(layer), result="ALREADY_READY")
        return "ALREADY_READY"

    def begin_layer(self, layer: int | None, telemetry: SchedulingTelemetry) -> str:
        if layer is None:
            raise P7SchedulingError("BEGIN_LAYER requires a target layer")
        return self.residency.verify_layer_ready(layer, telemetry)

    def end_layer(self, layer: int | None, telemetry: SchedulingTelemetry) -> str:
        if layer is None:
            raise P7SchedulingError("END_LAYER requires a target layer")
        telemetry.record("layer_end_no_evict", layer=int(layer))
        return "NO_EVICTION_RESIDENT_BACKBONE"


class MaterializationPolicy:
    def __init__(self, mx: Any | None = None, telemetry: SchedulingTelemetry | None = None):
        self.mx = mx
        self.telemetry = telemetry

    def before_engram_dependency(self, h_chunk: Any, pre_chunk: Any) -> None:
        if self.mx is not None and hasattr(self.mx, "async_eval"):
            self.mx.async_eval(h_chunk, pre_chunk)
            if self.telemetry:
                self.telemetry.record("mx_async_eval", boundary="before_engram")

    def after_engram_incorporated(self, h_after: Any, pre_chunk: Any) -> None:
        if self.mx is not None and hasattr(self.mx, "async_eval"):
            self.mx.async_eval(h_after, pre_chunk)
            if self.telemetry:
                self.telemetry.record("mx_async_eval", boundary="after_engram")


@dataclass(frozen=True)
class PendingEngramRequest:
    table: int
    layer: int
    target_command_index: int
    offset: int
    rows: int
    donor_issue_observed: bool
    micro_index: int = 0
    command_offset: int = 0


@dataclass(frozen=True)
class EngramMicrotarget:
    table: int
    layer: int
    command_index: int
    command_offset: int
    command_rows: int
    micro_index: int
    local_offset: int
    offset: int
    rows: int


class EngramPrefetchController:
    """Borrow model-owned EngramPrefetch and submit exact consumer chunks."""

    TABLE_TO_LAYER = {0: 1, 1: 14}

    def __init__(self, language_model: Any, telemetry: SchedulingTelemetry):
        self.language_model = language_model
        self.telemetry = telemetry
        self.donor = getattr(language_model, "_engram_prefetch", None)
        if self.donor is None:
            raise P7SchedulingError("P7 requires model-owned language_model._engram_prefetch donor")
        self.active_table: int | None = None
        self.active_layer: int | None = None
        self.commands: tuple[SweepCommand, ...] = ()
        self.revoked = False
        self.pending: PendingEngramRequest | None = None

    def set_command_stream(self, commands: Iterable[SweepCommand]) -> None:
        self.commands = tuple(commands)

    def activate(self, table: int, command: SweepCommand, arena: Any, hash_slice_fn: Callable[..., Any]) -> None:
        self._assert_live()
        self.active_table = int(table)
        self.active_layer = self.TABLE_TO_LAYER[int(table)]
        self.telemetry.record("engram_prefetch_activate", table=table, layer=self.active_layer, command_index=command.index)
        target = self._next_microtarget_after(command.index, self.active_layer)
        if target is not None:
            self._submit_microtarget(target, arena, hash_slice_fn)

    def before_microtarget(self, target: EngramMicrotarget) -> bool:
        self._assert_live()
        pending = self.pending
        logical_match = pending is not None and pending.target_command_index == target.command_index and pending.layer == target.layer and pending.offset == target.offset and pending.rows == target.rows and pending.micro_index == target.micro_index
        if pending is not None and not logical_match:
            self.telemetry.record("engram_prefetch_mismatch", layer=target.layer, command_index=target.command_index, micro_index=target.micro_index, pending_command_index=pending.target_command_index)
            raise P7SchedulingError("Engram microtarget did not match scheduled prefetch target")
        self.telemetry.record("engram_consume", table=target.table, layer=target.layer, command_index=target.command_index, transformer_command_index=target.command_index, command_offset=target.command_offset, micro_index=target.micro_index, micro_offset=target.offset, rows=target.rows, logical_match=logical_match, donor_issue_observed=bool(pending.donor_issue_observed) if pending is not None else False, consume_observed=True)
        return logical_match

    def after_microtarget(self, target: EngramMicrotarget, arena: Any, hash_slice_fn: Callable[..., Any]) -> None:
        nxt = self._next_microtarget_after(target.command_index, target.layer, after_target=target)
        if nxt is not None:
            self._submit_microtarget(nxt, arena, hash_slice_fn)
        else:
            self.telemetry.record("engram_table_pipeline_complete", layer=target.layer)
            self.pending = None

    def before_consumer(self, command: SweepCommand, arena: Any, hash_slice_fn: Callable[..., Any], ids: Any | None = None) -> Any | None:
        targets = self.microtargets_for_command(command)
        if not targets:
            return ids
        self.before_microtarget(targets[0])
        return ids

    def after_consumer(self, command: SweepCommand, arena: Any, hash_slice_fn: Callable[..., Any]) -> None:
        targets = self.microtargets_for_command(command)
        if targets:
            self.after_microtarget(targets[-1], arena, hash_slice_fn)

    def drain(self) -> None:
        if self.donor is not None and hasattr(self.donor, "drain"):
            self.donor.drain()
        self.telemetry.record("engram_prefetch_drain")
        self.pending = None

    def revoke(self) -> None:
        self.revoked = True
        self.drain()

    def _submit_microtarget(self, target: EngramMicrotarget, arena: Any, hash_slice_fn: Callable[..., Any]) -> None:
        self._assert_live()
        ids = hash_slice_fn(arena.engram.hashes.value, target.offset, target.rows, target.layer, self.language_model)
        if ids is None:
            return
        embed = _engram_storage_embed(self.language_model, target.layer)
        self.donor.submit(embed, ids)
        donor_issue_observed = _donor_issue_observed(self.donor, embed)
        self.pending = PendingEngramRequest(target.table, target.layer, target.command_index, target.offset, target.rows, donor_issue_observed, target.micro_index, target.command_offset)
        self.telemetry.record("engram_prefetch_submit", table=target.table, layer=target.layer, transformer_command_index=target.command_index, command_index=target.command_index, command_offset=target.command_offset, micro_index=target.micro_index, micro_offset=target.offset, rows=target.rows, logical_consumer_command_index=target.command_index, donor_issue_observed=donor_issue_observed)

    def microtargets_for_command(self, command: SweepCommand) -> tuple[EngramMicrotarget, ...]:
        layer = int(command.layer) if command.layer is not None else None
        if layer not in self.TABLE_TO_LAYER.values() or command.kind is not SweepCommandKind.ENCODE_ROWS:
            return ()
        table = 0 if layer == 1 else 1
        out = []
        local = 0
        micro_index = 0
        while local < int(command.rows):
            rows = min(P7_ENGRAM_TILE, int(command.rows) - local)
            out.append(EngramMicrotarget(table, layer, int(command.index), int(command.offset), int(command.rows), micro_index, local, int(command.offset) + local, rows))
            local += rows
            micro_index += 1
        return tuple(out)

    def _next_microtarget_after(self, index: int, layer: int, after_target: EngramMicrotarget | None = None) -> EngramMicrotarget | None:
        if after_target is not None:
            targets = self.microtargets_for_command(self._command_by_index(after_target.command_index))
            for target in targets:
                if target.micro_index > after_target.micro_index:
                    return target
            index = after_target.command_index
        for cmd in self.commands:
            if cmd.index > index and cmd.kind is SweepCommandKind.ENCODE_ROWS and cmd.layer == layer:
                targets = self.microtargets_for_command(cmd)
                return targets[0] if targets else None
        return None

    def _command_by_index(self, index: int) -> SweepCommand:
        for cmd in self.commands:
            if cmd.index == index:
                return cmd
        raise P7SchedulingError(f"missing command {index} in P7 command stream")

    def _assert_live(self) -> None:
        if self.revoked:
            raise P7SchedulingError("stale P7 coordinator cannot schedule after revocation")



def _slice_sequence(value: Any, offset: int, rows: int) -> Any:
    try:
        return value[:, int(offset):int(offset) + int(rows)]
    except Exception:
        return value


def _concat_sequence(values: list[Any], *, like: Any, p8_optimizer: Any | None = None, command: SweepCommand | None = None) -> Any:
    if not values:
        return like
    if len(values) == 1:
        result = values[0]
        if p8_optimizer is not None and getattr(p8_optimizer, "enabled", False):
            p8_optimizer.shape_registry.record_microtile_concat(outputs=values, result=result, references_released=True, command=command)
        return result
    concat_rows = getattr(values[0], "concat_rows", None)
    if concat_rows is not None:
        result = concat_rows(values)
        if p8_optimizer is not None and getattr(p8_optimizer, "enabled", False):
            p8_optimizer.shape_registry.record_microtile_concat(outputs=values, result=result, references_released=True, command=command)
        return result
    try:
        import mlx.core as mx  # type: ignore
        result = mx.concatenate(values, axis=1)
        if p8_optimizer is not None and getattr(p8_optimizer, "enabled", False):
            p8_optimizer.shape_registry.record_microtile_concat(outputs=values, result=result, references_released=True, command=command)
        return result
    except Exception as exc:
        if all(hasattr(v, "shape") for v in values):
            result = _MicrotileConcat(values)
            if p8_optimizer is not None and getattr(p8_optimizer, "enabled", False):
                p8_optimizer.shape_registry.record_microtile_concat(outputs=values, result=result, references_released=False, command=command)
            return result
        raise P7SchedulingError("cannot reassemble Engram microtiles") from exc


class _MicrotileConcat:
    def __init__(self, values: list[Any]):
        self.parts = tuple(values)
        first_shape = getattr(values[0], "shape", ())
        rows = sum(int(getattr(v, "shape", (0, 0))[1]) for v in values)
        self.shape = (first_shape[0], rows, *first_shape[2:]) if len(first_shape) >= 2 else first_shape
        self.last_slice = (0, rows)


def _engram_storage_embed(language_model: Any, layer: int) -> Any:
    try:
        engram = language_model.layers[int(layer)].engram
    except Exception as exc:
        raise P7SchedulingError(f"missing Engram wrapper for layer {layer}") from exc
    if not hasattr(engram, "embed"):
        raise P7SchedulingError(f"Engram layer {layer} has no storage embed")
    return engram.embed


def _donor_issue_observed(donor: Any, embed: Any) -> bool:
    pending = getattr(donor, "_pending", None)
    if isinstance(pending, tuple) and len(pending) >= 2 and pending[0] is embed:
        return True
    prefetched = getattr(embed, "_prefetched", None)
    return isinstance(prefetched, tuple) and len(prefetched) >= 2


class SchedulingCoordinator:
    def __init__(self, language_model: Any, *, mx: Any | None = None, qualification: LoaderQualification | None = None, telemetry: SchedulingTelemetry | None = None, p8_optimizer: Any | None = None):
        self.language_model = language_model
        self.telemetry = telemetry or SchedulingTelemetry()
        self.residency = ResidencyPolicy(qualification)
        self.residency.admit(language_model, self.telemetry)
        self.read_ahead = ReadAheadPolicy(self.residency)
        self.materialization = MaterializationPolicy(mx, self.telemetry)
        self.engram = EngramPrefetchController(language_model, self.telemetry)
        self.p8_optimizer = p8_optimizer
        self.active = False
        self.revoked = False

    def set_command_stream(self, commands: Iterable[SweepCommand]) -> None:
        self.engram.set_command_stream(commands)

    def handle_command(self, command: SweepCommand, arena: Any, hash_slice_fn: Callable[..., Any]) -> None:
        self._assert_live()
        if command.kind is SweepCommandKind.BEGIN_INVALIDATE:
            self.active = True
            self.telemetry.record("scheduling_active", command_index=command.index)
        elif command.kind is SweepCommandKind.PREFETCH_ENGRAM0:
            self.engram.activate(0, command, arena, hash_slice_fn)
        elif command.kind is SweepCommandKind.PREFETCH_ENGRAM1:
            self.engram.activate(1, command, arena, hash_slice_fn)
        elif command.kind is SweepCommandKind.SSD_READ_AHEAD:
            self.read_ahead.ssd_read_ahead(command.layer, self.telemetry)
        elif command.kind is SweepCommandKind.BEGIN_LAYER:
            self.read_ahead.begin_layer(command.layer, self.telemetry)
        elif command.kind is SweepCommandKind.END_LAYER:
            self.read_ahead.end_layer(command.layer, self.telemetry)

    def before_engram_consumer(self, command: SweepCommand, arena: Any, h_chunk: Any, pre_chunk: Any, hash_slice_fn: Callable[..., Any], ids: Any | None = None) -> Any | None:
        # The runner passes the exact hash slice it will give to layer.engram;
        # this avoids a second hash/history authority for consumer IDs.
        actual_ids = self.engram.before_consumer(command, arena, hash_slice_fn, ids=ids)
        if actual_ids is not None:
            self.materialization.before_engram_dependency(h_chunk, pre_chunk)
        return actual_ids

    def after_engram_consumer(self, command: SweepCommand, arena: Any, h_after: Any, pre_chunk: Any, hash_slice_fn: Callable[..., Any]) -> None:
        self.materialization.after_engram_incorporated(h_after, pre_chunk)
        self.engram.after_consumer(command, arena, hash_slice_fn)

    def apply_engram_micro_pipeline(self, command: SweepCommand, arena: Any, h_chunk: Any, pre_chunk: Any, engram_call: Callable[..., Any], image_mask: Any, hash_slice_fn: Callable[..., Any]) -> Any:
        targets = self.engram.microtargets_for_command(command)
        if not targets:
            return h_chunk
        outputs = []
        for target in targets:
            h_micro = _slice_sequence(h_chunk, target.local_offset, target.rows)
            pre_micro = _slice_sequence(pre_chunk, target.local_offset, target.rows)
            mask_micro = None if image_mask is None else _slice_sequence(image_mask, target.local_offset, target.rows)
            ids = hash_slice_fn(arena.engram.hashes.value, target.offset, target.rows, target.layer, self.language_model)
            if ids is None:
                continue
            self.engram.before_microtarget(target)
            self.materialization.before_engram_dependency(h_micro, pre_micro)
            h_after = engram_call(h_micro, ids, mask_micro)
            self.materialization.after_engram_incorporated(h_after, pre_micro)
            outputs.append(h_after)
            self.engram.after_microtarget(target, arena, hash_slice_fn)
        return _concat_sequence(outputs, like=h_chunk, p8_optimizer=self.p8_optimizer, command=command)

    def seal_success(self) -> None:
        self.engram.drain()
        self.active = False
        self.telemetry.record("scheduling_sealed")

    def revoke(self) -> None:
        self.revoked = True
        self.engram.revoke()
        self.active = False
        self.telemetry.record("scheduling_revoked")

    def _assert_live(self) -> None:
        if self.revoked:
            raise P7SchedulingError("stale P7 coordinator cannot schedule after revocation")


@dataclass
class RecordingDonor:
    """Test double with the same submit/drain surface as oMLX EngramPrefetch."""

    calls: list[tuple[str, Any, Any]] = field(default_factory=list)
    _pending: tuple[Any, object] | None = None

    def submit(self, embed: Any, ids: Any) -> None:
        self.calls.append(("submit", embed, ids))
        marker = object()
        try:
            embed._prefetched = (None, marker)
        except Exception:
            pass
        self._pending = (embed, marker)

    def drain(self) -> None:
        self.calls.append(("drain", None, None))
        if self._pending is not None:
            embed, _marker = self._pending
            try:
                embed._prefetched = None
            except Exception:
                pass
        self._pending = None
