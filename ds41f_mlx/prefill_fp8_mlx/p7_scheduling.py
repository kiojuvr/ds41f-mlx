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
            try:
                engram = language_model.layers[int(layer_id)].engram
            except Exception as exc:
                raise P7SchedulingError(f"missing Engram layer {layer_id}") from exc
            if type(engram).__name__ != "DiskEngramEmbedding" and not getattr(engram, "_p7_disk_engram", False):
                raise P7SchedulingError(f"Engram layer {layer_id} is not DiskEngramEmbedding under SSD mode")

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


@dataclass
class PendingEngramRequest:
    table: int
    layer: int
    command_index: int
    ids: Any


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
        nxt = self._next_consumer_after(command.index, self.active_layer)
        if nxt is not None:
            self._submit_for(nxt, arena, hash_slice_fn)

    def before_consumer(self, command: SweepCommand, arena: Any, hash_slice_fn: Callable[..., Any], ids: Any | None = None) -> Any | None:
        self._assert_live()
        layer = int(command.layer) if command.layer is not None else None
        if layer not in self.TABLE_TO_LAYER.values():
            return None
        if ids is None:
            ids = hash_slice_fn(arena.engram.hashes.value, command.offset, command.rows, layer, self.language_model)
        if ids is None:
            return None
        self.telemetry.record("engram_consume", layer=layer, offset=command.offset, rows=command.rows, command_index=command.index, exact_prefetch=self._ids_equal(self.pending.ids, ids) if self.pending is not None else False)
        if self.pending is not None and not self._ids_equal(self.pending.ids, ids):
            self.telemetry.record("engram_prefetch_mismatch", layer=layer, command_index=command.index)
        return ids

    def after_consumer(self, command: SweepCommand, arena: Any, hash_slice_fn: Callable[..., Any]) -> None:
        layer = int(command.layer) if command.layer is not None else None
        if layer not in self.TABLE_TO_LAYER.values():
            return
        nxt = self._next_consumer_after(command.index, layer)
        if nxt is not None:
            self._submit_for(nxt, arena, hash_slice_fn)
        else:
            self.telemetry.record("engram_table_pipeline_complete", layer=layer)
            self.pending = None

    def drain(self) -> None:
        if self.donor is not None and hasattr(self.donor, "drain"):
            self.donor.drain()
        self.telemetry.record("engram_prefetch_drain")
        self.pending = None

    def revoke(self) -> None:
        self.revoked = True
        self.drain()

    def _submit_for(self, command: SweepCommand, arena: Any, hash_slice_fn: Callable[..., Any]) -> None:
        self._assert_live()
        layer = int(command.layer)
        ids = hash_slice_fn(arena.engram.hashes.value, command.offset, command.rows, layer, self.language_model)
        if ids is None:
            return
        engram = self.language_model.layers[layer].engram
        self.donor.submit(engram, ids)
        table = 0 if layer == 1 else 1
        self.pending = PendingEngramRequest(table, layer, command.index, ids)
        self.telemetry.record("engram_prefetch_submit", table=table, layer=layer, offset=command.offset, rows=command.rows, command_index=command.index, ids=ids)

    def _next_consumer_after(self, index: int, layer: int) -> SweepCommand | None:
        for cmd in self.commands:
            if cmd.index > index and cmd.kind is SweepCommandKind.ENCODE_ROWS and cmd.layer == layer:
                return cmd
        return None

    def _assert_live(self) -> None:
        if self.revoked:
            raise P7SchedulingError("stale P7 coordinator cannot schedule after revocation")

    @staticmethod
    def _ids_equal(a: Any, b: Any) -> bool:
        if a is b:
            return True
        try:
            import numpy as np  # type: ignore
            return bool(np.array_equal(a, b))
        except Exception:
            return a == b


class SchedulingCoordinator:
    def __init__(self, language_model: Any, *, mx: Any | None = None, qualification: LoaderQualification | None = None, telemetry: SchedulingTelemetry | None = None):
        self.language_model = language_model
        self.telemetry = telemetry or SchedulingTelemetry()
        self.residency = ResidencyPolicy(qualification)
        self.residency.admit(language_model, self.telemetry)
        self.read_ahead = ReadAheadPolicy(self.residency)
        self.materialization = MaterializationPolicy(mx, self.telemetry)
        self.engram = EngramPrefetchController(language_model, self.telemetry)
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

    def submit(self, embed: Any, ids: Any) -> None:
        self.calls.append(("submit", embed, ids))

    def drain(self) -> None:
        self.calls.append(("drain", None, None))
