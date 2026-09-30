"""DwarfStar V4.1 sweep planner adapter for official FP8/MLX prefill.

P1 scope: normalize the existing native DwarfStar-derived sweep planner into a
production-package API.  This module does not execute model math and does not
wrap the oMLX whole-prefix layer loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from ds41f_mlx.native_prefill import (
    SWEEP_COMMAND_NAMES,
    SWEEP_PHASE_NAMES,
    NativePrefillLibrary,
    NativeSweepConfig,
    native_sweep_config_from_model,
    native_sweep_config_static,
)


class SweepCommandKind(str, Enum):
    BEGIN_INVALIDATE = "begin_sweep_checkpoint_invalid"
    PREFETCH_ENGRAM0 = "prefetch_engram_table0"
    PREFETCH_ENGRAM1 = "prefetch_engram_table1"
    SSD_READ_AHEAD = "ssd_read_ahead"
    BEGIN_LAYER = "begin_layer"
    ENCODE_ROWS = "encode_rows"
    DECODER_PREPARE_SUFFIX = "decoder_prepare_suffix"
    SWAP_HC_AFTER_LAYER = "swap_hc_after_layer"
    PUBLISH_FRONTIER = "publish_state_frontier"
    ENCODER_ONLY_COMPLETE_INVALID = "encoder_only_complete_still_invalid"
    DECODER_PENDING_INVALID = "decoder_pending_still_invalid"
    ENCODE_OUTPUT_HEAD = "encode_output_head"
    READ_LOGITS = "read_logits"
    CHECKPOINT_MAY_COMMIT = "checkpoint_may_commit"
    END_LAYER = "end_layer"
    UNKNOWN = "unknown"


class SweepPhase(str, Enum):
    SWEEP = "sweep"
    ENCODER = "encoder"
    DECODER_FULL = "decoder_full"
    DECODER_SUFFIX = "decoder_suffix"
    DEFERRED_DECODER = "deferred_decoder"
    FINAL_OUTPUT = "final_output"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SweepCommand:
    """A normalized DwarfStar sweep command."""

    index: int
    kind: SweepCommandKind
    raw_kind: int
    layer: int | None
    offset: int
    rows: int
    phase: SweepPhase
    raw_phase: int
    checkpoint_valid: bool

    @property
    def is_layer_command(self) -> bool:
        return self.layer is not None

    @property
    def is_materialization_boundary(self) -> bool:
        return self.kind in {
            SweepCommandKind.BEGIN_INVALIDATE,
            SweepCommandKind.PUBLISH_FRONTIER,
            SweepCommandKind.ENCODER_ONLY_COMPLETE_INVALID,
            SweepCommandKind.DECODER_PENDING_INVALID,
            SweepCommandKind.READ_LOGITS,
            SweepCommandKind.CHECKPOINT_MAY_COMMIT,
        }

    def to_json(self) -> dict[str, object]:
        return {
            "index": self.index,
            "kind": self.kind.value,
            "raw_kind": self.raw_kind,
            "layer": self.layer,
            "offset": self.offset,
            "rows": self.rows,
            "phase": self.phase.value,
            "raw_phase": self.raw_phase,
            "checkpoint_valid": self.checkpoint_valid,
            "is_materialization_boundary": self.is_materialization_boundary,
        }


@dataclass(frozen=True)
class SweepAllocation:
    semantic_role: str
    size_bytes: int
    alignment: int
    representation: str
    first_use_step: int
    last_use_step: int
    owner: str
    alias_reuse_class: str
    persistence_class: str
    survives_encoder_only: bool
    survives_deferred_decoder: bool

    @classmethod
    def from_json(cls, item: dict[str, Any]) -> "SweepAllocation":
        return cls(
            semantic_role=str(item["semantic_role"]),
            size_bytes=int(item["size_bytes"]),
            alignment=int(item["alignment"]),
            representation=str(item["representation"]),
            first_use_step=int(item["first_use_step"]),
            last_use_step=int(item["last_use_step"]),
            owner=str(item["owner"]),
            alias_reuse_class=str(item["alias_reuse_class"]),
            persistence_class=str(item["persistence_class"]),
            survives_encoder_only=bool(item["survives_encoder_only"]),
            survives_deferred_decoder=bool(item["survives_deferred_decoder"]),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "semantic_role": self.semantic_role,
            "size_bytes": self.size_bytes,
            "alignment": self.alignment,
            "representation": self.representation,
            "first_use_step": self.first_use_step,
            "last_use_step": self.last_use_step,
            "owner": self.owner,
            "alias_reuse_class": self.alias_reuse_class,
            "persistence_class": self.persistence_class,
            "survives_encoder_only": self.survives_encoder_only,
            "survives_deferred_decoder": self.survives_deferred_decoder,
        }


@dataclass(frozen=True)
class SweepPlan:
    """Normalized plan that must drive production prefill execution order."""

    count: int
    prefill_cap: int
    encoder_chunk: int
    wide: bool
    decoder_suffix: bool
    encoder_only: bool
    resume_encoder: bool
    defer_decoder_candidate: bool
    checkpoint_valid_during_sweep: bool
    checkpoint_valid_after_sweep: bool
    encoder_row_layer_work: int
    decoder_suffix_row_layer_work: int
    total_row_layer_work: int
    allocations: tuple[SweepAllocation, ...]
    commands: tuple[SweepCommand, ...]
    command_counts: dict[str, int]
    prefetch_event_count: int
    checkpoint_transition_count: int

    @classmethod
    def from_native_json(cls, plan: dict[str, Any]) -> "SweepPlan":
        commands = tuple(_command_from_native(i, item) for i, item in enumerate(plan["commands"]))
        allocations = tuple(SweepAllocation.from_json(item) for item in plan.get("allocations", ()))
        return cls(
            count=int(plan["count"]),
            prefill_cap=int(plan["prefill_cap"]),
            encoder_chunk=int(plan["encoder_chunk"]),
            wide=bool(plan["wide"]),
            decoder_suffix=bool(plan["decoder_suffix"]),
            encoder_only=bool(plan["encoder_only"]),
            resume_encoder=bool(plan["resume_encoder"]),
            defer_decoder_candidate=bool(plan["defer_decoder_candidate"]),
            checkpoint_valid_during_sweep=bool(plan["checkpoint_valid_during_sweep"]),
            checkpoint_valid_after_sweep=bool(plan["checkpoint_valid_after_sweep"]),
            encoder_row_layer_work=int(plan["encoder_row_layer_work"]),
            decoder_suffix_row_layer_work=int(plan["decoder_suffix_row_layer_work"]),
            total_row_layer_work=int(plan["total_row_layer_work"]),
            allocations=allocations,
            commands=commands,
            command_counts={str(k): int(v) for k, v in plan.get("command_counts", {}).items()},
            prefetch_event_count=int(plan["prefetch_event_count"]),
            checkpoint_transition_count=int(plan["checkpoint_transition_count"]),
        )

    @property
    def encode_commands(self) -> tuple[SweepCommand, ...]:
        return tuple(c for c in self.commands if c.kind is SweepCommandKind.ENCODE_ROWS)

    @property
    def publication_commands(self) -> tuple[SweepCommand, ...]:
        return tuple(c for c in self.commands if c.kind is SweepCommandKind.PUBLISH_FRONTIER)

    @property
    def materialization_boundaries(self) -> tuple[SweepCommand, ...]:
        return tuple(c for c in self.commands if c.is_materialization_boundary)

    def commands_by_kind(self, kind: SweepCommandKind) -> tuple[SweepCommand, ...]:
        return tuple(c for c in self.commands if c.kind is kind)

    def to_json(self) -> dict[str, object]:
        return {
            "count": self.count,
            "prefill_cap": self.prefill_cap,
            "encoder_chunk": self.encoder_chunk,
            "wide": self.wide,
            "decoder_suffix": self.decoder_suffix,
            "encoder_only": self.encoder_only,
            "resume_encoder": self.resume_encoder,
            "defer_decoder_candidate": self.defer_decoder_candidate,
            "checkpoint_valid_during_sweep": self.checkpoint_valid_during_sweep,
            "checkpoint_valid_after_sweep": self.checkpoint_valid_after_sweep,
            "encoder_row_layer_work": self.encoder_row_layer_work,
            "decoder_suffix_row_layer_work": self.decoder_suffix_row_layer_work,
            "total_row_layer_work": self.total_row_layer_work,
            "allocations": [a.to_json() for a in self.allocations],
            "commands": [c.to_json() for c in self.commands],
            "command_counts": dict(self.command_counts),
            "prefetch_event_count": self.prefetch_event_count,
            "checkpoint_transition_count": self.checkpoint_transition_count,
        }


@dataclass(frozen=True)
class SweepPlanner:
    """Adapter over the native DwarfStar-derived V4.1 sweep planner."""

    native: NativePrefillLibrary

    def build(self, cfg: NativeSweepConfig) -> SweepPlan:
        native_plan = self.native.build_sweep_plan(cfg).to_json()
        return SweepPlan.from_native_json(native_plan)

    def build_static(
        self,
        *,
        ctx: int,
        remaining: int,
        dim: int = 5120,
        hc_mult: int = 4,
        vocab_size: int = 129280,
        encoder_resident: bool = False,
        encoder_only: bool = False,
        resume_encoder: bool = False,
        memory_budget_bytes: int = 0,
    ) -> SweepPlan:
        cfg = native_sweep_config_static(
            ctx=ctx,
            remaining=remaining,
            dim=dim,
            hc_mult=hc_mult,
            vocab_size=vocab_size,
            encoder_resident=encoder_resident,
            encoder_only=encoder_only,
            resume_encoder=resume_encoder,
            memory_budget_bytes=memory_budget_bytes,
        )
        return self.build(cfg)

    def build_from_model(
        self,
        language_model: Any,
        *,
        ctx: int,
        remaining: int,
        encoder_resident: bool = False,
        encoder_only: bool = False,
        resume_encoder: bool = False,
        memory_budget_bytes: int = 0,
    ) -> SweepPlan:
        cfg = native_sweep_config_from_model(
            language_model,
            ctx=ctx,
            remaining=remaining,
            encoder_resident=encoder_resident,
            encoder_only=encoder_only,
            resume_encoder=resume_encoder,
            memory_budget_bytes=memory_budget_bytes,
        )
        return self.build(cfg)


def _command_from_native(index: int, item: dict[str, Any]) -> SweepCommand:
    raw_kind = int(item["kind"])
    kind_name = str(item.get("kind_name") or SWEEP_COMMAND_NAMES.get(raw_kind, "unknown"))
    phase_name = str(item.get("phase") or "unknown")
    raw_phase = _raw_phase_from_name(phase_name)
    raw_layer = item.get("layer")
    layer = None if raw_layer is None else int(raw_layer)
    return SweepCommand(
        index=index,
        kind=_kind_from_name(kind_name),
        raw_kind=raw_kind,
        layer=layer,
        offset=int(item["offset"]),
        rows=int(item["rows"]),
        phase=_phase_from_name(phase_name),
        raw_phase=raw_phase,
        checkpoint_valid=bool(item["checkpoint_valid"]),
    )


def _kind_from_name(name: str) -> SweepCommandKind:
    try:
        return SweepCommandKind(name)
    except ValueError:
        return SweepCommandKind.UNKNOWN


def _phase_from_name(name: str) -> SweepPhase:
    try:
        return SweepPhase(name)
    except ValueError:
        return SweepPhase.UNKNOWN


def _raw_phase_from_name(name: str) -> int:
    for phase_id, phase_name in SWEEP_PHASE_NAMES.items():
        if phase_name == name:
            return int(phase_id)
    try:
        return int(name)
    except ValueError:
        return -1


def iter_command_windows(commands: Iterable[SweepCommand], *, boundary: SweepCommandKind) -> Iterable[tuple[SweepCommand, ...]]:
    """Yield command windows ending at a boundary kind.

    This utility is for future executor/materialization code.  It keeps P1's
    command ownership explicit without executing any model operations.
    """

    window: list[SweepCommand] = []
    for command in commands:
        window.append(command)
        if command.kind is boundary:
            yield tuple(window)
            window = []
    if window:
        yield tuple(window)
