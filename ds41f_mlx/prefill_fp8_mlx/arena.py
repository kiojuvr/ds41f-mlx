"""Request-owned DwarfStar carry arena for FP8/MLX prefill.

P2 scope: establish ownership, alias, transaction, suffix, and publication
lifetimes driven by `SweepCommand` semantics.  This module intentionally does
not execute transformer blocks, materialize `DeepseekV41Cache`, export
`PrefillContinuationState`, or perform tensor CPU round-trips.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ds41f_mlx.prefill_fp8_mlx.planner import SweepAllocation, SweepCommand, SweepCommandKind, SweepPhase, SweepPlan


class ArenaTransactionError(RuntimeError):
    """Invalid DwarfStar arena transaction transition."""


class TensorOwnership(str, Enum):
    OWNED = "owned"
    VIEW = "view"
    EXTERNAL_HANDLE = "external_handle"
    UNBOUND = "unbound"


@dataclass
class TensorSlot:
    """A tensor handle plus ownership metadata.

    `value` is deliberately opaque: MLX arrays can be stored without converting
    them to CPU.  P2 tests use placeholder handles only.
    """

    role: str
    ownership: TensorOwnership
    value: Any = None
    alias_reuse_class: str | None = None
    allocation: SweepAllocation | None = None
    row_origin: int = 0
    rows: int | None = None

    @property
    def bound(self) -> bool:
        return self.value is not None

    def to_json(self) -> dict[str, object]:
        return {
            "role": self.role,
            "ownership": self.ownership.value,
            "bound": self.bound,
            "alias_reuse_class": self.alias_reuse_class,
            "allocation": None if self.allocation is None else self.allocation.to_json(),
            "row_origin": self.row_origin,
            "rows": self.rows,
        }


@dataclass(frozen=True)
class ArenaView:
    """Non-owning row view into an arena slot."""

    name: str
    base_role: str
    offset: int
    rows: int
    command_index: int
    layer: int | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "base_role": self.base_role,
            "offset": self.offset,
            "rows": self.rows,
            "command_index": self.command_index,
            "layer": self.layer,
            "ownership": TensorOwnership.VIEW.value,
        }


@dataclass
class AllocationState:
    allocation: SweepAllocation
    slot: TensorSlot
    active: bool = False

    @classmethod
    def from_allocation(cls, allocation: SweepAllocation) -> "AllocationState":
        return cls(
            allocation=allocation,
            slot=TensorSlot(
                role=allocation.semantic_role,
                ownership=TensorOwnership.UNBOUND,
                alias_reuse_class=allocation.alias_reuse_class,
                allocation=allocation,
            ),
        )

    def update_for_command(self, command_index: int) -> None:
        self.active = self.allocation.first_use_step <= command_index <= self.allocation.last_use_step
        if command_index > self.allocation.last_use_step and self.allocation.persistence_class in {"layer-persistent", "stage-local reusable scratch"}:
            # Drop transient tensor handles when their native lifetime ends so
            # real MLX graph references will not be retained accidentally.
            self.slot.value = None
            self.slot.ownership = TensorOwnership.UNBOUND

    def to_json(self) -> dict[str, object]:
        return {"allocation": self.allocation.to_json(), "slot": self.slot.to_json(), "active": self.active}


@dataclass
class TransactionState:
    valid: bool = True
    begun: bool = False
    committed: bool = False
    failed: bool = False

    def begin_invalid(self) -> None:
        if self.begun and not self.committed:
            raise ArenaTransactionError("sweep transaction already begun")
        if self.failed:
            raise ArenaTransactionError("cannot begin a failed transaction")
        self.valid = False
        self.begun = True
        self.committed = False

    def commit(self) -> None:
        if not self.begun or self.valid or self.failed:
            raise ArenaTransactionError("cannot commit without an invalid in-flight sweep")
        self.valid = True
        self.committed = True

    def fail(self) -> None:
        self.valid = False
        self.failed = True

    def to_json(self) -> dict[str, bool]:
        return {"valid": self.valid, "begun": self.begun, "committed": self.committed, "failed": self.failed}


@dataclass
class PublicationEvent:
    layer: int
    offset: int
    rows: int
    command_index: int
    checkpoint_valid: bool
    visible_after_commit_only: bool = True

    def to_json(self) -> dict[str, object]:
        return {
            "layer": self.layer,
            "offset": self.offset,
            "rows": self.rows,
            "command_index": self.command_index,
            "checkpoint_valid": self.checkpoint_valid,
            "visible_after_commit_only": self.visible_after_commit_only,
        }


@dataclass
class PublicationState:
    """Shared source/consumer publication handles owned by the request arena."""

    shared: dict[str, Any] = field(default_factory=lambda: {"kv": None, "index_k": None, "idx": None, "candidates": None, "topk": None})
    pending_events: list[PublicationEvent] = field(default_factory=list)
    committed_events: list[PublicationEvent] = field(default_factory=list)

    def publish(self, event: PublicationEvent) -> None:
        self.pending_events.append(event)

    def commit(self) -> None:
        self.committed_events.extend(self.pending_events)
        self.pending_events.clear()

    def to_json(self) -> dict[str, object]:
        return {
            "shared_keys": sorted(self.shared),
            "bound_shared_keys": sorted(k for k, v in self.shared.items() if v is not None),
            "pending_events": [e.to_json() for e in self.pending_events],
            "committed_events": [e.to_json() for e in self.committed_events],
        }


@dataclass
class CompressorPendingState:
    by_layer: dict[int, dict[str, TensorSlot]] = field(default_factory=dict)

    def ensure_layer(self, layer: int) -> dict[str, TensorSlot]:
        return self.by_layer.setdefault(
            int(layer),
            {
                "kv": TensorSlot(f"compressor_pending.layer{layer}.kv", TensorOwnership.UNBOUND),
                "gate": TensorSlot(f"compressor_pending.layer{layer}.gate", TensorOwnership.UNBOUND),
            },
        )

    def to_json(self) -> dict[str, object]:
        return {str(layer): {name: slot.to_json() for name, slot in slots.items()} for layer, slots in self.by_layer.items()}


@dataclass
class EngramState:
    hashes: TensorSlot = field(default_factory=lambda: TensorSlot("engram.hashes", TensorOwnership.UNBOUND))
    history: TensorSlot = field(default_factory=lambda: TensorSlot("engram.history", TensorOwnership.UNBOUND))

    def to_json(self) -> dict[str, object]:
        return {"hashes": self.hashes.to_json(), "history": self.history.to_json()}


@dataclass
class CarryState:
    current: TensorSlot
    next: TensorSlot
    pre: TensorSlot
    ffn_split: TensorSlot | None = None
    selected_comp: TensorSlot | None = None
    block_mask: TensorSlot | None = None
    swap_count: int = 0

    def swap(self) -> None:
        self.current, self.next = self.next, self.current
        self.swap_count += 1

    def chunk_view(self, command: SweepCommand) -> ArenaView:
        return ArenaView(
            name=f"carry.current.layer{command.layer}.rows{command.offset}:{command.offset + command.rows}",
            base_role=self.current.role,
            offset=command.offset,
            rows=command.rows,
            command_index=command.index,
            layer=command.layer,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "current": self.current.to_json(),
            "next": self.next.to_json(),
            "pre": self.pre.to_json(),
            "ffn_split": None if self.ffn_split is None else self.ffn_split.to_json(),
            "selected_comp": None if self.selected_comp is None else self.selected_comp.to_json(),
            "block_mask": None if self.block_mask is None else self.block_mask.to_json(),
            "swap_count": self.swap_count,
        }


@dataclass
class RequestArena:
    """Request-owned DwarfStar prefill arena.

    The arena is driven by `apply_command`; it has no independent layer loop and
    no decode-cache materialization authority.
    """

    plan: SweepPlan
    tokens: tuple[int, ...]
    base_frontier: int
    input_ids: TensorSlot
    allocations: dict[str, AllocationState]
    carry: CarryState
    engram: EngramState = field(default_factory=EngramState)
    publications: PublicationState = field(default_factory=PublicationState)
    compressor_pending: CompressorPendingState = field(default_factory=CompressorPendingState)
    transaction: TransactionState = field(default_factory=TransactionState)
    suffix_views: list[ArenaView] = field(default_factory=list)
    decoder_prepared_by_layer: dict[int, ArenaView] = field(default_factory=dict)
    encoder_final_h: Any = None
    encoder_final_pre: Any = None
    active_chunk_views: list[ArenaView] = field(default_factory=list)
    retired_chunk_view_count: int = 0
    command_history: list[dict[str, object]] = field(default_factory=list)
    cache_materialized: bool = False
    prefill_continuation_exported: bool = False
    p6_public_frontier: int | None = None
    p6_private_start: int | None = None
    p6_freeze_public_offsets: bool = False
    p6_segment_mode: str | None = None
    p6_segment_origin: int | None = None
    p6_private_layer_frontiers: dict[int, int] = field(default_factory=dict)
    p6_final_cone_origin: int | None = None
    p6_final_cone_rows: int = 0
    p6_final_cone_detached: bool = False
    p6_source_materialized: bool = False
    p6_source_materialization_events: list[dict[str, object]] = field(default_factory=list)

    @classmethod
    def from_plan(
        cls,
        plan: SweepPlan,
        *,
        token_ids: tuple[int, ...] | list[int],
        input_ids: Any = None,
        h_current: Any = None,
        h_next: Any = None,
        pre: Any = None,
        engram_hashes: Any = None,
        engram_history: Any = None,
        base_frontier: int = 0,
    ) -> "RequestArena":
        allocations = {a.semantic_role: AllocationState.from_allocation(a) for a in plan.allocations}
        input_slot = TensorSlot("tokens.input_ids", TensorOwnership.OWNED, input_ids)
        current = _slot_for_role(allocations, "batch_cur_hc", fallback_value=h_current)
        nxt = _slot_for_role(allocations, "batch_next_hc", fallback_value=h_next)
        pre_slot = _slot_for_role(allocations, "carry.pre", fallback_value=pre)
        carry = CarryState(
            current=current,
            next=nxt,
            pre=pre_slot,
            ffn_split=_optional_slot_for_role(allocations, "carry.ffn_split"),
            selected_comp=_optional_slot_for_role(allocations, "carry.selected_comp"),
            block_mask=_optional_slot_for_role(allocations, "carry.block_mask"),
        )
        arena = cls(
            plan=plan,
            tokens=tuple(int(t) for t in token_ids),
            base_frontier=int(base_frontier),
            input_ids=input_slot,
            allocations=allocations,
            carry=carry,
        )
        arena.engram.hashes.value = engram_hashes
        arena.engram.hashes.ownership = TensorOwnership.OWNED if engram_hashes is not None else TensorOwnership.UNBOUND
        arena.engram.history.value = engram_history
        arena.engram.history.ownership = TensorOwnership.OWNED if engram_history is not None else TensorOwnership.UNBOUND
        arena.update_allocation_lifetimes(0)
        return arena

    def update_allocation_lifetimes(self, command_index: int) -> None:
        for state in self.allocations.values():
            state.update_for_command(command_index)

    def active_allocation_roles(self) -> tuple[str, ...]:
        return tuple(sorted(role for role, state in self.allocations.items() if state.active))

    def _retire_transient_views(self) -> None:
        if self.active_chunk_views:
            self.retired_chunk_view_count += len(self.active_chunk_views)
            self.active_chunk_views.clear()

    def bind_slot(self, role: str, value: Any, *, ownership: TensorOwnership = TensorOwnership.OWNED) -> None:
        self.allocations[role].slot.value = value
        self.allocations[role].slot.ownership = ownership

    def apply_command(self, command: SweepCommand) -> None:
        self._retire_transient_views()
        self.update_allocation_lifetimes(command.index)
        self.command_history.append({
            "index": command.index,
            "kind": command.kind.value,
            "semantic_boundary": command.semantic_boundary.value,
            "active_allocation_roles": list(self.active_allocation_roles()),
        })
        if command.kind is SweepCommandKind.BEGIN_INVALIDATE:
            self.transaction.begin_invalid()
        elif command.kind is SweepCommandKind.ENCODE_ROWS:
            self.active_chunk_views.append(self.carry.chunk_view(command))
        elif command.kind is SweepCommandKind.DECODER_PREPARE_SUFFIX:
            self.prepare_decoder_suffix(command)
        elif command.kind is SweepCommandKind.SWAP_HC_AFTER_LAYER:
            if command.layer == 19:
                self.encoder_final_h = self.carry.next.value
                self.encoder_final_pre = self.carry.pre.value
            self.carry.swap()
        elif command.kind is SweepCommandKind.PUBLISH_FRONTIER:
            if command.layer is None:
                raise ArenaTransactionError("publication command requires a layer")
            self.publications.publish(PublicationEvent(command.layer, command.offset, command.rows, command.index, command.checkpoint_valid))
        elif command.kind is SweepCommandKind.CHECKPOINT_MAY_COMMIT:
            self.transaction.commit()
        elif command.kind is SweepCommandKind.ENCODER_ONLY_COMPLETE_INVALID:
            if self.transaction.valid:
                raise ArenaTransactionError("encoder-only boundary must remain checkpoint-invalid")
        elif command.kind is SweepCommandKind.DECODER_PENDING_INVALID:
            if self.transaction.valid:
                raise ArenaTransactionError("deferred decoder pending boundary must remain checkpoint-invalid")
        elif command.kind in {SweepCommandKind.READ_LOGITS, SweepCommandKind.ENCODE_OUTPUT_HEAD}:
            pass

    def apply_commands(self, commands: tuple[SweepCommand, ...] | list[SweepCommand]) -> None:
        for command in commands:
            self.apply_command(command)

    def retained_suffix_rows(self) -> int:
        return sum(view.rows for view in self.suffix_views)

    def to_json(self) -> dict[str, object]:
        return {
            "token_count": len(self.tokens),
            "base_frontier": self.base_frontier,
            "input_ids": self.input_ids.to_json(),
            "transaction": self.transaction.to_json(),
            "carry": self.carry.to_json(),
            "engram": self.engram.to_json(),
            "publications": self.publications.to_json(),
            "compressor_pending": self.compressor_pending.to_json(),
            "suffix_views": [view.to_json() for view in self.suffix_views],
            "decoder_prepared_by_layer": {str(k): v.to_json() for k, v in self.decoder_prepared_by_layer.items()},
            "encoder_final_h_bound": self.encoder_final_h is not None,
            "encoder_final_pre_bound": self.encoder_final_pre is not None,
            "active_chunk_views": [view.to_json() for view in self.active_chunk_views],
            "retired_chunk_view_count": self.retired_chunk_view_count,
            "allocation_roles": sorted(self.allocations),
            "active_allocation_roles": list(self.active_allocation_roles()),
            "cache_materialized": self.cache_materialized,
            "prefill_continuation_exported": self.prefill_continuation_exported,
            "p6_public_frontier": self.p6_public_frontier,
            "p6_private_start": self.p6_private_start,
            "p6_freeze_public_offsets": self.p6_freeze_public_offsets,
            "p6_segment_mode": self.p6_segment_mode,
            "p6_segment_origin": self.p6_segment_origin,
            "p6_private_layer_frontiers": dict(self.p6_private_layer_frontiers),
            "p6_final_cone_origin": self.p6_final_cone_origin,
            "p6_final_cone_rows": self.p6_final_cone_rows,
            "p6_final_cone_detached": self.p6_final_cone_detached,
            "p6_source_materialized": self.p6_source_materialized,
            "p6_source_materialization_events": list(self.p6_source_materialization_events),
            "command_count_applied": len(self.command_history),
        }

    def retire_encoder_range_after_source_boundary(self) -> None:
        """Drop transient full-range HC/pre/hash/selection references at P6 boundaries."""
        self.carry.current.value = None
        self.carry.next.value = None
        self.carry.pre.value = None
        self.input_ids.value = None
        self.encoder_final_h = None
        self.encoder_final_pre = None
        self.active_chunk_views.clear()
        self.suffix_views.clear()
        self.decoder_prepared_by_layer.clear()
        self.publications.shared["idx"] = None
        self.publications.shared["candidates"] = None
        self.engram.hashes.value = None
        self.engram.hashes.ownership = TensorOwnership.UNBOUND

    def materialize_p6_source_boundary(self, *, command_index: int, frontier: int) -> None:
        self.p6_source_materialized = True
        self.p6_source_materialization_events.append({"command_index": int(command_index), "frontier": int(frontier)})

    def detach_final_decoder_cone(self, *, origin: int, rows: int, h_value: Any | None = None, pre_value: Any | None = None, row_origin: int | None = None) -> None:
        """Record an owned bounded final decoder cone and release full parents."""
        self.p6_final_cone_origin = int(origin)
        self.p6_final_cone_rows = int(rows)
        self.p6_final_cone_detached = True
        local_origin = int(origin if row_origin is None else row_origin)
        if h_value is not None:
            self.carry.current.value = h_value
            self.carry.current.row_origin = local_origin
            self.carry.current.rows = int(rows)
        if pre_value is not None:
            self.carry.pre.value = pre_value
            self.carry.pre.row_origin = local_origin
            self.carry.pre.rows = int(rows)
        self.encoder_final_h = None
        self.encoder_final_pre = None
        self.active_chunk_views.clear()

    def prepare_decoder_suffix(self, command: SweepCommand) -> None:
        if command.layer is None:
            raise ArenaTransactionError("decoder suffix prepare requires a layer")
        if not self._allocation_survives_deferred("decoder_suffix_rows"):
            raise ArenaTransactionError("decoder suffix allocation is not available/survivable")
        role = self.decoder_prepare_role(command)
        view = ArenaView(role, self.carry.current.role, command.offset, command.rows, command.index, command.layer)
        self.suffix_views.append(view)
        if role == "decoder_local_window_prepare":
            self.decoder_prepared_by_layer[int(command.layer)] = view

    def decoder_prepare_role(self, command: SweepCommand) -> str:
        if command.layer == 20 and command.offset == 0 and command.rows == self.plan.count:
            return "decoder_full_source_publish"
        if command.rows == 127:
            return "decoder_local_window_prepare"
        raise ArenaTransactionError(f"invalid decoder_prepare_suffix shape layer={command.layer} offset={command.offset} rows={command.rows}")

    def require_decoder_prepared(self, command: SweepCommand) -> None:
        if command.phase is not SweepPhase.DECODER_SUFFIX or command.layer is None:
            return
        view = self.decoder_prepared_by_layer.get(int(command.layer))
        if view is None:
            raise ArenaTransactionError(f"decoder suffix layer {command.layer} executed without prepare")
        first_query = view.offset + view.rows
        if command.offset < first_query:
            raise ArenaTransactionError(f"decoder suffix layer {command.layer} rows {command.offset}:{command.offset + command.rows} precede prepared query cone {first_query}")

    def _allocation_survives_deferred(self, role: str) -> bool:
        state = self.allocations.get(role)
        return bool(state and state.allocation.survives_deferred_decoder)


def _slot_for_role(allocations: dict[str, AllocationState], role: str, *, fallback_value: Any = None) -> TensorSlot:
    state = allocations.get(role)
    if state is None:
        return TensorSlot(role, TensorOwnership.EXTERNAL_HANDLE if fallback_value is not None else TensorOwnership.UNBOUND, fallback_value)
    state.slot.value = fallback_value
    state.slot.ownership = TensorOwnership.OWNED if fallback_value is not None else TensorOwnership.UNBOUND
    return state.slot


def _optional_slot_for_role(allocations: dict[str, AllocationState], role: str) -> TensorSlot | None:
    state = allocations.get(role)
    return None if state is None else state.slot
