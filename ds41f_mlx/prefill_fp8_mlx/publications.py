"""Transaction-local publication manager for DwarfStar FP8/MLX prefill.

The manager is connected to P3 block execution: oMLX blocks receive a shared
state dictionary whose visibility is controlled by DwarfStar publication
frontiers.  It does not own a second persistent tensor store; values are opaque
references produced by block/cache operations and tracked through the request
arena's transaction lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ds41f_mlx.prefill_fp8_mlx.arena import RequestArena
from ds41f_mlx.prefill_fp8_mlx.planner import SweepCommand, SweepCommandKind

V41_PUBLISHES_BY_LAYER: dict[int, tuple[str, ...]] = {
    1: ("engram",),
    2: ("kv", "index_k", "idx"),
    8: ("kv", "index_k", "idx"),
    14: ("engram", "kv", "index_k", "idx"),
    20: ("kv", "index_k", "idx", "candidates"),
    24: ("index_k", "idx"),
    28: ("index_k", "idx"),
    32: ("index_k", "idx"),
    36: ("index_k", "idx"),
}

SHARED_KEYS = ("kv", "index_k", "idx", "candidates", "topk", "engram")


class PublicationError(RuntimeError):
    """Invalid publication/consumer visibility transition."""


@dataclass(frozen=True)
class SourceGeneration:
    key: str
    layer: int
    value: Any
    command_index: int
    published: bool = False
    committed: bool = False

    def visible_copy(self, *, published: bool | None = None, committed: bool | None = None) -> "SourceGeneration":
        return SourceGeneration(
            key=self.key,
            layer=self.layer,
            value=self.value,
            command_index=self.command_index,
            published=self.published if published is None else published,
            committed=self.committed if committed is None else committed,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "key": self.key,
            "layer": self.layer,
            "command_index": self.command_index,
            "published": self.published,
            "committed": self.committed,
            "has_value": self.value is not None,
        }


@dataclass
class PublicationManager:
    """DwarfStar source/consumer visibility controller."""

    arena: RequestArena
    pending_by_layer: dict[int, dict[str, SourceGeneration]] = field(default_factory=dict)
    transaction_visible: dict[str, SourceGeneration] = field(default_factory=dict)
    committed_visible: dict[str, SourceGeneration] = field(default_factory=dict)
    failed: bool = False

    def begin_transaction(self) -> None:
        self.pending_by_layer.clear()
        self.transaction_visible.clear()
        self.failed = False

    def shared_for_layer(self, layer: int) -> dict[str, Any]:
        shared = {key: gen.value for key, gen in self.transaction_visible.items() if key in SHARED_KEYS}
        # Ensure oMLX blocks may write expected keys without seeing unpublished generations.
        for key in SHARED_KEYS:
            shared.setdefault(key, None)
        return shared

    def capture_layer_outputs(self, layer: int, shared_after: dict[str, Any], *, command_index: int) -> None:
        keys = V41_PUBLISHES_BY_LAYER.get(int(layer), ())
        if not keys:
            return
        bucket = self.pending_by_layer.setdefault(int(layer), {})
        for key in keys:
            bucket[key] = SourceGeneration(key=key, layer=int(layer), value=shared_after.get(key), command_index=command_index)

    def publish_frontier(self, command: SweepCommand) -> None:
        if command.kind is not SweepCommandKind.PUBLISH_FRONTIER or command.layer is None:
            raise PublicationError("publish_frontier requires a layer publication command")
        layer = int(command.layer)
        for key, gen in self.pending_by_layer.pop(layer, {}).items():
            published = gen.visible_copy(published=True, committed=False)
            self.transaction_visible[key] = published
            self.arena.publications.shared[key] = published.value

    def consume(self, key: str, *, layer: int) -> Any:
        if key not in self.transaction_visible:
            raise PublicationError(f"layer {layer} attempted to consume unpublished source {key!r}")
        return self.transaction_visible[key].value

    def commit(self) -> None:
        if self.failed:
            raise PublicationError("cannot commit failed publication transaction")
        self.committed_visible = {key: gen.visible_copy(committed=True) for key, gen in self.transaction_visible.items()}
        self.arena.publications.commit()

    def fail(self) -> None:
        self.failed = True
        self.pending_by_layer.clear()
        self.transaction_visible.clear()
        # Deliberately leave committed_visible untouched; pending invalid sweep is not exposed.

    def to_json(self) -> dict[str, object]:
        return {
            "pending_by_layer": {str(layer): {k: v.to_json() for k, v in gens.items()} for layer, gens in self.pending_by_layer.items()},
            "transaction_visible": {k: v.to_json() for k, v in self.transaction_visible.items()},
            "committed_visible": {k: v.to_json() for k, v in self.committed_visible.items()},
            "failed": self.failed,
        }
