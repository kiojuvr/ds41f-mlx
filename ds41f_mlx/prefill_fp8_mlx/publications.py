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

EXPECTED_V41_PUBLISHES_BY_LAYER: dict[int, tuple[str, ...]] = {
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

CUMULATIVE_KEYS = frozenset({"kv", "index_k", "engram"})
ROW_SPAN_KEYS = frozenset({"idx", "candidates"})
SHARED_KEYS = tuple(sorted(CUMULATIVE_KEYS | ROW_SPAN_KEYS | {"topk"}))


class PublicationError(RuntimeError):
    """Invalid publication/consumer visibility transition."""


@dataclass(frozen=True)
class SourceGeneration:
    key: str
    layer: int
    value: Any
    command_index: int
    offset: int | None = None
    rows: int | None = None
    published: bool = False
    committed: bool = False

    @property
    def end(self) -> int | None:
        return None if self.offset is None or self.rows is None else self.offset + self.rows

    def visible_copy(self, *, published: bool | None = None, committed: bool | None = None) -> "SourceGeneration":
        return SourceGeneration(
            key=self.key,
            layer=self.layer,
            value=self.value,
            command_index=self.command_index,
            offset=self.offset,
            rows=self.rows,
            published=self.published if published is None else published,
            committed=self.committed if committed is None else committed,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "key": self.key,
            "layer": self.layer,
            "command_index": self.command_index,
            "offset": self.offset,
            "rows": self.rows,
            "published": self.published,
            "committed": self.committed,
            "has_value": self.value is not None,
        }


@dataclass(frozen=True)
class PublicationTopology:
    publishes_by_layer: dict[int, tuple[str, ...]]
    consumes_by_layer: dict[int, tuple[str, ...]]

    @classmethod
    def from_model_config(cls, config: Any) -> "PublicationTopology":
        publishes: dict[int, set[str]] = {}
        for layer in tuple(int(x) for x in getattr(config, "engram_layer_ids")):
            publishes.setdefault(layer, set()).add("engram")
        for layer in tuple(int(x) for x in getattr(config, "kv_source_layers")):
            publishes.setdefault(layer, set()).update(("kv", "index_k", "idx"))
        for layer in tuple(int(x) for x in getattr(config, "index_source_layers")):
            publishes.setdefault(layer, set()).update(("index_k", "idx"))
        cand = int(getattr(config, "candidate_source_layer"))
        if cand >= 0:
            publishes.setdefault(cand, set()).add("candidates")
        normalized = {k: tuple(x for x in ("engram", "kv", "index_k", "idx", "candidates") if x in v) for k, v in publishes.items()}
        if normalized != EXPECTED_V41_PUBLISHES_BY_LAYER:
            raise PublicationError(f"model-derived publication topology mismatch: {normalized}")
        consumes: dict[int, tuple[str, ...]] = {}
        compress = getattr(config, "compress_ratios")
        kv_sources = set(int(x) for x in getattr(config, "kv_source_layers"))
        for layer in range(40):
            try:
                ratio = int(compress[layer])
            except Exception:
                ratio = int(compress.get(layer, 0)) if hasattr(compress, "get") else 0
            req: list[str] = []
            if ratio and layer not in kv_sources:
                req.extend(["kv", "idx"])
            if cand >= 0 and layer > cand:
                req.append("candidates")
            consumes[layer] = tuple(req)
        return cls({k: tuple(v) for k, v in normalized.items()}, consumes)

    @classmethod
    def v41_expected(cls) -> "PublicationTopology":
        cfg = type("Cfg", (), {
            "engram_layer_ids": (1, 14),
            "kv_source_layers": (2, 8, 14, 20),
            "index_source_layers": (24, 28, 32, 36),
            "candidate_source_layer": 20,
            "compress_ratios": {i: (4 if i >= 2 else 0) for i in range(40)},
        })()
        return cls.from_model_config(cfg)


@dataclass
class PublicationManager:
    """DwarfStar source/consumer visibility controller.

    Cumulative state (`kv`, `index_k`, Engram) is represented by the latest
    cumulative generation because the oMLX cache/shared value already contains
    preceding rows.  Row-span state (`idx`, `candidates`) is stored by exact row
    coverage so later chunks cannot overwrite earlier query-row-aligned tensors.
    """

    arena: RequestArena
    topology: PublicationTopology = field(default_factory=PublicationTopology.v41_expected)
    pending_cumulative_by_layer: dict[int, dict[str, SourceGeneration]] = field(default_factory=dict)
    pending_spans_by_layer: dict[int, dict[str, list[SourceGeneration]]] = field(default_factory=dict)
    visible_cumulative: dict[str, SourceGeneration] = field(default_factory=dict)
    visible_spans: dict[str, list[SourceGeneration]] = field(default_factory=dict)
    committed_cumulative: dict[str, SourceGeneration] = field(default_factory=dict)
    committed_spans: dict[str, list[SourceGeneration]] = field(default_factory=dict)
    failed: bool = False

    def begin_transaction(self) -> None:
        self.pending_cumulative_by_layer.clear()
        self.pending_spans_by_layer.clear()
        self.visible_cumulative.clear()
        self.visible_spans.clear()
        self.failed = False

    def shared_for_span(self, layer: int, offset: int, rows: int, *, require_keys: tuple[str, ...] = ()) -> dict[str, Any]:
        shared: dict[str, Any] = {key: gen.value for key, gen in self.visible_cumulative.items() if key in CUMULATIVE_KEYS}
        for key in ROW_SPAN_KEYS:
            if key in require_keys or self.visible_spans.get(key):
                shared[key] = self._row_value_for_span(key, offset, rows, consumer_layer=layer)
            else:
                shared[key] = None
        for key in CUMULATIVE_KEYS:
            if key in require_keys and key not in shared:
                raise PublicationError(f"layer {layer} attempted to consume unpublished cumulative source {key!r}")
            shared.setdefault(key, None)
        shared.setdefault("topk", None)
        return shared

    def shared_for_layer(self, layer: int) -> dict[str, Any]:
        return self.shared_for_span(layer, 0, 0)

    def capture_layer_outputs(self, layer: int, shared_after: dict[str, Any], *, command_index: int, offset: int = 0, rows: int = 0) -> None:
        keys = self.topology.publishes_by_layer.get(int(layer), ())
        if not keys:
            return
        for key in keys:
            gen = SourceGeneration(key=key, layer=int(layer), value=shared_after.get(key), command_index=command_index, offset=offset, rows=rows)
            if key in ROW_SPAN_KEYS:
                self.pending_spans_by_layer.setdefault(int(layer), {}).setdefault(key, []).append(gen)
            elif key in CUMULATIVE_KEYS:
                self.pending_cumulative_by_layer.setdefault(int(layer), {})[key] = gen
            else:
                self.pending_cumulative_by_layer.setdefault(int(layer), {})[key] = gen

    def publish_frontier(self, command: SweepCommand) -> None:
        if command.kind is not SweepCommandKind.PUBLISH_FRONTIER or command.layer is None:
            raise PublicationError("publish_frontier requires a layer publication command")
        layer = int(command.layer)
        for key, gen in self.pending_cumulative_by_layer.pop(layer, {}).items():
            published = gen.visible_copy(published=True, committed=False)
            self.visible_cumulative[key] = published
            self.arena.publications.shared[key] = published.value
        for key, spans in self.pending_spans_by_layer.pop(layer, {}).items():
            visible = self.visible_spans.setdefault(key, [])
            for gen in spans:
                published = gen.visible_copy(published=True, committed=False)
                visible.append(published)
            visible.sort(key=lambda g: (g.offset if g.offset is not None else -1, g.rows if g.rows is not None else -1))
            # The arena shared dictionary exposes the latest object for compatibility only;
            # span selection remains authoritative here.
            if visible:
                self.arena.publications.shared[key] = visible[-1].value

    def consume(self, key: str, *, layer: int) -> Any:
        if key in CUMULATIVE_KEYS:
            if key not in self.visible_cumulative:
                raise PublicationError(f"layer {layer} attempted to consume unpublished cumulative source {key!r}")
            return self.visible_cumulative[key].value
        raise PublicationError(f"row-span source {key!r} requires consume_span")

    def consume_span(self, key: str, *, layer: int, offset: int, rows: int) -> Any:
        if key not in ROW_SPAN_KEYS:
            return self.consume(key, layer=layer)
        return self._row_value_for_span(key, offset, rows, consumer_layer=layer)

    def commit(self) -> None:
        if self.failed:
            raise PublicationError("cannot commit failed publication transaction")
        self.committed_cumulative = {key: gen.visible_copy(committed=True) for key, gen in self.visible_cumulative.items()}
        self.committed_spans = {key: [gen.visible_copy(committed=True) for gen in spans] for key, spans in self.visible_spans.items()}
        self.arena.publications.commit()

    def fail(self) -> None:
        self.failed = True
        self.pending_cumulative_by_layer.clear()
        self.pending_spans_by_layer.clear()
        self.visible_cumulative.clear()
        self.visible_spans.clear()
        # Deliberately leave committed_* untouched; pending invalid sweep is not exposed.

    @property
    def transaction_visible(self) -> dict[str, SourceGeneration]:
        """Compatibility view for cumulative latest generations."""
        return self.visible_cumulative

    @property
    def committed_visible(self) -> dict[str, SourceGeneration]:
        """Compatibility view for cumulative committed generations."""
        return self.committed_cumulative

    def _row_value_for_span(self, key: str, offset: int, rows: int, *, consumer_layer: int) -> Any:
        if rows <= 0:
            return None
        spans = self.visible_spans.get(key, [])
        selected: list[SourceGeneration] = []
        pos = int(offset)
        end = int(offset) + int(rows)
        for gen in sorted(spans, key=lambda g: int(g.offset if g.offset is not None else -1)):
            if gen.offset is None or gen.rows is None or gen.end is None:
                continue
            if gen.offset <= pos < gen.end:
                selected.append(gen)
                pos = gen.end
                if pos >= end:
                    break
        if pos < end:
            raise PublicationError(f"layer {consumer_layer} attempted to consume unpublished/incomplete {key!r} span {offset}:{end}")
        if len(selected) == 1 and selected[0].offset == offset and selected[0].rows == rows:
            return selected[0].value
        return _concat_row_values([gen.value for gen in selected], key=key)

    def to_json(self) -> dict[str, object]:
        return {
            "pending_cumulative_by_layer": {str(layer): {k: v.to_json() for k, v in gens.items()} for layer, gens in self.pending_cumulative_by_layer.items()},
            "pending_spans_by_layer": {str(layer): {k: [v.to_json() for v in spans] for k, spans in gens.items()} for layer, gens in self.pending_spans_by_layer.items()},
            "visible_cumulative": {k: v.to_json() for k, v in self.visible_cumulative.items()},
            "visible_spans": {k: [v.to_json() for v in spans] for k, spans in self.visible_spans.items()},
            "committed_cumulative": {k: v.to_json() for k, v in self.committed_cumulative.items()},
            "committed_spans": {k: [v.to_json() for v in spans] for k, spans in self.committed_spans.items()},
            "failed": self.failed,
        }


def _concat_row_values(values: list[Any], *, key: str) -> Any:
    if not values:
        raise PublicationError(f"cannot concatenate empty row-span values for {key}")
    first = values[0]
    concat = getattr(first, "concat_rows", None)
    if concat is not None:
        return concat(values)
    try:
        import mlx.core as mx  # type: ignore
        return mx.concatenate(values, axis=1)
    except Exception as exc:
        if all(isinstance(v, list) for v in values):
            out: list[Any] = []
            for v in values:
                out.extend(v)
            return out
        raise PublicationError(f"cannot concatenate row-span values for {key}; provide MLX tensors or concat_rows") from exc
