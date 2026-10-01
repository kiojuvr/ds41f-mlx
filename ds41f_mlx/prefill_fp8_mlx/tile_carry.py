"""Tile-native transient carry transport for P8 A/B qualification.

This module changes only the physical request-local h/pre transport between
DwarfStar commands.  It does not own cache/publication/model state and does not
materialize tensors merely to describe spans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence
import os

from ds41f_mlx.prefill_fp8_mlx.planner import SweepCommand, SweepCommandKind, SweepPhase, SweepPlan


class TileCarryError(RuntimeError):
    """Invalid tile-native carry operation or unsupported geometry."""


@dataclass(frozen=True)
class TileSpan:
    value: Any
    logical_offset: int
    rows: int
    role: str
    row_origin: int = 0
    absolute_start: int | None = None

    @property
    def end(self) -> int:
        return self.logical_offset + self.rows

    def slice(self, offset: int, rows: int) -> Any:
        rel = int(offset) - int(self.logical_offset)
        if rel < 0 or rel + int(rows) > self.rows:
            raise TileCarryError(f"span {self.role} {self.logical_offset}:{self.end} cannot serve {offset}:{offset + rows}")
        if rel == 0 and int(rows) == self.rows:
            return self.value
        try:
            return self.value[:, rel:rel + int(rows)]
        except Exception as exc:
            raise TileCarryError(f"failed to slice tile {self.role} rows {rel}:{rel + int(rows)}") from exc

    def to_json(self) -> dict[str, object]:
        return {
            "logical_offset": self.logical_offset,
            "rows": self.rows,
            "role": self.role,
            "row_origin": self.row_origin,
            "absolute_start": self.absolute_start,
            "shape": _shape(self.value),
            "dtype": _dtype(self.value),
        }


@dataclass
class TileCarryTelemetry:
    tile_input_bindings: int = 0
    tile_output_bindings: int = 0
    dense_carry_writes: int = 0
    dense_layer_transport_slices: int = 0
    logical_swaps: int = 0
    full_source_assemblies: int = 0
    final_cone_tile_slices: int = 0
    final_cone_minimal_concats: int = 0
    initial_tile_views: int = 0

    def to_json(self) -> dict[str, int]:
        return self.__dict__.copy()


@dataclass
class TileCarryState:
    """Tile collections for transient h/pre transport.

    `current_h` and `pre` contain exact command-span inputs for the current
    layer.  `next_h`/`next_pre` are populated by Block outputs, then become the
    current maps at the DwarfStar swap command.  `next_h` after swap retains the
    previous current map, preserving the ping-pong spare semantics needed by P6
    compact next-cone extraction.
    """

    count: int
    current_h: dict[int, TileSpan]
    pre: dict[int, TileSpan]
    next_h: dict[int, TileSpan] = field(default_factory=dict)
    next_pre: dict[int, TileSpan] = field(default_factory=dict)
    telemetry: TileCarryTelemetry = field(default_factory=TileCarryTelemetry)
    admitted: bool = True
    retired: bool = False
    mode: str = "tile_native_carry"
    source_phase_complete: bool = False
    full_source_h: Any = None
    full_source_pre: Any = None

    @classmethod
    def from_dense_initial(cls, *, plan: SweepPlan, h_current: Any, pre: Any, base_frontier: int = 0) -> "TileCarryState":
        spans = source_command_spans(plan)
        if not spans:
            raise TileCarryError("tile-native carry admission requires source ENCODE_ROWS spans")
        h_tiles: dict[int, TileSpan] = {}
        pre_tiles: dict[int, TileSpan] = {}
        tel = TileCarryTelemetry()
        for off, rows in spans:
            h_value = _slice_rows(h_current, off, rows, role="tile.initial_h")
            pre_value = _slice_rows(pre, off, rows, role="tile.initial_pre")
            tel.initial_tile_views += 2
            h_tiles[off] = TileSpan(h_value, off, rows, "tile.current_h", absolute_start=int(base_frontier) + off)
            pre_tiles[off] = TileSpan(pre_value, off, rows, "tile.pre", absolute_start=int(base_frontier) + off)
        return cls(count=int(plan.count), current_h=h_tiles, pre=pre_tiles, telemetry=tel)

    def require_not_retired(self) -> None:
        if self.retired:
            raise TileCarryError("tile carry has been retired")

    def input_for(self, command: SweepCommand, *, base_frontier: int) -> tuple[Any, Any, int]:
        self.require_not_retired()
        h = self._exact(self.current_h, command.offset, command.rows, "current_h")
        pre = self._exact(self.pre, command.offset, command.rows, "pre")
        absolute_start = int(base_frontier) + int(command.offset)
        if h.absolute_start is not None and int(h.absolute_start) != absolute_start:
            raise TileCarryError("tile absolute_start mismatch")
        self.telemetry.tile_input_bindings += 2
        return h.value, pre.value, absolute_start

    def bind_output(self, command: SweepCommand, h_out: Any, pre_out: Any, *, base_frontier: int) -> None:
        self.require_not_retired()
        rows = int(command.rows)
        self._validate_rows(h_out, rows, "h_out")
        self._validate_rows(pre_out, rows, "pre_out")
        off = int(command.offset)
        absolute_start = int(base_frontier) + off
        self.next_h[off] = TileSpan(h_out, off, rows, "tile.next_h", absolute_start=absolute_start)
        self.next_pre[off] = TileSpan(pre_out, off, rows, "tile.next_pre", absolute_start=absolute_start)
        self.telemetry.tile_output_bindings += 2

    def logical_swap(self) -> None:
        self.require_not_retired()
        if not self.next_h or not self.next_pre:
            raise TileCarryError("cannot swap tile carry before next tiles are complete")
        _validate_cover(self.next_h.values(), self.count, "next_h")
        _validate_cover(self.next_pre.values(), self.count, "next_pre")
        old_current = self.current_h
        self.current_h = dict(self.next_h)
        self.pre = dict(self.next_pre)
        self.next_h = dict(old_current)
        self.next_pre = {}
        self.telemetry.logical_swaps += 1

    def assemble_full_source(self, *, mx: Any | None = None) -> tuple[Any, Any]:
        self.require_not_retired()
        h = self._assemble(self.current_h, "encoder_final_h", mx=mx)
        pre = self._assemble(self.pre, "encoder_final_pre", mx=mx)
        self.full_source_h = h
        self.full_source_pre = pre
        self.telemetry.full_source_assemblies += 1
        return h, pre

    def cone_value(self, which: str, offset: int, rows: int, *, mx: Any | None = None) -> Any:
        self.require_not_retired()
        maps = {"current": self.current_h, "next": self.next_h, "pre": self.pre}
        if which not in maps:
            raise TileCarryError(f"unknown cone source {which!r}")
        pieces: list[Any] = []
        pos = int(offset)
        end = pos + int(rows)
        for span in sorted(maps[which].values(), key=lambda s: s.logical_offset):
            if span.end <= pos:
                continue
            if span.logical_offset >= end:
                break
            left = max(pos, span.logical_offset)
            right = min(end, span.end)
            pieces.append(span.slice(left, right - left))
            self.telemetry.final_cone_tile_slices += 1
            pos = right
            if pos >= end:
                break
        if pos < end:
            raise TileCarryError(f"{which} cone has gap at {pos}:{end}")
        if len(pieces) == 1:
            return pieces[0]
        self.telemetry.final_cone_minimal_concats += 1
        return _concat(pieces, mx=mx, role=f"{which}_cone")

    def retire(self) -> None:
        self.current_h.clear(); self.next_h.clear(); self.pre.clear(); self.next_pre.clear()
        self.full_source_h = None; self.full_source_pre = None
        self.retired = True

    def _exact(self, tiles: dict[int, TileSpan], offset: int, rows: int, role: str) -> TileSpan:
        span = tiles.get(int(offset))
        if span is None or int(span.rows) != int(rows):
            raise TileCarryError(f"missing exact {role} tile for {offset}:{offset + rows}")
        return span

    def _assemble(self, tiles: dict[int, TileSpan], role: str, *, mx: Any | None) -> Any:
        ordered = sorted(tiles.values(), key=lambda s: s.logical_offset)
        _validate_cover(ordered, self.count, role)
        if len(ordered) == 1:
            return ordered[0].value
        return _concat([s.value for s in ordered], mx=mx, role=role)

    def _validate_rows(self, value: Any, rows: int, role: str) -> None:
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) > 1 and int(shape[1]) != int(rows):
            raise TileCarryError(f"{role} rows {shape[1]} != expected {rows}")

    def to_json(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "count": self.count,
            "retired": self.retired,
            "current_h_offsets": sorted(self.current_h),
            "next_h_offsets": sorted(self.next_h),
            "pre_offsets": sorted(self.pre),
            "telemetry": self.telemetry.to_json(),
            "full_source_h_bound": self.full_source_h is not None,
            "full_source_pre_bound": self.full_source_pre is not None,
        }


def tile_native_enabled() -> bool:
    return os.environ.get("DS41F_P8_TILE_NATIVE_CARRY", "").strip() in {"1", "true", "TRUE", "yes", "on"}


def source_command_spans(plan: SweepPlan) -> tuple[tuple[int, int], ...]:
    by_offset: dict[int, int] = {}
    for c in plan.commands:
        if c.kind is SweepCommandKind.ENCODE_ROWS and c.phase in {SweepPhase.ENCODER, SweepPhase.DECODER_FULL}:
            by_offset.setdefault(int(c.offset), int(c.rows))
            if by_offset[int(c.offset)] != int(c.rows):
                raise TileCarryError("inconsistent source command span geometry")
    return tuple(sorted(by_offset.items()))


def admit_tile_native(plan: SweepPlan) -> tuple[bool, str]:
    spans = source_command_spans(plan)
    if not spans:
        return False, "no source command spans"
    try:
        _validate_cover([TileSpan(None, o, r, "admission") for o, r in spans], int(plan.count), "admission")
    except Exception as exc:
        return False, str(exc)
    if not any(c.kind is SweepCommandKind.P6_SOURCE_COMPLETE_AND_DETACH_CONE for c in plan.commands):
        return False, "initial implementation admits P6 source lifecycle only"
    return True, "admitted"


def _validate_cover(spans: Iterable[TileSpan], count: int, role: str) -> None:
    pos = 0
    for span in sorted(spans, key=lambda s: s.logical_offset):
        if int(span.logical_offset) != pos:
            raise TileCarryError(f"{role} tile gap/overlap at {pos}, got {span.logical_offset}")
        if int(span.rows) <= 0:
            raise TileCarryError(f"{role} tile has non-positive rows")
        pos += int(span.rows)
    if pos != int(count):
        raise TileCarryError(f"{role} tiles cover {pos}, expected {count}")


def _slice_rows(value: Any, offset: int, rows: int, *, role: str) -> Any:
    if value is None:
        raise TileCarryError(f"cannot slice unbound {role}")
    try:
        return value[:, int(offset):int(offset) + int(rows)]
    except Exception as exc:
        raise TileCarryError(f"failed initial tile slice for {role} {offset}:{offset + rows}") from exc


def _concat(values: Sequence[Any], *, mx: Any | None, role: str) -> Any:
    if not values:
        raise TileCarryError(f"cannot concatenate empty values for {role}")
    first = values[0]
    concat = getattr(first, "concat_rows", None)
    if concat is not None:
        return concat(list(values))
    if mx is None:
        try:
            import mlx.core as mx  # type: ignore
        except Exception as exc:
            raise TileCarryError(f"MLX concatenate required for {role}") from exc
    try:
        return mx.concatenate(list(values), axis=1)
    except Exception as exc:
        raise TileCarryError(f"failed to concatenate {role}") from exc


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    return None if shape is None else tuple(int(x) for x in shape)


def _dtype(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    return None if dtype is None else str(dtype)
