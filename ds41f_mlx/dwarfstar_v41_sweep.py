"""Planner contract for DwarfStar DeepSeek-V4.1 ds41_graph_prefill_sweep.

No Metal/data-plane work lives here.  This module transcribes the control-flow
shape of antirez/ds4 @ 0aaea5a... so the native planner can be reconciled before
kernel work resumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DS4_AUTHORITY_SHA = "0aaea5a238fb41a35106a551e73c8409dfb751ac"
DS4_AUTHORITY_REMOTE = "https://github.com/antirez/ds4.git"
DS41_ENCODER_LAYERS = 20
DS41_N_LAYERS = 40
DS41_PREFILL_CAP = 8192
DS41_DECODER_WINDOW = 127


@dataclass(frozen=True)
class CarryRow:
    name: str
    width_expr: str
    format: str
    compact: bool


DS41_CARRY_ROWS = (
    CarryRow("residual", "hc_mult * dim", "BF16", True),
    CarryRow("pre", "hc_mult", "F32", False),
    CarryRow("ffn_split", "24", "F32", False),
    CarryRow("selected_comp", "index_topk", "F32", False),
    CarryRow("block_mask", "ceil(ctx / 8)", "MASK", True),
)


@dataclass(frozen=True)
class SweepLayerPlan:
    layer: int
    phase: str
    first_row: int
    rows: int
    chunk: int
    chunks: list[dict[str, int]]
    publishes_encoder_keys: bool = False
    decoder_prepare_rows: list[dict[str, int | bool]] = field(default_factory=list)


@dataclass(frozen=True)
class SweepPlan:
    ctx: int
    remaining: int
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
    carry_rows: tuple[CarryRow, ...]
    layers: list[SweepLayerPlan]
    prefetch_events: list[dict[str, Any]]
    source_authority: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "ctx": self.ctx,
            "remaining": self.remaining,
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
            "carry_rows": [c.__dict__ for c in self.carry_rows],
            "layers": [
                {
                    "layer": l.layer,
                    "phase": l.phase,
                    "first_row": l.first_row,
                    "rows": l.rows,
                    "chunk": l.chunk,
                    "chunks": l.chunks,
                    "publishes_encoder_keys": l.publishes_encoder_keys,
                    "decoder_prepare_rows": l.decoder_prepare_rows,
                }
                for l in self.layers
            ],
            "prefetch_events": self.prefetch_events,
            "source_authority": self.source_authority,
        }


def ds41_prefill_limit(ctx: int) -> int:
    if ctx < 8192:
        limit = 2048
    elif ctx < 16384:
        limit = 4096
    else:
        limit = DS41_PREFILL_CAP
    return min(ctx, limit)


def ds41_encoder_chunk_cap(prefill_cap: int, total_count: int) -> int:
    if total_count < 8192 and prefill_cap > 2048:
        return 2048
    if total_count < 16384 and prefill_cap > 4096:
        return 4096
    return prefill_cap


def ds41_prefill_count(ctx: int, remaining: int, carry_cap: int | None = None) -> int:
    prefill_cap = ds41_prefill_limit(ctx)
    cap = carry_cap if carry_cap is not None else remaining
    if cap and remaining >= 4096:
        count = min(remaining, cap)
        return count - count % 2048
    tail_cap = min(prefill_cap, 2048)
    return min(remaining, tail_cap)


def _chunks(first: int, rows: int, chunk: int) -> list[dict[str, int]]:
    out = []
    end = first + rows
    off = first
    while off < end:
        n = min(chunk, end - off)
        out.append({"offset": off, "rows": n})
        off += n
    return out


def build_ds41_sweep_plan(
    *,
    ctx: int,
    remaining: int,
    encoder_resident: bool = False,
    encoder_only: bool = False,
    resume_encoder: bool = False,
    carry_cap: int | None = None,
) -> SweepPlan:
    prefill_cap = ds41_prefill_limit(ctx)
    count = ds41_prefill_count(ctx, remaining, carry_cap=carry_cap)
    encoder_chunk = ds41_encoder_chunk_cap(prefill_cap, count)
    wide = count > encoder_chunk
    decoder_suffix = wide and count >= 8192
    defer_decoder = (not encoder_resident) and count >= 16384 and remaining - count >= 8192
    if (encoder_only or resume_encoder) and not decoder_suffix:
        raise ValueError("DwarfStar rejects encoder_only/resume_encoder without decoder_suffix")

    layers: list[SweepLayerPlan] = []
    for il in range(DS41_N_LAYERS):
        if encoder_only and il == DS41_ENCODER_LAYERS:
            layers.append(
                SweepLayerPlan(
                    layer=il,
                    phase="encoder_publish_stop",
                    first_row=0,
                    rows=count,
                    chunk=encoder_chunk,
                    chunks=_chunks(0, count, encoder_chunk),
                    publishes_encoder_keys=True,
                    decoder_prepare_rows=[{"offset": 0, "rows": count, "publish": True}],
                )
            )
            break
        first = 0
        rows = count
        prepares: list[dict[str, int | bool]] = []
        phase = "encoder" if il < DS41_ENCODER_LAYERS else "decoder_full"
        if decoder_suffix and il >= DS41_ENCODER_LAYERS:
            if il == DS41_ENCODER_LAYERS:
                prepares.append({"offset": 0, "rows": count, "publish": True})
            needed = 1 + (DS41_N_LAYERS - 1 - il) * DS41_DECODER_WINDOW
            first = count - needed
            rows = needed
            prepares.append({"offset": first - DS41_DECODER_WINDOW, "rows": DS41_DECODER_WINDOW, "publish": False})
            phase = "decoder_suffix"
        chunk = 2048 if (wide and il >= DS41_ENCODER_LAYERS and encoder_chunk > 2048) else encoder_chunk
        layers.append(
            SweepLayerPlan(
                layer=il,
                phase=phase,
                first_row=first,
                rows=rows,
                chunk=chunk,
                chunks=_chunks(first, rows, chunk),
                decoder_prepare_rows=prepares,
            )
        )

    return SweepPlan(
        ctx=ctx,
        remaining=remaining,
        count=count,
        prefill_cap=prefill_cap,
        encoder_chunk=encoder_chunk,
        wide=wide,
        decoder_suffix=decoder_suffix,
        encoder_only=encoder_only,
        resume_encoder=resume_encoder,
        defer_decoder_candidate=defer_decoder,
        checkpoint_valid_during_sweep=False,
        checkpoint_valid_after_sweep=not encoder_only,
        carry_rows=DS41_CARRY_ROWS,
        layers=layers,
        prefetch_events=[
            {"event": "engram_prefetch_start", "table": 0, "before_layer": 0, "rows": count, "condition": "count>=1024 and batch Engram enabled"},
            {"event": "engram_prefetch_start", "table": 1, "before_layer": 2, "rows": count, "condition": "overlap Engram enabled"},
            {"event": "ssd_layer_prepare/read_ahead", "before_layer": "il+1", "condition": "streaming and next layer exists"},
        ],
        source_authority={
            "repository": DS4_AUTHORITY_REMOTE,
            "commit": DS4_AUTHORITY_SHA,
            "functions": [
                "ds4.c: ds41_graph_prefill_sweep",
                "ds4.c: ds41_prefill_limit",
                "ds4.c: ds41_encoder_chunk_cap",
                "ds4.c: ds41_prefill_count",
                "ds4.c: ds41_decoder_prepare",
                "ds4.c: DS41_CARRY_ROWS",
                "ds4.c: session defer_decoder / decoder_pending block",
            ],
        },
    )
