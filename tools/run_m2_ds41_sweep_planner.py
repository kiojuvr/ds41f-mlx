#!/usr/bin/env python3
"""Record a DwarfStar ds41_graph_prefill_sweep planner contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ds41f_mlx.dwarfstar_v41_sweep import build_ds41_sweep_plan  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--remaining", type=int, default=8192)
    ap.add_argument("--encoder-resident", action="store_true")
    ap.add_argument("--encoder-only", action="store_true")
    ap.add_argument("--resume-encoder", action="store_true")
    ap.add_argument("--out", default="artifacts/m2/dwarfstar-prefill/ds41-sweep-plan-8k.json")
    args = ap.parse_args()

    plan = build_ds41_sweep_plan(
        ctx=args.ctx,
        remaining=args.remaining,
        encoder_resident=args.encoder_resident,
        encoder_only=args.encoder_only,
        resume_encoder=args.resume_encoder,
    )
    rows_by_phase: dict[str, int] = {}
    chunks_by_phase: dict[str, int] = {}
    for layer in plan.layers:
        rows_by_phase[layer.phase] = rows_by_phase.get(layer.phase, 0) + layer.rows
        chunks_by_phase[layer.phase] = chunks_by_phase.get(layer.phase, 0) + len(layer.chunks)
    decoder_layers = [l for l in plan.layers if l.phase.startswith("decoder")]
    record = {
        "schema": "ds41f.m2.dwarfstar-ds41-sweep-plan.v1",
        "purpose": "planner reconciliation contract for ds41_graph_prefill_sweep; no Metal/data-plane work",
        "plan": plan.to_json(),
        "summary": {
            "layer_count_planned": len(plan.layers),
            "rows_by_phase": rows_by_phase,
            "chunks_by_phase": chunks_by_phase,
            "decoder_suffix_rows": sum(l.rows for l in decoder_layers),
            "total_row_layer_work": sum(l.rows for l in plan.layers),
        },
        "gates": {
            "authority_pinned": bool(plan.source_authority.get("commit")),
            "checkpoint_invalid_during_sweep": plan.checkpoint_valid_during_sweep is False,
            "structured_carry_rows_recorded": len(plan.carry_rows) == 5,
            "wide_8k_uses_decoder_suffix": (args.remaining < 8192) or plan.decoder_suffix,
            "decoder_suffix_shrinks_when_wide": (not plan.decoder_suffix) or (decoder_layers[-1].rows == 1),
            "engram_prefetch_order_recorded": len(plan.prefetch_events) >= 2,
        },
    }
    record["ok"] = all(record["gates"].values())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
