#!/usr/bin/env python3
"""Qualification-only MLX 0.32.2 structural graph probes for P8.

These probes do not model full ds41f performance and do not implement any
replacement path.  They only measure whether repeated MLX slice assignment and
concat-then-update structures defer increasing materialization work.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx


def mem() -> dict[str, int | None]:
    out = {}
    for name in ("get_active_memory", "get_cache_memory", "get_peak_memory"):
        f = getattr(mx, name, None)
        try:
            out[name.replace("get_", "")] = int(f()) if f is not None else None
        except Exception:
            out[name.replace("get_", "")] = None
    return out


def timed(fn):
    b = mem(); t0 = time.perf_counter(); result = fn(); elapsed = time.perf_counter() - t0; a = mem()
    return result, elapsed, b, a


def slice_update_probe(writes: int, *, cols: int, dim: int, rows_per_write: int) -> dict[str, Any]:
    base = mx.zeros((1, cols, dim), dtype=mx.bfloat16)
    update = mx.ones((1, rows_per_write, dim), dtype=mx.bfloat16)
    def build():
        nonlocal base
        for i in range(writes):
            start = (i * rows_per_write) % (cols - rows_per_write + 1)
            base[:, start:start + rows_per_write] = update
        return base
    _, build_s, b0, b1 = timed(build)
    _, eval_s, e0, e1 = timed(lambda: mx.eval(base))
    return {"writes": writes, "graph_build_wall_s": build_s, "final_eval_wall_s": eval_s, "memory_before_build": b0, "memory_after_build": b1, "memory_before_eval": e0, "memory_after_eval": e1, "shape": [1, cols, dim], "rows_per_write": rows_per_write}


def concat_update_probe(inputs: int, writes: int, *, tile_rows: int, dim: int) -> dict[str, Any]:
    parts = [mx.ones((1, tile_rows, dim), dtype=mx.bfloat16) * (i + 1) for i in range(inputs)]
    base = mx.zeros((1, tile_rows * inputs, dim), dtype=mx.bfloat16)
    def build():
        nonlocal base
        cat = mx.concatenate(parts, axis=1)
        for i in range(writes):
            base[:, 0:tile_rows * inputs] = cat
        return base
    _, build_s, b0, b1 = timed(build)
    _, eval_s, e0, e1 = timed(lambda: mx.eval(base))
    return {"concat_inputs": inputs, "writes_after_concat": writes, "graph_build_wall_s": build_s, "final_eval_wall_s": eval_s, "memory_before_build": b0, "memory_after_build": b1, "memory_before_eval": e0, "memory_after_eval": e1, "tile_rows": tile_rows, "dim": dim}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("artifacts/p8/synthetic_mlx_graph_probes.json"))
    ap.add_argument("--cols", type=int, default=4096)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--rows-per-write", type=int, default=128)
    args = ap.parse_args()
    rec = {
        "schema": "ds41f.p8.synthetic_mlx_graph_probes.v1",
        "qualification_only": True,
        "mlx_version": "0.32.2",
        "slice_assignment_semantics": "Python base[:, start:end] = update uses MLX lazy slice_update descriptor replacement in pinned source; this probe measures structural pressure only.",
        "slice_update_scaling": [slice_update_probe(w, cols=args.cols, dim=args.dim, rows_per_write=args.rows_per_write) for w in (1, 2, 4, 8, 16, 32)],
        "concat_update": [concat_update_probe(n, w, tile_rows=args.rows_per_write, dim=args.dim) for n in (2, 4) for w in (1, 4, 16)],
        "non_claims": ["not full-model performance", "not an optimization candidate", "does not change production selector"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=2))
    print(json.dumps({"out": str(args.out), "slice_cases": len(rec["slice_update_scaling"]), "concat_cases": len(rec["concat_update"])}))


if __name__ == "__main__":
    main()
