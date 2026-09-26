#!/usr/bin/env python3
"""Tiny M2 loop-inversion oMLX-compatibility fixture.

This is not a benchmark, not a production scheduler, and not official DeepSeek
correctness evidence.  It keeps the existing oMLX DeepSeek-V4.1 layer operations
intact but inverts the tiny fixture loop
from chunk-major:

    chunk 0: layer 0..39
    chunk 1: layer 0..39

to layer-major:

    layer 0: chunk 0..1
    layer 1: chunk 0..1

The goal is only to prove/reject that explicit per-frontier state publication
can carry the current oMLX adapter behavior across more than one chunk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for p in (ROOT, TOOLS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ds41f_mlx.m2_layer_major import layer_major_forward  # noqa: E402
from ds41f_mlx.m2_state_publication import (  # noqa: E402
    ExplicitStatePublication,
    build_publication_frontiers,
    cache_digest,
    records_to_json,
    tensor_digest,
)
from ds41f_mlx.runtime.omlx_core import OmlxRuntime, OmlxRuntimeConfig  # noqa: E402
from run_m2_state_publication_fixture import explicit_replay_forward  # noqa: E402


def greedy_id(logits: Any) -> int:
    import mlx.core as mx  # type: ignore
    import numpy as np

    last = logits[:, -1, :]
    mx.eval(last)
    return int(np.asarray(mx.argmax(last, axis=-1)).reshape(-1)[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="ping\nloop inversion fixture\n")
    ap.add_argument("--target-tokens", type=int, default=16)
    ap.add_argument("--chunk-tokens", type=int, default=8)
    ap.add_argument("--out", default="artifacts/m2/layer-major-correctness/fixture-16tok-2chunk.json")
    ap.add_argument("--final-logits-only", action="store_true", help="Compare only final prompt-position logits; hidden/cache/frontier work is unchanged")
    ap.add_argument("--eval-boundary", choices=("none", "chunk", "layer"), default="none", help="Materialize retained layer-major state at this boundary")
    ap.add_argument("--output-mode", choices=("concat", "full_chunks_last"), default="concat", help="Layer-major logits retention mode")
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"))
    ap.add_argument("--omlx", default=os.environ.get("DS41F_OMLX", str(Path.home() / "omlx-0.7.0.dev2")))
    args = ap.parse_args()

    import mlx.core as mx  # type: ignore

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "schema": "ds41f.m2.layer-major-omlx-compatibility-fixture.v2",
        "purpose": "historical loop-inversion oMLX-compatibility only; not performance; not official qualification",
        "classification": "historical_only_omlx_compatibility_reference_not_official_qualification",
        "checkpoint": args.checkpoint,
        "omlx": args.omlx,
        "target_tokens": args.target_tokens,
        "chunk_tokens": args.chunk_tokens,
        "final_logits_only": args.final_logits_only,
        "eval_boundary": args.eval_boundary,
        "output_mode": args.output_mode,
    }
    t0 = time.perf_counter()
    rt = OmlxRuntime(OmlxRuntimeConfig(omlx_path=Path(args.omlx), checkpoint_path=Path(args.checkpoint), engram_ssd_offload=True, preserve_mtp=True))
    try:
        model, processor = rt.load_model()
        lm = model.language_model
        record["load_seconds"] = time.perf_counter() - t0
        tokenizer = processor.tokenizer
        text = args.prompt
        ids = tokenizer.encode(text, add_special_tokens=False)
        while len(ids) < args.target_tokens:
            text += args.prompt
            ids = tokenizer.encode(text, add_special_tokens=False)
        ids = ids[: args.target_tokens]
        chunks = [ids[i : i + args.chunk_tokens] for i in range(0, len(ids), args.chunk_tokens)]
        if len(chunks) < 2:
            raise ValueError("fixture requires at least two chunks")
        record["input_token_count"] = len(ids)
        record["chunk_lengths"] = [len(c) for c in chunks]
        record["frontiers"] = {str(k): list(v.publishes) for k, v in build_publication_frontiers(lm._config).items()}

        accepted_cache = lm.make_cache()
        chunk_replay_cache = lm.make_cache()
        layer_major_cache = lm.make_cache()
        accepted_parts = []
        chunk_replay_parts = []
        chunk_replay_records = []
        for chunk in chunks:
            x = mx.array(chunk)[None]
            accepted = lm._forward(x, cache=accepted_cache)
            replay, replay_records = explicit_replay_forward(lm, x, chunk_replay_cache, ExplicitStatePublication)
            mx.eval(accepted, replay)
            accepted_parts.append(accepted)
            chunk_replay_parts.append(replay)
            chunk_replay_records.append(records_to_json(replay_records))
        if args.final_logits_only:
            accepted_logits = accepted_parts[-1][:, -1:, :]
            chunk_replay_logits = chunk_replay_parts[-1][:, -1:, :]
        elif args.output_mode == "full_chunks_last":
            accepted_logits = accepted_parts[-1]
            chunk_replay_logits = chunk_replay_parts[-1]
        else:
            accepted_logits = mx.concatenate(accepted_parts, axis=1)
            chunk_replay_logits = mx.concatenate(chunk_replay_parts, axis=1)
        layer_major_logits, layer_major_records = layer_major_forward(
            lm,
            chunks,
            layer_major_cache,
            final_logits_only=args.final_logits_only,
            eval_boundary=args.eval_boundary,
            output_mode=args.output_mode,
        )
        mx.eval(accepted_logits, chunk_replay_logits, layer_major_logits)
        mx.synchronize()

        record["greedy"] = {
            "accepted": greedy_id(accepted_logits),
            "chunk_replay": greedy_id(chunk_replay_logits),
            "layer_major": greedy_id(layer_major_logits),
        }
        record["logits_digest"] = {
            "accepted": tensor_digest(accepted_logits),
            "chunk_replay": tensor_digest(chunk_replay_logits),
            "layer_major": tensor_digest(layer_major_logits),
        }
        record["final_cache_digest"] = {
            "accepted": [cache_digest(c) for c in accepted_cache],
            "chunk_replay": [cache_digest(c) for c in chunk_replay_cache],
            "layer_major": [cache_digest(c) for c in layer_major_cache],
        }
        record["publication_records"] = {
            "chunk_major_explicit": chunk_replay_records,
            "layer_major_explicit": layer_major_records,
        }
        record["gates"] = {
            "two_chunks_exercised": len(chunks) >= 2,
            "chunk_replay_logits_exact": record["logits_digest"]["chunk_replay"] == record["logits_digest"]["accepted"],
            "layer_major_logits_exact": record["logits_digest"]["layer_major"] == record["logits_digest"]["accepted"],
            "layer_major_token_exact": record["greedy"]["layer_major"] == record["greedy"]["accepted"],
            "chunk_replay_cache_exact": record["final_cache_digest"]["chunk_replay"] == record["final_cache_digest"]["accepted"],
            "layer_major_cache_exact": record["final_cache_digest"]["layer_major"] == record["final_cache_digest"]["accepted"],
            "publication_frontiers_exact": layer_major_records == chunk_replay_records,
        }
        record["ok"] = all(record["gates"].values())
    except BaseException as exc:
        record.update(ok=False, error_type=type(exc).__name__, error=str(exc))
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        raise
    finally:
        rt.close()
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
