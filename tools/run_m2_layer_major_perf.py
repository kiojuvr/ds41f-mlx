#!/usr/bin/env python3
"""Bounded M2 layer-major performance prototype.

This is a single-request, text-only oMLX-compatibility-gated timing harness,
not official DeepSeek qualification.  It uses historical oMLX full-chunk
projection behavior: do not pass ``final_logits_only``.  It skips publication
digest materialization during timed layer-major execution, but keeps the
explicit shared-state object and all oMLX model/cache math.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ds41f_mlx.m2_layer_major import layer_major_forward  # noqa: E402
from ds41f_mlx.m2_state_publication import build_publication_frontiers, cache_digest  # noqa: E402
from ds41f_mlx.runtime.omlx_core import OmlxRuntime, OmlxRuntimeConfig  # noqa: E402


def greedy_id(logits: Any) -> int:
    import mlx.core as mx  # type: ignore
    import numpy as np

    last = logits[:, -1, :]
    mx.eval(last)
    return int(np.asarray(mx.argmax(last, axis=-1)).reshape(-1)[0])


def memory_record() -> dict[str, int | None]:
    try:
        import mlx.core as mx  # type: ignore

        return {
            "active_memory_bytes": int(mx.get_active_memory()),
            "peak_memory_bytes": int(mx.get_peak_memory()),
            "cache_memory_bytes": int(mx.get_cache_memory()),
        }
    except Exception:
        return {"active_memory_bytes": None, "peak_memory_bytes": None, "cache_memory_bytes": None}


def release_mlx_temporaries() -> None:
    import gc
    import mlx.core as mx  # type: ignore

    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def synchronize() -> None:
    import mlx.core as mx  # type: ignore

    mx.synchronize()


def time_chunk_major(lm: Any, chunks: list[list[int]], cache: Any) -> tuple[list[Any], float]:
    import mlx.core as mx  # type: ignore

    parts = []
    t0 = time.perf_counter()
    for chunk in chunks:
        logits = lm._forward(mx.array(chunk)[None], cache=cache)
        mx.eval(logits)
        parts.append(logits)
    synchronize()
    return parts, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="ping\nlayer major bounded perf\n")
    ap.add_argument("--target-tokens", type=int, default=4096)
    ap.add_argument("--chunk-tokens", type=int, default=2048)
    ap.add_argument("--out", default="artifacts/m2/layer-major-perf/perf-4k-full-chunks-last.json")
    ap.add_argument("--eval-boundary", choices=("none", "chunk", "layer"), default="none", help="Materialize retained layer-major state at this boundary to bound MLX graph lifetime")
    ap.add_argument("--output-mode", choices=("concat", "full_chunks_last"), default="full_chunks_last", help="Layer-major logits retention mode; full_chunks_last still projects every chunk at full length")
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"))
    ap.add_argument("--omlx", default=os.environ.get("DS41F_OMLX", str(Path.home() / "omlx-0.7.0.dev2")))
    args = ap.parse_args()

    import mlx.core as mx  # type: ignore

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "schema": "ds41f.m2.layer-major-bounded-perf.v2",
        "purpose": "bounded 4K-ish performance prototype after tiny oMLX-compatibility gates; not production qualification; not official DeepSeek correctness",
        "classification": "performance_omlx_compatibility_only_not_official_qualification",
        "checkpoint": args.checkpoint,
        "omlx": args.omlx,
        "target_tokens": args.target_tokens,
        "chunk_tokens": args.chunk_tokens,
        "projection_semantics": "historical oMLX full-chunk/full-position logits behavior; final-logits-only disabled; output_mode may avoid retaining non-final chunk logits after full projection",
        "publication_digest_materialization_timed": False,
        "accepted_lifetime_isolated_before_layer_major": True,
        "layer_major_eval_boundary": args.eval_boundary,
        "layer_major_output_mode": args.output_mode,
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
            raise ValueError("bounded M2 timing requires at least two chunks")
        record["input_token_count"] = len(ids)
        record["chunk_lengths"] = [len(c) for c in chunks]
        record["frontiers"] = {str(k): list(v.publishes) for k, v in build_publication_frontiers(lm._config).items()}
        record["memory_before"] = memory_record()

        release_mlx_temporaries()
        accepted_cache = lm.make_cache()
        accepted_parts, accepted_seconds = time_chunk_major(lm, chunks, accepted_cache)
        accepted_last = accepted_parts[-1]
        accepted_greedy = greedy_id(accepted_last)
        accepted_cache_digest = [cache_digest(c) for c in accepted_cache]
        record["accepted"] = {
            "seconds": accepted_seconds,
            "tokens_per_second": len(ids) / accepted_seconds,
            "greedy": accepted_greedy,
            "memory_after": memory_record(),
        }
        del accepted_parts, accepted_last, accepted_cache
        release_mlx_temporaries()
        record["memory_after_accepted_release"] = memory_record()

        layer_cache = lm.make_cache()
        t1 = time.perf_counter()
        layer_logits, _records = layer_major_forward(
            lm,
            chunks,
            layer_cache,
            final_logits_only=False,
            record_publications=False,
            eval_boundary=args.eval_boundary,
            output_mode=args.output_mode,
        )
        mx.eval(layer_logits)
        synchronize()
        layer_seconds = time.perf_counter() - t1
        record["layer_major"] = {
            "seconds": layer_seconds,
            "tokens_per_second": len(ids) / layer_seconds,
            "greedy": greedy_id(layer_logits),
            "memory_after": memory_record(),
        }
        record["speedup_vs_accepted"] = accepted_seconds / layer_seconds if layer_seconds else None
        record["gates"] = {
            "two_or_more_chunks": len(chunks) >= 2,
            "historical_omlx_full_projection_behavior": True,
            "omlx_greedy_token_exact": record["layer_major"]["greedy"] == accepted_greedy,
            "omlx_final_cache_exact": [cache_digest(c) for c in layer_cache] == accepted_cache_digest,
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
