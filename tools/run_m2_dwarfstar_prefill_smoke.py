#!/usr/bin/env python3
"""Smoke the DwarfStar-authoritative prefill engine seam.

This is not a new kernel path and not official DeepSeek qualification.  It
verifies that the engine can build a DwarfStar-derived architecture plan and
execute it through the current historical oMLX operation adapter.  Logits/cache
comparisons are oMLX-compatibility diagnostics only.
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

from ds41f_mlx.dwarfstar_prefill import DwarfStarPrefillEngine  # noqa: E402
from ds41f_mlx.m2_state_publication import cache_digest, tensor_digest  # noqa: E402
from ds41f_mlx.runtime.omlx_core import OmlxRuntime, OmlxRuntimeConfig  # noqa: E402


def greedy_id(logits: Any) -> int:
    import mlx.core as mx  # type: ignore
    import numpy as np

    last = logits[:, -1, :]
    mx.eval(last)
    return int(np.asarray(mx.argmax(last, axis=-1)).reshape(-1)[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="ping\ndwarfstar prefill seam\n")
    ap.add_argument("--target-tokens", type=int, default=32)
    ap.add_argument("--chunk-tokens", type=int, default=8)
    ap.add_argument("--out", default="artifacts/m2/dwarfstar-prefill/smoke-32tok-4chunk.json")
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"))
    ap.add_argument("--omlx", default=os.environ.get("DS41F_OMLX", str(Path.home() / "omlx-0.7.0.dev2")))
    args = ap.parse_args()

    import mlx.core as mx  # type: ignore

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "schema": "ds41f.m2.dwarfstar-prefill-smoke.v2",
        "purpose": "prefill-engine seam: DwarfStar architecture authority, historical oMLX operation adapter compatibility diagnostics",
        "classification": "mixed_architecture_clean_semantics_omlx_compatibility_not_official_qualification",
        "checkpoint": args.checkpoint,
        "omlx": args.omlx,
        "target_tokens": args.target_tokens,
        "chunk_tokens": args.chunk_tokens,
    }
    rt = OmlxRuntime(OmlxRuntimeConfig(omlx_path=Path(args.omlx), checkpoint_path=Path(args.checkpoint), engram_ssd_offload=True, preserve_mtp=True))
    try:
        t0 = time.perf_counter()
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
            raise ValueError("DwarfStar prefill smoke requires at least two chunks")
        record["input_token_count"] = len(ids)
        record["chunk_lengths"] = [len(c) for c in chunks]

        accepted_cache = lm.make_cache()
        accepted_parts = []
        for chunk in chunks:
            logits = lm._forward(mx.array(chunk)[None], cache=accepted_cache)
            mx.eval(logits)
            accepted_parts.append(logits)
        accepted_last = accepted_parts[-1]

        engine_cache = lm.make_cache()
        engine = DwarfStarPrefillEngine(lm)
        result = engine.prefill(chunks, engine_cache, output_semantics="full_chunks_last", record_publications=True)
        mx.eval(result.logits)
        mx.synchronize()

        record["plan"] = result.plan.summary()
        record["semantics_contract"] = engine.semantics_contract()
        record["accepted"] = {
            "greedy": greedy_id(accepted_last),
            "logits_digest": tensor_digest(accepted_last),
            "cache_digest": [cache_digest(c) for c in accepted_cache],
        }
        record["dwarfstar_prefill_engine"] = {
            "seconds": result.seconds,
            "tokens_per_second": len(ids) / result.seconds if result.seconds > 0 else None,
            "greedy": greedy_id(result.logits),
            "logits_digest": tensor_digest(result.logits),
            "cache_digest": result.cache_digest,
            "publication_records": result.publication_records,
        }
        record["gates"] = {
            "two_or_more_chunks": len(chunks) >= 2,
            "plan_authority_recorded": bool(record["plan"].get("authority")),
            "plan_is_layer_major": record["plan"]["step_counts"].get("encode_layer_batch") == getattr(lm._config, "n_layers", 40),
            "historical_omlx_bridge_present": bool(record["semantics_contract"].get("bindings")),
            "decode_policy_is_omlx_compatibility": "oMLX" in record["semantics_contract"].get("decode_policy", ""),
            "omlx_greedy_exact": record["dwarfstar_prefill_engine"]["greedy"] == record["accepted"]["greedy"],
            "omlx_last_chunk_logits_exact": record["dwarfstar_prefill_engine"]["logits_digest"] == record["accepted"]["logits_digest"],
            "omlx_final_cache_exact": record["dwarfstar_prefill_engine"]["cache_digest"] == record["accepted"]["cache_digest"],
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
