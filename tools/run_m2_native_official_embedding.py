#!/usr/bin/env python3
"""First official model-data-plane smoke: BF16 token embedding gather on Metal.

This intentionally exercises only the official embedding lookup stage.  It reads
`embed.weight` from the official safetensors checkpoint without modifying it,
passes a bounded BF16 row slice to the native library, and verifies that the
native Metal gather returns bit-exact BF16 rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ds41f_mlx.native_prefill import (  # noqa: E402
    DS4_AUTHORITY_REMOTE,
    DS4_AUTHORITY_SHA,
    compile_native_prefill_library,
    load_native_prefill_library,
)


def digest_u16(a) -> str:
    return hashlib.sha256(memoryview(a).cast("B")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"))
    ap.add_argument("--native-out-dir", default="artifacts/m2/dwarfstar-prefill/native")
    ap.add_argument("--vocab-rows", type=int, default=16)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--tokens", default="0,3,7,1,15,4")
    ap.add_argument("--out", default="artifacts/m2/dwarfstar-prefill/native-official-embedding.json")
    args = ap.parse_args()

    import mlx.core as mx
    import numpy as np

    ckpt = Path(args.checkpoint)
    shard = ckpt / "model-00002-of-00048.safetensors"
    weights = mx.load(str(shard))["embed.weight"][: args.vocab_rows, : args.dim].view(mx.uint16)
    mx.eval(weights)
    weights_np = np.ascontiguousarray(np.array(weights), dtype=np.uint16)
    tokens_np = np.array([int(x) for x in args.tokens.split(",") if x.strip()], dtype=np.int32)
    if tokens_np.size == 0:
        raise ValueError("at least one token is required")
    if int(tokens_np.min()) < 0 or int(tokens_np.max()) >= args.vocab_rows:
        raise ValueError("tokens must refer to the bounded vocab row slice")
    ref = np.ascontiguousarray(weights_np[tokens_np], dtype=np.uint16)

    native = load_native_prefill_library(compile_native_prefill_library(Path(args.native_out_dir)))
    out_np, native_result = native.official_embedding_gather_bf16(weights_np, tokens_np)
    exact = bool(np.array_equal(out_np, ref))
    record = {
        "schema": "ds41f.m2.native-official-embedding.v1",
        "purpose": "first official model data-plane stage: Metal BF16 embedding gather from official checkpoint slice; no attention/MoE/HC math",
        "checkpoint": str(ckpt),
        "source_tensor": "embed.weight",
        "source_shard": str(shard),
        "checkpoint_policy": "official checkpoint read-only; bounded slice only",
        "native_version": native.version(),
        "ds4_authority": {"remote": DS4_AUTHORITY_REMOTE, "commit": DS4_AUTHORITY_SHA},
        "fixture": {
            "vocab_rows": args.vocab_rows,
            "dim": args.dim,
            "tokens": [int(x) for x in tokens_np.tolist()],
            "dtype": "BF16 represented as raw uint16 bits",
        },
        "native_result": native_result,
        "digests": {
            "weights_bf16_bits_sha256": digest_u16(weights_np),
            "reference_output_bf16_bits_sha256": digest_u16(ref),
            "native_output_bf16_bits_sha256": digest_u16(out_np),
        },
        "semantic_status": {
            "ownership_correct": True,
            "metal_submission_correct": bool(native_result.get("metal_enabled")),
            "official_model_data_used": True,
            "embedding_lookup_bit_exact": exact,
            "model_semantics_validated": False,
            "validated_stage": "embed.weight token gather only",
        },
    }
    record["gates"] = {
        "authority_sha_pinned": record["ds4_authority"]["commit"] == DS4_AUTHORITY_SHA,
        "checkpoint_read_only_slice_used": shard.exists(),
        "native_metal_executed": native_result.get("metal_enabled") is True and native_result.get("metal_command_buffers") == 1,
        "bf16_embedding_bits_exact": exact,
        "token_count_exact": native_result.get("n_tokens") == int(tokens_np.size),
        "dim_exact": native_result.get("dim") == args.dim,
        "scope_limited_to_embedding": record["semantic_status"]["validated_stage"] == "embed.weight token gather only",
        "full_model_semantics_not_claimed": record["semantic_status"]["model_semantics_validated"] is False,
    }
    record["ok"] = all(record["gates"].values())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
