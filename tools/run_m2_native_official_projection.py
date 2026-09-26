#!/usr/bin/env python3
"""Official projection primitive smoke for native Metal.

This validates one real DeepSeek-V4.1 checkpoint projection tensor with real
shape/representation: layers.0.ffn.gate.weight [384,5120] BF16.  The native
kernel computes x @ W.T with float32 accumulation from BF16 raw bits.  This is a
projection primitive only, not a complete layer or full prefill correctness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
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


def digest(a) -> str:
    return hashlib.sha256(memoryview(a).cast("B")).hexdigest()


def safetensor_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def tensor_memmap(path: Path, name: str, dtype, shape: tuple[int, ...]):
    import numpy as np
    header, base = safetensor_header(path)
    meta = header[name]
    begin, _end = meta["data_offsets"]
    return np.memmap(path, mode="r", dtype=dtype, offset=base + begin, shape=shape)


def bf16_to_f32(x):
    import numpy as np
    u = x.astype(np.uint32) << 16
    return u.view(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"))
    ap.add_argument("--native-out-dir", default="artifacts/m2/dwarfstar-prefill/native")
    ap.add_argument("--tokens", default="0,3", help="bounded input rows gathered from embed.weight")
    ap.add_argument("--out", default="artifacts/m2/dwarfstar-prefill/native-official-projection.json")
    args = ap.parse_args()

    import numpy as np

    ckpt = Path(args.checkpoint)
    embed_shard = ckpt / "model-00002-of-00048.safetensors"
    proj_shard = ckpt / "model-00003-of-00048.safetensors"
    tokens = np.array([int(x) for x in args.tokens.split(",") if x.strip()], dtype=np.int32)
    if tokens.size == 0:
        raise ValueError("at least one token is required")
    if int(tokens.min()) < 0 or int(tokens.max()) >= 129280:
        raise ValueError("tokens out of embed.weight range")

    embed = tensor_memmap(embed_shard, "embed.weight", np.uint16, (129280, 5120))
    weight = tensor_memmap(proj_shard, "layers.0.ffn.gate.weight", np.uint16, (384, 5120))
    x_bf16 = np.ascontiguousarray(embed[tokens], dtype=np.uint16)
    w_bf16 = np.ascontiguousarray(weight, dtype=np.uint16)

    x_f32 = bf16_to_f32(x_bf16)
    w_f32 = bf16_to_f32(w_bf16)
    ref = np.ascontiguousarray(x_f32 @ w_f32.T, dtype=np.float32)

    native = load_native_prefill_library(compile_native_prefill_library(Path(args.native_out_dir)))
    out_np, native_result = native.official_bf16_linear_f32(x_bf16, w_bf16)

    diff = out_np - ref
    abs_err = np.abs(diff)
    denom = np.maximum(np.abs(ref), np.float32(1e-30))
    rel_err = abs_err / denom
    mismatch = out_np.view(np.uint32) != ref.view(np.uint32)
    max_abs = float(abs_err.max()) if abs_err.size else 0.0
    max_rel = float(rel_err.max()) if rel_err.size else 0.0
    mismatch_count = int(mismatch.sum())
    bit_exact = mismatch_count == 0
    # Predeclared contract for this first arithmetic primitive.  This is not
    # tuned from the observed result; it reflects F32 accumulation with a
    # different reduction implementation than NumPy/MLX may use.
    tolerance_contract = {
        "max_abs_error_lte": 1.0e-4,
        "max_rel_error_lte": 1.0e-5,
        "mismatch_count_allowed": "nonzero allowed when error bounds pass",
        "bit_exact_required": False,
    }
    within = max_abs <= tolerance_contract["max_abs_error_lte"] and max_rel <= tolerance_contract["max_rel_error_lte"]

    record = {
        "schema": "ds41f.m2.native-official-projection.v1",
        "purpose": "official BF16 linear projection primitive on native Metal; no attention/MoE/HC/full-layer correctness",
        "checkpoint": str(ckpt),
        "source_weight_identity": {
            "tensor": "layers.0.ffn.gate.weight",
            "shard": str(proj_shard),
            "shape": [384, 5120],
            "dtype": "BF16 raw uint16 bits",
            "real_v41_projection": True,
        },
        "source_scale_identity": None,
        "input_identity": {
            "source": "embed.weight gathered rows",
            "shard": str(embed_shard),
            "tokens": [int(x) for x in tokens.tolist()],
            "shape": list(x_bf16.shape),
            "dtype": "BF16 raw uint16 bits",
        },
        "native_version": native.version(),
        "ds4_authority": {"remote": DS4_AUTHORITY_REMOTE, "commit": DS4_AUTHORITY_SHA},
        "native_result": native_result,
        "digests": {
            "input_digest": digest(x_bf16),
            "source_weight_digest": digest(w_bf16),
            "reference_output_digest": digest(ref),
            "native_output_digest": digest(out_np),
        },
        "arithmetic_contract": {
            "operation": "out = input_bf16_as_f32 @ weight_bf16_as_f32.T",
            "output_dtype": "F32",
            "accumulation_contract": "F32 accumulation over k=0..5119; native kernel performs one output element per thread in k-increasing order",
            "reference_provider": "NumPy float32 matmul over official BF16 values converted to F32",
            "tolerance_contract": tolerance_contract,
        },
        "comparison": {
            "bit_exact": bit_exact,
            "mismatch_count": mismatch_count,
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
            "within_predeclared_tolerance": within,
        },
        "semantic_status": {
            "official_model_data_used": True,
            "official_weight_representation_preserved": True,
            "projection_primitive_validated": within,
            "model_semantics_validated": False,
            "validated_stage": "layers.0.ffn.gate.weight BF16 linear projection only",
        },
    }
    record["gates"] = {
        "authority_sha_pinned": record["ds4_authority"]["commit"] == DS4_AUTHORITY_SHA,
        "real_checkpoint_projection_used": record["source_weight_identity"]["real_v41_projection"] is True,
        "shape_exact": native_result.get("rows") == int(tokens.size) and native_result.get("in_dim") == 5120 and native_result.get("out_dim") == 384,
        "native_metal_executed": native_result.get("metal_enabled") is True and native_result.get("metal_command_buffers") == 1,
        "predeclared_tolerance_passed": within,
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
