#!/usr/bin/env python3
"""Generate an official-reference-derived RMSNorm fixture.

Models reviewed `RMSNorm.forward` from the official reference over official
checkpoint BF16 tensors. It does not execute oMLX, PyTorch, MLX, or native code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CHECKPOINT = "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"
MODEL_PY_SHA = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
RMSNORM_SHA = "ac829397ad0c5f99412def7adb54ba0334397baa5c2ab795d4531f072fc47ecc"
RMSNORM_FORWARD_SHA = "adb7c70ed245e8830f6692b8c026fd7f2d17518cd98b68bf973e75524a921c85"
VOCAB_SIZE = 129280
DIM = 5120
EPS = 1.0e-6


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast("B")).hexdigest()


def safetensor_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def tensor_memmap(path: Path, name: str, dtype: Any, shape: tuple[int, ...]) -> np.memmap:
    header, base = safetensor_header(path)
    meta = header[name]
    begin, _end = meta["data_offsets"]
    return np.memmap(path, mode="r", dtype=dtype, offset=base + begin, shape=shape)


def bf16_to_f32(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.uint32) << 16
    return u.view(np.float32)


def f32_to_bf16_rne(x: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x.astype(np.float32)).view(np.uint32)
    lsb = (u >> 16) & 1
    rounded = u + np.uint32(0x7FFF) + lsb
    return (rounded >> 16).astype(np.uint16)


def parse_tokens(s: str) -> np.ndarray:
    toks = np.array([int(x) for x in s.split(",") if x.strip()], dtype=np.int64)
    if toks.size == 0:
        raise ValueError("at least one token is required")
    if int(toks.min()) < 0 or int(toks.max()) >= VOCAB_SIZE:
        raise ValueError("token out of embed.weight range")
    return toks


def rmsnorm_bf16_fixture(x_bf16: np.ndarray, weight_bf16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = bf16_to_f32(x_bf16)
    w = bf16_to_f32(weight_bf16)
    var = np.mean(np.square(x, dtype=np.float32), axis=-1, keepdims=True, dtype=np.float32)
    y_f32 = (x * (np.float32(1.0) / np.sqrt(var + np.float32(EPS), dtype=np.float32))) * w
    y_bf16 = f32_to_bf16_rne(y_f32)
    return np.ascontiguousarray(y_f32, dtype=np.float32), np.ascontiguousarray(y_bf16, dtype=np.uint16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", DEFAULT_CHECKPOINT))
    ap.add_argument("--tokens", default="0,3")
    ap.add_argument("--out", default="artifacts/rmsnorm-official-reference-fixture.json")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    tokens = parse_tokens(args.tokens)
    embed_shard = ckpt / "model-00002-of-00048.safetensors"
    norm_shard = ckpt / "model-00003-of-00048.safetensors"
    embed = tensor_memmap(embed_shard, "embed.weight", np.uint16, (VOCAB_SIZE, DIM))
    weight = tensor_memmap(norm_shard, "layers.0.attn_norm.weight", np.uint16, (DIM,))
    x_bf16 = np.ascontiguousarray(embed[tokens], dtype=np.uint16)
    w_bf16 = np.ascontiguousarray(weight, dtype=np.uint16)
    y_f32, y_bf16 = rmsnorm_bf16_fixture(x_bf16, w_bf16)

    record: dict[str, Any] = {
        "schema": "ds41f.rmsnorm-official-reference-fixture.v1",
        "classification": "official_reference_derived_independent_arithmetic_contract",
        "not_omlx_derived": True,
        "purpose": "bounded RMSNorm.forward fixture from reviewed official reference over official BF16 checkpoint tensors",
        "checkpoint": str(ckpt),
        "official_reference": {
            "file": "inference/model.py",
            "file_sha256": MODEL_PY_SHA,
            "targets": [
                {"name": "RMSNorm", "source_lines": [281, 293], "source_sha256": RMSNORM_SHA},
                {"name": "RMSNorm.forward", "source_lines": [288, 293], "source_sha256": RMSNORM_FORWARD_SHA},
            ],
        },
        "source_tensors": {
            "input": {"source": "embed.weight gathered rows", "shard": str(embed_shard), "tokens": [int(x) for x in tokens.tolist()], "shape": list(x_bf16.shape), "dtype": "BF16 raw uint16 bits", "digest": digest(x_bf16)},
            "weight": {"name": "layers.0.attn_norm.weight", "shard": str(norm_shard), "shape": [DIM], "dtype": "BF16 raw uint16 bits", "digest": digest(w_bf16)},
        },
        "operation_contract": {
            "official_reference_operation": "dtype=x.dtype; x=x.float(); var=x.square().mean(-1, keepdim=True); x=x*torch.rsqrt(var+eps); return (weight*x).to(dtype)",
            "fixture_operation": "F32 variance/rsqrt/multiply over BF16 input and weight converted to F32, then BF16 round-to-nearest-even output",
            "eps": EPS,
            "reduction_dim": DIM,
            "output_shape": list(y_bf16.shape),
            "output_dtype": "BF16 raw uint16 bits",
            "expected_provider": "independent NumPy implementation of reviewed RMSNorm.forward arithmetic contract over official checkpoint bits; no oMLX/PyTorch/MLX/native execution",
        },
        "digests": {
            "intermediate_output_f32_sha256": digest(y_f32),
            "expected_output_bf16_bits_sha256": digest(y_bf16),
        },
        "comparison": {"self_consistent": True},
        "native_validation_status": "available_in_artifacts/native-rmsnorm-official-reference-validation.json",
        "non_claims": [
            "does not execute oMLX",
            "does not execute PyTorch RMSNorm",
            "does not validate backend-specific rsqrt beyond declared NumPy F32 contract",
            "does not validate HC repeat/pre-mask, attention, MoE, Engram, DSpark/MTP, logits, cache, or full layer correctness",
            "does not implement or validate native RMSNorm",
        ],
        "ok": True,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
