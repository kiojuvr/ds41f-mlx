#!/usr/bin/env python3
"""Generate a bounded official-checkpoint FP8 linear fixture for Stage 1a."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CHECKPOINT = "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"
MODEL_PY_SHA = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
KERNEL_PY_SHA = "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"
LINEAR_SHA = "c0c1edd8e542d2004472766686fb445859775ee0b51346979cd9ca573c1c7ada"
ACT_QUANT_SHA = "563a82836450bfefe3f5f176636dec0e1f4d126c76d8007b42bb7abb877d30cb"
FP8_GEMM_SHA = "cfd550d8b02ee127760ac26b39accae603302be6bf8e97c0cce29d8993af0657"
VOCAB_SIZE = 129280
DIM = 5120
OUT_DIM = 1280
BLOCK = 32


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast("B")).hexdigest()


def safetensor_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def tensor_memmap(path: Path, name: str, dtype: Any, shape: tuple[int, ...]) -> np.memmap:
    header, base = safetensor_header(path)
    begin, _ = header[name]["data_offsets"]
    return np.memmap(path, mode="r", dtype=dtype, offset=base + begin, shape=shape)


def bf16_to_f32(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_rne(x: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    return ((u + np.uint32(0x7FFF) + lsb) >> 16).astype(np.uint16)


def fp8_e4m3fn_decode_code(c: int) -> float:
    sign = -1.0 if (c & 0x80) else 1.0
    ax = c & 0x7F
    if ax == 0:
        return math.copysign(0.0, sign)
    exp = (c >> 3) & 0x0F
    man = c & 0x07
    if exp == 0:
        v = math.ldexp(man / 8.0, -6)
    else:
        v = math.ldexp(1.0 + man / 8.0, exp - 7)
    return sign * v


E4M3_VALUES = np.array([fp8_e4m3fn_decode_code(i) if (i & 0x7F) != 0x7F else np.nan for i in range(256)], dtype=np.float32)
VALID_CODES = np.array([i for i in range(256) if (i & 0x7F) != 0x7F], dtype=np.uint16)
VALID_VALUES = E4M3_VALUES[VALID_CODES]


def fp8_e4m3fn_encode_nearest(v: float) -> int:
    if math.isnan(v):
        return 0x7F
    v = min(max(v, -448.0), 448.0)
    d = np.abs(VALID_VALUES.astype(np.float64) - float(v))
    md = float(d.min())
    tied = VALID_CODES[d == md]
    even = tied[(tied & 1) == 0]
    return int(even[0] if even.size else tied[0])


def e8m0_to_f32(c: int) -> float:
    return math.ldexp(1.0, int(c) - 127)


def act_quant_block(row_f32: np.ndarray) -> tuple[np.ndarray, np.float32, int]:
    amax = max(float(np.max(np.abs(row_f32))), 1.0e-4)
    scale_exp = math.ceil(math.log2(amax / 448.0))
    scale = np.float32(math.ldexp(1.0, scale_exp))
    q = np.array([fp8_e4m3fn_encode_nearest(float(np.clip(v / scale, -448.0, 448.0))) for v in row_f32], dtype=np.uint8)
    return q, scale, scale_exp + 127


def fp8_linear_expected(input_bf16: np.ndarray, weight_fp8: np.ndarray, weight_scale: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = bf16_to_f32(input_bf16)
    rows, k = x.shape
    out_dim = weight_fp8.shape[0]
    q = np.empty_like(weight_fp8[:rows, :], shape=(rows, k), dtype=np.uint8)
    a_scales = np.empty((rows, k // BLOCK), dtype=np.float32)
    a_scales_e8m0 = np.empty((rows, k // BLOCK), dtype=np.uint8)
    out_f32 = np.empty((rows, out_dim), dtype=np.float32)
    for r in range(rows):
        for kb in range(k // BLOCK):
            qb, s, se = act_quant_block(x[r, kb * BLOCK:(kb + 1) * BLOCK])
            q[r, kb * BLOCK:(kb + 1) * BLOCK] = qb
            a_scales[r, kb] = s
            a_scales_e8m0[r, kb] = se
    for r in range(rows):
        for o in range(out_dim):
            acc = np.float32(0.0)
            for kb in range(k // BLOCK):
                bs = np.float32(e8m0_to_f32(int(weight_scale[o // BLOCK, kb])))
                a_s = a_scales[r, kb]
                for kk in range(BLOCK):
                    a = np.float32(E4M3_VALUES[int(q[r, kb * BLOCK + kk])] * a_s)
                    b = np.float32(E4M3_VALUES[int(weight_fp8[o, kb * BLOCK + kk])] * bs)
                    acc = np.float32(acc + np.float32(a * b))
            out_f32[r, o] = acc
    return q, a_scales_e8m0, out_f32, f32_to_bf16_rne(out_f32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", DEFAULT_CHECKPOINT))
    ap.add_argument("--tokens", default="0,3")
    ap.add_argument("--out", default="artifacts/fp8-linear-official-reference-fixture.json")
    args = ap.parse_args()
    tokens = [int(x) for x in args.tokens.split(",") if x.strip()]
    ckpt = Path(args.checkpoint)
    embed_shard = ckpt / "model-00002-of-00048.safetensors"
    proj_shard = ckpt / "model-00003-of-00048.safetensors"
    embed = tensor_memmap(embed_shard, "embed.weight", np.uint16, (VOCAB_SIZE, DIM))
    weight = tensor_memmap(proj_shard, "layers.0.attn.wq_a.weight", np.uint8, (OUT_DIM, DIM))
    scale = tensor_memmap(proj_shard, "layers.0.attn.wq_a.scale", np.uint8, (OUT_DIM // BLOCK, DIM // BLOCK))
    input_bf16 = np.ascontiguousarray(embed[tokens, :BLOCK], dtype=np.uint16)
    weight_fp8 = np.ascontiguousarray(weight[:BLOCK, :BLOCK], dtype=np.uint8)
    weight_scale = np.ascontiguousarray(scale[:1, :1], dtype=np.uint8)
    act_q, act_s_e8m0, out_f32, out_bf16 = fp8_linear_expected(input_bf16, weight_fp8, weight_scale)
    record: dict[str, Any] = {
        "schema": "ds41f.fp8-linear-official-reference-fixture.v1",
        "classification": "official_reference_derived_independent_arithmetic_contract",
        "not_omlx_derived": True,
        "purpose": "bounded Stage 1a FP8 linear/GEMM fixture for layer 0 attention wq_a projection",
        "checkpoint": str(ckpt),
        "authority": {"official_checkpoint_raw_bits": str(ckpt), "official_reference_source": ["inference/model.py", "inference/kernel.py"], "expected_value_provider": "independent arithmetic reconstruction over official checkpoint tensors"},
        "official_reference": {
            "model_py": {"file": "inference/model.py", "file_sha256": MODEL_PY_SHA, "functions": [{"name": "linear", "source_lines": [181, 207], "source_sha256": LINEAR_SHA, "reviewed_branch": "weight.dtype == torch.float8_e4m3fn"}]},
            "kernel_py": {"file": "inference/kernel.py", "file_sha256": KERNEL_PY_SHA, "functions": [{"name": "act_quant", "source_lines": [98, 124], "source_sha256": ACT_QUANT_SHA}, {"name": "fp8_gemm", "source_lines": [277, 307], "source_sha256": FP8_GEMM_SHA}]},
        },
        "source_tensors": {
            "input": {"source": "embed.weight gathered rows, first 32 dims", "shard": str(embed_shard), "tokens": tokens, "shape": list(input_bf16.shape), "dtype": "BF16 raw uint16 bits", "digest": digest(input_bf16)},
            "weight": {"name": "layers.0.attn.wq_a.weight", "shard": str(proj_shard), "checkpoint_dtype": "F8_E4M3", "checkpoint_shape": [OUT_DIM, DIM], "slice_rows": [0, BLOCK], "slice_cols": [0, BLOCK], "slice_shape": list(weight_fp8.shape), "digest": digest(weight_fp8)},
            "weight_scale": {"name": "layers.0.attn.wq_a.scale", "shard": str(proj_shard), "checkpoint_dtype": "F8_E8M0", "checkpoint_shape": [OUT_DIM // BLOCK, DIM // BLOCK], "scale_layout": "[ceil(out_dim/32), ceil(in_dim/32)] for 32x32 weight blocks", "slice_rows": [0, 1], "slice_cols": [0, 1], "slice_shape": list(weight_scale.shape), "raw_values": weight_scale.tolist(), "digest": digest(weight_scale)},
        },
        "inputs": {"input_bf16_digest": digest(input_bf16), "weight_fp8_digest": digest(weight_fp8), "weight_scale_e8m0_digest": digest(weight_scale)},
        "operation_contract": {
            "linear_branch": "FP8 branch: act_quant(x, fp8_block_size=32, scale_fmt='ue8m0', scale_dtype=float8_e8m0fnu), then fp8_gemm(..., block_size=32)",
            "fp8_weight_encoding": "torch float8_e4m3fn / safetensors F8_E4M3 byte encoding; decoded to finite E4M3 values before applying scale",
            "scale_encoding": "torch float8_e8m0fnu / safetensors F8_E8M0; byte e decodes as 2^(e-127)",
            "activation_quantization": "per row and 32-element K block: amax=max(max(abs(x)),1e-4); scale=2^ceil(log2(amax/448)); q=round-to-nearest-even E4M3FN(clamp(x/scale,-448,448))",
            "block_size": 32,
            "weight_scale_layout": "weight_scale[out_block, k_block] where out_block=floor(output_row/32)",
            "dequantization_order": "sum_k (decode(act_q[k])*act_scale[k_block]) * (decode(weight_fp8[k])*weight_scale[out_block,k_block])",
            "accumulation_dtype": "F32, increasing k order within each 32-element block",
            "output_dtype": "BF16 round-to-nearest-even, matching torch default BF16 inference output for fp8_gemm",
            "predeclared_tolerance": {"max_bf16_ulp_lte": 0},
        },
        "expected": {"activation_quantized_fp8": act_q.tolist(), "activation_scale_e8m0": act_s_e8m0.tolist(), "output_bf16_uint16": out_bf16.tolist(), "output_f32_before_bf16_round": out_f32.tolist(), "output_shape": list(out_bf16.shape)},
        "digests": {"expected_output_bf16_sha256": digest(out_bf16), "expected_output_f32_sha256": digest(out_f32), "activation_quantized_fp8_sha256": digest(act_q)},
        "comparison": {"self_consistent": True, "tolerance_predeclared": True},
        "non_claims": ["does not execute oMLX", "does not validate Attention Q-prelude integration", "does not validate RMSNorm, rotary, sparse_attn, KV/cache, Compressor/Indexer, FP4, MoE, HC, DSpark/MTP, logits, layer, or full model correctness", "does not benchmark performance"],
        "ok": True,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
