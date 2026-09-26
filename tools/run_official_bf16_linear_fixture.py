#!/usr/bin/env python3
"""Generate an official-reference-scoped BF16 linear fixture.

The reviewed official reference `linear()` dispatches non-quantized weights to
`torch.nn.functional.linear(x, weight)`. This bounded fixture records that source
identity and uses an explicit independent arithmetic contract over official BF16
checkpoint tensors: F32 accumulation of `input @ weight.T`. It does not execute
oMLX, PyTorch, MLX, or native Metal code.
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
LINEAR_SHA = "c0c1edd8e542d2004472766686fb445859775ee0b51346979cd9ca573c1c7ada"
LINEAR_FORWARD_SHA = "2b53749a0c9c5bd1c3ec1553ec4f107378f3050a8f585c3c773dd0b2a728404e"
VOCAB_SIZE = 129280
DIM = 5120
OUT_DIM = 384


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


def parse_tokens(s: str) -> np.ndarray:
    toks = np.array([int(x) for x in s.split(",") if x.strip()], dtype=np.int64)
    if toks.size == 0:
        raise ValueError("at least one token is required")
    if int(toks.min()) < 0 or int(toks.max()) >= VOCAB_SIZE:
        raise ValueError("token out of embed.weight range")
    return toks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", DEFAULT_CHECKPOINT))
    ap.add_argument("--tokens", default="0,3")
    ap.add_argument("--out", default="artifacts/bf16-linear-official-reference-fixture.json")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    tokens = parse_tokens(args.tokens)
    embed_shard = ckpt / "model-00002-of-00048.safetensors"
    proj_shard = ckpt / "model-00003-of-00048.safetensors"
    embed = tensor_memmap(embed_shard, "embed.weight", np.uint16, (VOCAB_SIZE, DIM))
    weight = tensor_memmap(proj_shard, "layers.0.ffn.gate.weight", np.uint16, (OUT_DIM, DIM))

    x_bf16 = np.ascontiguousarray(embed[tokens], dtype=np.uint16)
    w_bf16 = np.ascontiguousarray(weight, dtype=np.uint16)
    ref = np.ascontiguousarray(bf16_to_f32(x_bf16) @ bf16_to_f32(w_bf16).T, dtype=np.float32)

    record: dict[str, Any] = {
        "schema": "ds41f.bf16-linear-official-reference-fixture.v1",
        "classification": "official_reference_derived_independent_arithmetic_contract",
        "not_omlx_derived": True,
        "purpose": "bounded BF16 linear fixture for reviewed official non-quantized linear/F.linear branch over official checkpoint tensors",
        "checkpoint": str(ckpt),
        "official_reference": {
            "file": "inference/model.py",
            "file_sha256": MODEL_PY_SHA,
            "functions": [
                {"name": "linear", "source_lines": [181, 207], "source_sha256": LINEAR_SHA, "reviewed_branch": "else: return F.linear(x, weight)"},
                {"name": "Linear.forward", "source_lines": [242, 243], "source_sha256": LINEAR_FORWARD_SHA},
            ],
        },
        "source_tensors": {
            "input": {"source": "embed.weight gathered rows", "shard": str(embed_shard), "tokens": [int(x) for x in tokens.tolist()], "shape": list(x_bf16.shape), "dtype": "BF16 raw uint16 bits", "digest": digest(x_bf16)},
            "weight": {"name": "layers.0.ffn.gate.weight", "shard": str(proj_shard), "shape": [OUT_DIM, DIM], "dtype": "BF16 raw uint16 bits", "digest": digest(w_bf16)},
        },
        "operation_contract": {
            "official_reference_branch": "non-quantized weight path calls torch.nn.functional.linear(x, weight)",
            "fixture_operation": "output = input_bf16_as_f32 @ weight_bf16_as_f32.T",
            "accumulation_contract": "F32 accumulation over k=0..5119",
            "output_dtype": "F32",
            "output_shape": list(ref.shape),
            "expected_provider": "independent NumPy F32 matmul over official BF16 input/weight values; no oMLX/PyTorch/MLX/native execution",
        },
        "digests": {"expected_output_f32_sha256": digest(ref)},
        "comparison": {"self_consistent": True},
        "non_claims": [
            "does not execute oMLX",
            "does not execute PyTorch backend F.linear",
            "does not validate PyTorch backend-specific BF16 accumulation beyond the declared F32 arithmetic contract",
            "does not validate activation, RMSNorm, MoE, attention, logits, cache, or full layer correctness",
            "does not implement or validate additional native model math",
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
