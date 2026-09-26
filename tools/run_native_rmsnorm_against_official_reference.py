#!/usr/bin/env python3
"""Validate native RMSNorm primitive against official-reference-derived fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ds41f_mlx.native_prefill import DS4_AUTHORITY_REMOTE, DS4_AUTHORITY_SHA, compile_native_prefill_library, load_native_prefill_library  # noqa: E402

DEFAULT_CHECKPOINT = "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"
DEFAULT_REFERENCE = "artifacts/rmsnorm-official-reference-fixture.json"
VOCAB_SIZE = 129280
DIM = 5120


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


def rmsnorm_expected(x_bf16: np.ndarray, weight_bf16: np.ndarray, eps: float) -> np.ndarray:
    x = bf16_to_f32(x_bf16)
    w = bf16_to_f32(weight_bf16)
    var = np.mean(np.square(x, dtype=np.float32), axis=-1, keepdims=True, dtype=np.float32)
    y_f32 = (x * (np.float32(1.0) / np.sqrt(var + np.float32(eps), dtype=np.float32))) * w
    return np.ascontiguousarray(f32_to_bf16_rne(y_f32), dtype=np.uint16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", DEFAULT_CHECKPOINT))
    ap.add_argument("--reference", default=DEFAULT_REFERENCE)
    ap.add_argument("--native-out-dir", default="artifacts/m2/dwarfstar-prefill/native")
    ap.add_argument("--out", default="artifacts/native-rmsnorm-official-reference-validation.json")
    args = ap.parse_args()

    ref = json.loads(Path(args.reference).read_text())
    if ref.get("not_omlx_derived") is not True:
        raise ValueError("reference fixture must be not_omlx_derived")
    if ref.get("classification") != "official_reference_derived_independent_arithmetic_contract":
        raise ValueError("unexpected reference fixture classification")

    tokens = np.array(ref["source_tensors"]["input"]["tokens"], dtype=np.int32)
    eps = float(ref["operation_contract"]["eps"])
    ckpt = Path(args.checkpoint)
    embed_shard = ckpt / "model-00002-of-00048.safetensors"
    norm_shard = ckpt / "model-00003-of-00048.safetensors"
    embed = tensor_memmap(embed_shard, "embed.weight", np.uint16, (VOCAB_SIZE, DIM))
    weight = tensor_memmap(norm_shard, "layers.0.attn_norm.weight", np.uint16, (DIM,))
    x_bf16 = np.ascontiguousarray(embed[tokens], dtype=np.uint16)
    w_bf16 = np.ascontiguousarray(weight, dtype=np.uint16)

    native = load_native_prefill_library(compile_native_prefill_library(Path(args.native_out_dir)))
    out_np, native_result = native.official_rmsnorm_bf16(x_bf16, w_bf16, eps=eps)
    native_digest = digest(out_np)
    expected_digest = ref["digests"]["expected_output_bf16_bits_sha256"]
    expected = rmsnorm_expected(x_bf16, w_bf16, eps)
    expected_recomputed_digest = digest(expected)
    if expected_recomputed_digest != expected_digest:
        raise RuntimeError("recomputed expected RMSNorm digest does not match reference fixture")
    diff = out_np.astype(np.int32) - expected.astype(np.int32)
    abs_ulp = np.abs(diff)
    mismatch_count = int((out_np != expected).sum())
    max_ulp = int(abs_ulp.max()) if abs_ulp.size else 0
    matches = native_digest == expected_digest
    tolerance_contract = {"max_bf16_ulp_lte": 1, "mismatch_count_allowed": "nonzero allowed when max_bf16_ulp_lte passes"}
    within_tolerance = max_ulp <= tolerance_contract["max_bf16_ulp_lte"]

    record: dict[str, Any] = {
        "schema": "ds41f.native-rmsnorm-official-reference-validation.v1",
        "classification": "official_reference_derived_native_validation",
        "not_omlx_derived": True,
        "purpose": "validate native BF16 RMSNorm primitive against official-reference-derived fixture",
        "checkpoint": str(ckpt),
        "reference_fixture": args.reference,
        "official_reference": ref["official_reference"],
        "operation_contract": ref["operation_contract"],
        "native_version": native.version(),
        "ds4_authority": {"remote": DS4_AUTHORITY_REMOTE, "commit": DS4_AUTHORITY_SHA},
        "native_result": native_result,
        "digests": {"reference_expected_output_bf16_bits_sha256": expected_digest, "recomputed_expected_output_bf16_bits_sha256": expected_recomputed_digest, "native_output_bf16_bits_sha256": native_digest},
        "tolerance_contract": tolerance_contract,
        "comparison": {"bit_exact": matches, "mismatch_count": mismatch_count, "max_bf16_ulp_error": max_ulp, "within_predeclared_tolerance": within_tolerance},
        "semantic_status": {
            "official_reference_fixture_used": True,
            "native_rmsnorm_validated_against_contract": within_tolerance,
            "model_semantics_validated": False,
            "validated_stage": "BF16 RMSNorm primitive only",
        },
        "non_claims": [
            "does not execute oMLX",
            "does not validate backend-specific PyTorch rsqrt beyond declared fixture contract",
            "does not validate HC repeat/pre-mask, attention, MoE, Engram, DSpark/MTP, logits, cache, or full layer correctness",
        ],
    }
    record["gates"] = {
        "reference_not_omlx_derived": ref.get("not_omlx_derived") is True,
        "reference_classification_expected": ref.get("classification") == "official_reference_derived_independent_arithmetic_contract",
        "native_metal_executed": native_result.get("metal_enabled") is True and native_result.get("metal_command_buffers") == 1,
        "shape_exact": native_result.get("rows") == int(tokens.size) and native_result.get("dim") == DIM,
        "native_within_reference_tolerance": within_tolerance,
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
