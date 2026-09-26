#!/usr/bin/env python3
"""Validate native BF16 linear primitive against official-reference-scoped fixture."""

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
DEFAULT_REFERENCE = "artifacts/bf16-linear-official-reference-fixture.json"
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", DEFAULT_CHECKPOINT))
    ap.add_argument("--reference", default=DEFAULT_REFERENCE)
    ap.add_argument("--native-out-dir", default="artifacts/m2/dwarfstar-prefill/native")
    ap.add_argument("--out", default="artifacts/native-linear-official-reference-validation.json")
    args = ap.parse_args()

    ref = json.loads(Path(args.reference).read_text())
    if ref.get("not_omlx_derived") is not True:
        raise ValueError("reference fixture must be not_omlx_derived")
    if ref.get("classification") != "official_reference_derived_independent_arithmetic_contract":
        raise ValueError("unexpected reference fixture classification")

    tokens = np.array(ref["source_tensors"]["input"]["tokens"], dtype=np.int32)
    ckpt = Path(args.checkpoint)
    embed_shard = ckpt / "model-00002-of-00048.safetensors"
    proj_shard = ckpt / "model-00003-of-00048.safetensors"
    embed = tensor_memmap(embed_shard, "embed.weight", np.uint16, (VOCAB_SIZE, DIM))
    weight = tensor_memmap(proj_shard, "layers.0.ffn.gate.weight", np.uint16, (OUT_DIM, DIM))
    x_bf16 = np.ascontiguousarray(embed[tokens], dtype=np.uint16)
    w_bf16 = np.ascontiguousarray(weight, dtype=np.uint16)

    native = load_native_prefill_library(compile_native_prefill_library(Path(args.native_out_dir)))
    out_np, native_result = native.official_bf16_linear_f32(x_bf16, w_bf16)
    native_digest = digest(out_np)
    expected_digest = ref["digests"]["expected_output_f32_sha256"]
    matches = native_digest == expected_digest

    record: dict[str, Any] = {
        "schema": "ds41f.native-linear-official-reference-validation.v1",
        "classification": "official_reference_derived_native_validation",
        "not_omlx_derived": True,
        "purpose": "validate existing native BF16 linear primitive against official-reference-scoped independent arithmetic fixture",
        "checkpoint": str(ckpt),
        "reference_fixture": args.reference,
        "official_reference": ref["official_reference"],
        "operation_contract": ref["operation_contract"],
        "native_version": native.version(),
        "ds4_authority": {"remote": DS4_AUTHORITY_REMOTE, "commit": DS4_AUTHORITY_SHA},
        "native_result": native_result,
        "digests": {"reference_expected_output_f32_sha256": expected_digest, "native_output_f32_sha256": native_digest},
        "comparison": {"bit_exact": matches, "mismatch_count": 0 if matches else None},
        "semantic_status": {
            "official_reference_fixture_used": True,
            "native_linear_validated_against_contract": matches,
            "model_semantics_validated": False,
            "validated_stage": "BF16 dense linear primitive only",
        },
        "non_claims": [
            "does not execute oMLX",
            "does not validate PyTorch backend-specific BF16 accumulation beyond declared F32 arithmetic contract",
            "does not validate activation, RMSNorm, MoE, attention, logits, cache, or full layer correctness",
            "does not add or modify native Metal code",
        ],
    }
    record["gates"] = {
        "reference_not_omlx_derived": ref.get("not_omlx_derived") is True,
        "reference_classification_expected": ref.get("classification") == "official_reference_derived_independent_arithmetic_contract",
        "native_metal_executed": native_result.get("metal_enabled") is True and native_result.get("metal_command_buffers") == 1,
        "shape_exact": native_result.get("rows") == int(tokens.size) and native_result.get("in_dim") == DIM and native_result.get("out_dim") == OUT_DIM,
        "native_matches_reference_fixture": matches,
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
