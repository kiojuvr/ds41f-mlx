#!/usr/bin/env python3
"""Validate native Metal FP8 linear primitive against the Stage 1a fixture."""

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
DEFAULT_REFERENCE = "artifacts/fp8-linear-official-reference-fixture.json"
VOCAB_SIZE = 129280
DIM = 5120
OUT_DIM = 1280


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


def bf16_ulp_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a.astype(np.int32) - b.astype(np.int32))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", DEFAULT_CHECKPOINT))
    ap.add_argument("--reference", default=DEFAULT_REFERENCE)
    ap.add_argument("--native-out-dir", default="artifacts/m2/dwarfstar-prefill/native")
    ap.add_argument("--out", default="artifacts/native-fp8-linear-official-reference-validation.json")
    args = ap.parse_args()
    ref = json.loads(Path(args.reference).read_text())
    if ref.get("not_omlx_derived") is not True:
        raise ValueError("reference fixture must be not_omlx_derived")
    ckpt = Path(args.checkpoint)
    tokens = np.array(ref["source_tensors"]["input"]["tokens"], dtype=np.int64)
    w_rows = ref["source_tensors"]["weight"]["slice_rows"]
    w_cols = ref["source_tensors"]["weight"]["slice_cols"]
    s_rows = ref["source_tensors"]["weight_scale"]["slice_rows"]
    s_cols = ref["source_tensors"]["weight_scale"]["slice_cols"]
    block = int(ref["operation_contract"]["block_size"])
    embed = tensor_memmap(ckpt / "model-00002-of-00048.safetensors", "embed.weight", np.uint16, (VOCAB_SIZE, DIM))
    weight = tensor_memmap(ckpt / "model-00003-of-00048.safetensors", "layers.0.attn.wq_a.weight", np.uint8, (OUT_DIM, DIM))
    scale = tensor_memmap(ckpt / "model-00003-of-00048.safetensors", "layers.0.attn.wq_a.scale", np.uint8, (OUT_DIM // block, DIM // block))
    input_bf16 = np.ascontiguousarray(embed[tokens, w_cols[0]:w_cols[1]], dtype=np.uint16)
    weight_fp8 = np.ascontiguousarray(weight[w_rows[0]:w_rows[1], w_cols[0]:w_cols[1]], dtype=np.uint8)
    weight_scale = np.ascontiguousarray(scale[s_rows[0]:s_rows[1], s_cols[0]:s_cols[1]], dtype=np.uint8)
    expected = np.asarray(ref["expected"]["output_bf16_uint16"], dtype=np.uint16)
    native = load_native_prefill_library(compile_native_prefill_library(Path(args.native_out_dir)))
    out, native_result = native.official_fp8_linear_bf16(input_bf16, weight_fp8, weight_scale, block)
    diff = bf16_ulp_diff(out, expected)
    max_ulp = int(diff.max()) if diff.size else 0
    mismatch_count = int(np.count_nonzero(diff))
    tol = ref["operation_contract"]["predeclared_tolerance"]
    record: dict[str, Any] = {
        "schema": "ds41f.native-fp8-linear-official-reference-validation.v1",
        "classification": "official_reference_derived_native_validation",
        "not_omlx_derived": True,
        "purpose": "validate native Metal FP8 linear primitive against bounded official-checkpoint Stage 1a fixture",
        "checkpoint": str(ckpt),
        "authority": {"reference_fixture": args.reference, "official_checkpoint_raw_bits": str(ckpt), "native_validation_provider": "Metal-backed ds41f_official_fp8_linear_bf16"},
        "reference_fixture": args.reference,
        "official_reference": ref["official_reference"],
        "operation_contract": ref["operation_contract"],
        "native_version": native.version(),
        "ds4_authority": {"remote": DS4_AUTHORITY_REMOTE, "commit": DS4_AUTHORITY_SHA},
        "native_result": native_result,
        "digests": {"reference_expected_output_bf16_sha256": ref["digests"]["expected_output_bf16_sha256"], "native_output_bf16_sha256": digest(out)},
        "comparison": {"bit_exact": mismatch_count == 0, "mismatch_count": mismatch_count, "max_bf16_ulp_error": max_ulp, "max_bf16_ulp_lte": int(tol["max_bf16_ulp_lte"]), "within_tolerance": max_ulp <= int(tol["max_bf16_ulp_lte"]), "shape_matches": list(out.shape) == list(expected.shape)},
        "semantic_status": {"official_reference_fixture_used": True, "native_fp8_linear_validated_against_contract": max_ulp <= int(tol["max_bf16_ulp_lte"]), "attention_q_prelude_integrated": False, "model_semantics_validated": False, "validated_stage": "FP8 linear/GEMM primitive only"},
        "non_claims": ref["non_claims"] + ["does not validate native full attention integration"],
    }
    record["gates"] = {"reference_not_omlx_derived": ref.get("not_omlx_derived") is True, "reference_classification_expected": ref.get("classification") == "official_reference_derived_independent_arithmetic_contract", "native_metal_executed": native_result.get("metal_enabled") is True and native_result.get("metal_command_buffers") == 1, "shape_exact": record["comparison"]["shape_matches"], "native_within_predeclared_tolerance": record["comparison"]["within_tolerance"], "attention_integration_not_claimed": record["semantic_status"]["attention_q_prelude_integrated"] is False, "full_model_semantics_not_claimed": record["semantic_status"]["model_semantics_validated"] is False}
    record["ok"] = all(record["gates"].values())
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(outp)
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
