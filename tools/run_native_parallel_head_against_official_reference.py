#!/usr/bin/env python3
"""Validate native FP32 ParallelHead/linear seam against the bounded fixture."""

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
DEFAULT_REFERENCE = "artifacts/parallel-head-official-reference-fixture.json"
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


def compare(actual: np.ndarray, expected: np.ndarray, max_abs_lte: float, max_rel_lte: float) -> dict[str, Any]:
    a = np.ascontiguousarray(actual, dtype=np.float32)
    e = np.ascontiguousarray(expected, dtype=np.float32)
    diff = np.abs(a - e)
    denom = np.maximum(np.abs(e), np.float32(1.0e-30))
    rel = diff / denom
    max_abs = float(diff.max()) if diff.size else 0.0
    max_rel = float(rel.max()) if rel.size else 0.0
    return {
        "shape_matches": list(a.shape) == list(e.shape),
        "bit_exact": digest(a) == digest(e),
        "mismatch_count": int(np.count_nonzero(a.view(np.uint32) != e.view(np.uint32))),
        "max_abs_error": max_abs,
        "max_relative_error": max_rel,
        "max_abs_error_lte": max_abs_lte,
        "max_relative_error_lte": max_rel_lte,
        "within_tolerance": max_abs <= max_abs_lte and max_rel <= max_rel_lte,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", DEFAULT_CHECKPOINT))
    ap.add_argument("--reference", default=DEFAULT_REFERENCE)
    ap.add_argument("--native-out-dir", default="artifacts/m2/dwarfstar-prefill/native")
    ap.add_argument("--out", default="artifacts/native-parallel-head-official-reference-validation.json")
    args = ap.parse_args()

    ref = json.loads(Path(args.reference).read_text())
    if ref.get("not_omlx_derived") is not True:
        raise ValueError("reference fixture must be not_omlx_derived")
    if ref.get("classification") != "official_reference_derived_independent_arithmetic_contract":
        raise ValueError("unexpected reference fixture classification")

    ckpt = Path(args.checkpoint)
    tokens = np.array(ref["source_tensors"]["input"]["tokens"], dtype=np.int64)
    vocab_rows = np.array(ref["source_tensors"]["weight_slice"]["vocab_rows"], dtype=np.int64)
    embed = tensor_memmap(ckpt / "model-00002-of-00048.safetensors", "embed.weight", np.uint16, (VOCAB_SIZE, DIM))
    head = tensor_memmap(ckpt / "model-00043-of-00048.safetensors", "head.weight", np.uint16, (VOCAB_SIZE, DIM))
    input_f32 = np.ascontiguousarray(bf16_to_f32(np.ascontiguousarray(embed[tokens], dtype=np.uint16)), dtype=np.float32)
    weight_f32 = np.ascontiguousarray(bf16_to_f32(np.ascontiguousarray(head[vocab_rows], dtype=np.uint16)), dtype=np.float32)

    native = load_native_prefill_library(compile_native_prefill_library(Path(args.native_out_dir)))
    true_flat, native_true_result = native.official_f32_linear_f32(input_f32, weight_f32)
    false_flat, native_false_result = native.official_f32_linear_f32(input_f32[-1:, :], weight_f32)
    native_true = true_flat.reshape(1, int(tokens.size), int(vocab_rows.size))
    native_false = false_flat.reshape(1, int(vocab_rows.size))

    expected_true = np.asarray(ref["expected"]["full_logits_true_selected_f32"], dtype=np.float32)
    expected_false = np.asarray(ref["expected"]["full_logits_false_selected_f32"], dtype=np.float32)
    tol = ref["operation_contract"]["predeclared_tolerance"]
    max_abs_lte = float(tol["max_abs_error_lte"])
    max_rel_lte = float(tol["max_relative_error_lte"])
    true_cmp = compare(native_true, expected_true, max_abs_lte, max_rel_lte)
    false_cmp = compare(native_false, expected_false, max_abs_lte, max_rel_lte)
    last_cmp = compare(native_false, native_true[:, -1, :], 0.0, 0.0)

    record: dict[str, Any] = {
        "schema": "ds41f.native-parallel-head-official-reference-validation.v1",
        "classification": "official_reference_derived_native_validation",
        "not_omlx_derived": True,
        "purpose": "validate native bounded FP32 ParallelHead/F.linear seam against official-reference-derived fixture",
        "checkpoint": str(ckpt),
        "authority": {
            "reference_fixture": args.reference,
            "official_checkpoint_raw_bits": str(ckpt),
            "native_validation_provider": "Metal-backed ds41f_official_f32_linear_f32",
        },
        "reference_fixture": args.reference,
        "official_reference": ref["official_reference"],
        "operation_contract": ref["operation_contract"],
        "native_version": native.version(),
        "ds4_authority": {"remote": DS4_AUTHORITY_REMOTE, "commit": DS4_AUTHORITY_SHA},
        "native_result": {"full_logits_true": native_true_result, "full_logits_false": native_false_result},
        "digests": {
            "reference_expected_full_logits_true_f32_sha256": ref["digests"]["expected_full_logits_true_f32_sha256"],
            "reference_expected_full_logits_false_f32_sha256": ref["digests"]["expected_full_logits_false_f32_sha256"],
            "native_full_logits_true_f32_sha256": digest(native_true),
            "native_full_logits_false_f32_sha256": digest(native_false),
        },
        "comparison": {
            "full_logits_true": true_cmp,
            "full_logits_false": false_cmp,
            "false_output_equals_true_last_position": last_cmp,
        },
        "semantic_status": {
            "official_reference_fixture_used": True,
            "native_parallel_head_selected_rows_validated_against_contract": true_cmp["within_tolerance"] and false_cmp["within_tolerance"] and last_cmp["bit_exact"],
            "full_model_logits_correctness_claimed": False,
            "validated_stage": "ParallelHead.forward selected-vocabulary output-head primitive only",
        },
        "non_claims": ref["non_claims"] + ["does not validate native full-vocabulary logits allocation/readback"],
    }
    record["gates"] = {
        "reference_not_omlx_derived": ref.get("not_omlx_derived") is True,
        "reference_classification_expected": ref.get("classification") == "official_reference_derived_independent_arithmetic_contract",
        "native_metal_executed": native_true_result.get("metal_enabled") is True and native_false_result.get("metal_enabled") is True,
        "full_logits_true_within_tolerance": true_cmp["within_tolerance"],
        "full_logits_false_within_tolerance": false_cmp["within_tolerance"],
        "last_position_equivalence_passed": last_cmp["bit_exact"],
        "full_model_logits_not_claimed": record["semantic_status"]["full_model_logits_correctness_claimed"] is False,
    }
    record["ok"] = all(record["gates"].values())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
