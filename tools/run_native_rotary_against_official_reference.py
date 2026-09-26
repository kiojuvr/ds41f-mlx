#!/usr/bin/env python3
"""Validate native rotary helper path against official-reference-derived fixture."""

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
DEFAULT_REFERENCE = "artifacts/rotary-official-reference-fixture.json"


def f32_from_hex(h: str) -> np.float32:
    return np.float32(struct.unpack("<f", bytes.fromhex(h))[0])


def nested_hex_to_np(obj: Any) -> np.ndarray:
    return np.array(_nested_hex_to_float(obj), dtype=np.float32)


def _nested_hex_to_float(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_nested_hex_to_float(x) for x in obj]
    return float(f32_from_hex(obj))


def freqs_hex_to_arrays(obj: Any) -> tuple[np.ndarray, np.ndarray]:
    rows_r: list[list[float]] = []
    rows_i: list[list[float]] = []
    for row in obj:
        rr, ii = [], []
        for r_hex, i_hex in row:
            rr.append(float(f32_from_hex(r_hex)))
            ii.append(float(f32_from_hex(i_hex)))
        rows_r.append(rr)
        rows_i.append(ii)
    return np.array(rows_r, dtype=np.float32), np.array(rows_i, dtype=np.float32)


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast("B")).hexdigest()


def f32_ordered_int(a: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    mask = np.where((u & np.uint32(0x80000000)) != 0, np.uint32(0xFFFFFFFF), np.uint32(0x80000000))
    return (u ^ mask).astype(np.int64)


def compare(native: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    if native.shape != expected.shape:
        return {"shape_exact": False, "native_shape": list(native.shape), "expected_shape": list(expected.shape)}
    diff = native.astype(np.float64) - expected.astype(np.float64)
    ulp = np.abs(f32_ordered_int(native) - f32_ordered_int(expected))
    return {
        "shape_exact": True,
        "bit_exact": digest(native) == digest(expected),
        "mismatch_count": int((native.view(np.uint32) != expected.view(np.uint32)).sum()),
        "max_abs_error": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "max_f32_ulp_error": int(np.max(ulp)) if ulp.size else 0,
        "native_sha256": digest(native),
        "expected_sha256": digest(expected),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", DEFAULT_CHECKPOINT))
    ap.add_argument("--reference", default=DEFAULT_REFERENCE)
    ap.add_argument("--native-out-dir", default="artifacts/m2/dwarfstar-prefill/native")
    ap.add_argument("--out", default="artifacts/native-rotary-official-reference-validation.json")
    args = ap.parse_args()

    ref = json.loads(Path(args.reference).read_text())
    if ref.get("not_omlx_derived") is not True:
        raise ValueError("reference fixture must be not_omlx_derived")
    if ref.get("classification") != "official_reference_derived_independent_arithmetic_contract":
        raise ValueError("unexpected reference fixture classification")

    exp_freq_r, exp_freq_i = freqs_hex_to_arrays(ref["expected"]["freqs_cis_complex64_hex"])
    x = nested_hex_to_np(ref["expected"]["input_f32_hex"])
    exp_rotated = nested_hex_to_np(ref["expected"]["rotated_f32_hex"])
    exp_inverse = nested_hex_to_np(ref["expected"]["inverse_after_forward_f32_hex"])
    params = ref["inputs"]["params"]

    native = load_native_prefill_library(compile_native_prefill_library(Path(args.native_out_dir)))
    got_freq_r, got_freq_i, got_rotated, got_inverse, native_result = native.official_rotary_f32(x, params)

    comparisons = {
        "freqs_real": compare(got_freq_r, exp_freq_r),
        "freqs_imag": compare(got_freq_i, exp_freq_i),
        "rotated": compare(got_rotated, exp_rotated),
        "inverse_after_forward": compare(got_inverse, exp_inverse),
    }
    tolerance_contract = {
        "max_abs_error_lte": 1.0e-6,
        "max_f32_ulp_error_lte": 64,
        "bit_exact_not_required": True,
        "reason": "Metal trigonometric implementations may differ from the pure-Python fixture while preserving the reviewed f32 rotary contract within a bounded tolerance",
    }
    within_tolerance = all(
        c.get("shape_exact") is True
        and c.get("max_abs_error", 0.0) <= tolerance_contract["max_abs_error_lte"]
        and c.get("max_f32_ulp_error", 0) <= tolerance_contract["max_f32_ulp_error_lte"]
        for c in comparisons.values()
    )

    record: dict[str, Any] = {
        "schema": "ds41f.native-rotary-official-reference-validation.v1",
        "classification": "official_reference_derived_native_validation",
        "not_omlx_derived": True,
        "purpose": "validate native precompute_freqs_cis/apply_rotary_emb data path against official-reference-derived fixture",
        "checkpoint": args.checkpoint,
        "reference_fixture": args.reference,
        "official_reference": ref["official_reference"],
        "operation_contract": ref["operation_contract"],
        "native_version": native.version(),
        "ds4_authority": {"remote": DS4_AUTHORITY_REMOTE, "commit": DS4_AUTHORITY_SHA},
        "native_result": native_result,
        "digests": {"reference": ref["digests"]},
        "tolerance_contract": tolerance_contract,
        "comparison": comparisons,
        "semantic_status": {
            "official_reference_fixture_used": True,
            "native_rotary_validated_against_contract": within_tolerance,
            "model_semantics_validated": False,
            "validated_stage": "rotary frequency/application helpers for the reference fixture scope",
        },
        "non_claims": [
            "does not execute oMLX",
            "validates YaRN only when original_seq_len > 0 in the reference fixture params",
            "does not validate attention, cache publication, logits, layers, or full-model correctness",
        ],
    }
    record["gates"] = {
        "reference_not_omlx_derived": ref.get("not_omlx_derived") is True,
        "reference_classification_expected": ref.get("classification") == "official_reference_derived_independent_arithmetic_contract",
        "native_metal_executed": native_result.get("metal_enabled") is True and native_result.get("metal_command_buffers") == 1 and native_result.get("metal_compute_encoders") == 3,
        "shape_exact": all(c.get("shape_exact") is True for c in comparisons.values()),
        "native_within_reference_tolerance": within_tolerance,
        "fixture_scope_explicit": isinstance(params.get("original_seq_len"), int),
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
