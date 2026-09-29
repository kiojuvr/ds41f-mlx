#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT, DEFAULT_OMLX
from tools.run_m4_block0_ffn_pre_norm_composition import arr_info, cmp_bf16, file_sha, git_identity
from tools.run_native_first_incremental_block0_engram1_validation import apply_engram1_dynamic
from tools.source_identity import source_identity

SCHEMA = "ds41f.m4.engram1-same-input.v1"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def hash_cmp(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    return {
        "actual": arr_info(a, values=True),
        "expected": arr_info(b, values=True),
        "exact": bool(np.array_equal(a, b)),
        "within_contract": bool(np.array_equal(a, b)),
        "contract": "Engram hash IDs exact",
    }


def run_engram(ck: Path, h: np.ndarray, ids: np.ndarray) -> dict[str, Any]:
    post, sparse_embedding, wkv, gate, residual = apply_engram1_dynamic(ck, h, ids)
    return {
        "post": post,
        "sparse_embedding": sparse_embedding,
        "wkv": wkv,
        "gate": gate,
        "residual": residual,
    }


def endpoint_summary(label: str, got: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    return {"name": label, "output": arr_info(got), "target": arr_info(target), "comparison": cmp_bf16(got, target)}


def drift_shape(cmp: dict[str, Any]) -> str:
    # Descriptive only.
    bins = cmp.get("ulp_histogram_bins", {})
    if cmp.get("max_bf16_ulp", 0) == 0:
        return "eliminates"
    if cmp.get("max_bf16_ulp", 0) < 196:
        return "attenuates"
    if cmp.get("max_bf16_ulp", 0) == 196:
        return "approximately_preserves_max"
    return "amplifies"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    ap.add_argument("--omlx-path", default=os.environ.get("DS41F_OMLX", str(DEFAULT_OMLX)))
    ap.add_argument("--actual-npz", default="artifacts/m4/actual-layer2-capture/actual-boundaries.npz")
    ap.add_argument("--expected-npz", default="artifacts/m4/actual-layer2-capture/expected-boundaries.npz")
    ap.add_argument("--out", default="artifacts/m4/engram1-same-input/result.json")
    args = ap.parse_args()

    ck = Path(args.checkpoint)
    omlx = Path(args.omlx_path)
    act = load_npz(ROOT / args.actual_npz)
    exp = load_npz(ROOT / args.expected_npz)

    expected_h = exp["engram1_input_h"]
    actual_h = act["engram1_input_h"]
    expected_ids = exp["engram1_hash_ids"]
    actual_ids = act["engram1_hash_ids"]

    ee = run_engram(ck, expected_h, expected_ids)
    aa = run_engram(ck, actual_h, actual_ids)

    ee_endpoint = endpoint_summary("EE_vs_Boundary13d_Engram1_output", ee["post"], exp["engram1_output_h"])
    aa_endpoint = endpoint_summary("AA_vs_actual_oMLX_Engram1_output", aa["post"], act["engram1_output_h"])
    input_cmp = cmp_bf16(actual_h, expected_h)
    connected = cmp_bf16(aa["post"], ee["post"])
    actual_vs_ideal_existing = cmp_bf16(act["engram1_output_h"], exp["engram1_output_h"])
    hids = hash_cmp(actual_ids, expected_ids)

    ee_ok = bool(ee_endpoint["comparison"]["within_contract"])
    aa_ok = bool(aa_endpoint["comparison"]["within_contract"])
    if not ee_ok:
        classification = "BOUNDARY13D_ENGRAM1_EXPECTED_GENERATION_INCONSISTENT"
        complete = False
        next_frontier = "repair Boundary13d Engram@1 expected generation"
        ok = False
    elif not hids["within_contract"]:
        classification = "ENGRAM1_HASH_ID_DIVERGENCE"
        complete = False
        next_frontier = "localize Engram@1 hash ID mismatch"
        ok = False
    elif aa_ok:
        classification = "ENGRAM1_COMPOSITIONAL_UPSTREAM_NUMERICAL_DRIFT"
        complete = True
        next_frontier = "Block1 actual-input same-input qualification"
        ok = True
    else:
        classification = "ENGRAM1_SAME_INPUT_DIVERGENCE"
        complete = False
        next_frontier = "localize Engram@1 same-input internals"
        ok = False

    rec = {
        "schema": SCHEMA,
        "checkpoint": str(ck),
        "omlx_path": str(omlx),
        "source_checkpoint_provenance": {
            "checkpoint": str(ck),
            "official_source": {
                "model_py_Engram_forward": source_identity("inference/model.py", 328, 366, ck),
                "model_py_ParallelEngramEmbedding": source_identity("inference/model.py", 288, 323, ck),
            },
            "helper": "tools.run_native_first_incremental_block0_engram1_validation.apply_engram1_dynamic",
            "covered_path": [
                "hash IDs", "sparse embedding rows", "embedding dequantization", "flatten", "FP8 WKV projection", "key/value split", "q_weight/k_weight", "normalized weighted dot", "signed sqrt gate", "sigmoid", "residual update", "BF16 output",
            ],
            "weights_not_stored_in_artifact": True,
        },
        "actual_loaded_omlx_source_identity": {
            "path": str(omlx),
            **git_identity(omlx),
            "language_py_sha256": file_sha(omlx / "omlx/patches/deepseek_v41/language.py"),
        },
        "phase_a_rerun_required": False,
        "inputs": {
            "expected_engram_input_h": arr_info(expected_h),
            "actual_engram_input_h": arr_info(actual_h),
            "expected_hash_ids": arr_info(expected_ids, values=True),
            "actual_hash_ids": arr_info(actual_ids, values=True),
            "hash_id_comparison": hids,
            "incoming_connected_drift_actual_vs_expected": input_cmp,
        },
        "endpoints": {
            "EE": arr_info(ee["post"]),
            "AA": arr_info(aa["post"]),
            "EE_vs_Boundary13d": ee_endpoint,
            "AA_vs_actual_production": aa_endpoint,
        },
        "same_input_qualification": {
            "official_engram1_reproduces_actual_output": bool(aa_ok),
            "hash_ids_exact": bool(hids["within_contract"]),
            "bf16_contract_changed": False,
            "same_input_bf16_result": aa_endpoint["comparison"],
        },
        "internal_summaries": {
            "EE_sparse_embedding": ee["sparse_embedding"],
            "EE_wkv": ee["wkv"],
            "EE_gate": {"gate_digest": ee["gate"].get("gate_digest"), "record_count": len(ee["gate"].get("source_records", []))},
            "EE_residual": ee["residual"],
            "AA_sparse_embedding": aa["sparse_embedding"],
            "AA_wkv": aa["wkv"],
            "AA_gate": {"gate_digest": aa["gate"].get("gate_digest"), "record_count": len(aa["gate"].get("source_records", []))},
            "AA_residual": aa["residual"],
        },
        "connected_trajectory_diagnostic": {
            "AA_vs_EE": connected,
            "actual_capture_vs_expected_capture": actual_vs_ideal_existing,
            "incoming_block0_output_max_bf16_ulp": int(input_cmp["max_bf16_ulp"]),
            "outgoing_engram1_output_max_bf16_ulp": int(connected["max_bf16_ulp"]),
            "max_distance_effect": drift_shape(connected),
            "note": "incoming/outgoing max-distance effect is descriptive only and not a correctness gate",
        },
        "downstream_risk_context": {
            "block0_connected_output_max_ulp": 196,
            "engram1_connected_output_max_ulp": int(connected["max_bf16_ulp"]),
            "block1_connected_output_max_ulp": 30260,
            "block1_returned_pre_max_abs": 0.0007171035,
            "note": "large growth currently appears after entering Block1; inspect Block1 with same-input methodology next if Engram@1 qualifies",
        },
        "methodology": {
            "same_input_operator_qualification": "primary correctness question for Engram@1",
            "connected_ideal_trajectory_comparison": "diagnostic only after upstream drift is established",
            "tolerance_changed": False,
        },
        "classification": classification,
        "engram1_is_operation_defect": bool(not ok),
        "engram1_same_input_correctness": "COMPLETE" if complete else "INCOMPLETE",
        "engram1_same_input_complete": bool(complete),
        "next_frontier": next_frontier,
        "ok": bool(ok),
    }

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print(out)
    print("classification", classification, "ok", ok)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
