#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT, DEFAULT_OMLX
from tools.run_m4_block0_ffn_pre_norm_composition import arr_info, cmp_bf16, cmp_f32, file_sha, git_identity
from tools.run_native_layer24_25_connected_validation import get_cfg, moe_layer
from tools.source_identity import source_identity

SCHEMA = "ds41f.m4.block0-moe-same-input.v1"


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def align_like(a: np.ndarray, target: np.ndarray) -> np.ndarray:
    if a.shape == target.shape:
        return a
    s = np.squeeze(a)
    if s.shape == target.shape:
        return s
    if a.size == target.size:
        return a.reshape(target.shape)
    return a


def route_cmp(ids_a: np.ndarray, ids_b: np.ndarray, weights_a: np.ndarray, weights_b: np.ndarray) -> dict[str, Any]:
    ids_a = align_like(ids_a, ids_b)
    weights_a = align_like(weights_a, weights_b)
    return {
        "route_ids": {
            "a": arr_info(ids_a, values=True),
            "b": arr_info(ids_b, values=True),
            "exact": bool(np.array_equal(ids_a, ids_b)),
            "within_contract": bool(np.array_equal(ids_a, ids_b)),
            "contract": "route IDs exact",
        },
        "route_weights": cmp_f32(weights_a.astype(np.float32), weights_b.astype(np.float32), 1e-3),
    }


def moe_summary(m: dict[str, Any]) -> dict[str, Any]:
    per = []
    for r in m["per"]:
        per.append({k: r[k] for k in ["token", "topk_slot", "expert_id", "routing_weight_f32", "expert_output_before_weight_digest", "weighted_expert_contribution_digest"]})
    return {
        "raw_gate_projection": arr_info(np.ascontiguousarray(m["raw"])),
        "route_ids": arr_info(np.ascontiguousarray(m["idx"]), values=True),
        "route_weights": arr_info(np.ascontiguousarray(m["weights"]), values=True),
        "selected_expert_set": [int(x) for x in m["expert_ids"]],
        "selected_expert_count": len(m["expert_ids"]),
        "selected_expert_contributions": per,
        "routed_sum": arr_info(np.ascontiguousarray(m["routed"])),
        "shared_expert_output": arr_info(np.ascontiguousarray(m["shared"])),
        "final_moe_output": arr_info(np.ascontiguousarray(m["final"])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    ap.add_argument("--omlx-path", default=os.environ.get("DS41F_OMLX", str(DEFAULT_OMLX)))
    ap.add_argument("--actual-npz", default="artifacts/m4/actual-layer2-capture/actual-boundaries.npz")
    ap.add_argument("--expected-npz", default="artifacts/m4/actual-layer2-capture/expected-boundaries.npz")
    ap.add_argument("--out", default="artifacts/m4/block0-moe-same-input/result.json")
    args = ap.parse_args()

    ck = Path(args.checkpoint)
    omlx = Path(args.omlx_path)
    cfg = get_cfg(ck)
    act = load_npz(ROOT / args.actual_npz)
    exp = load_npz(ROOT / args.expected_npz)

    ee = moe_layer(ck, cfg, 0, exp["block0_moe_input"])
    aa = moe_layer(ck, cfg, 0, act["block0_moe_input"])

    ee_route = route_cmp(ee["idx"], exp["block0_route_ids"], ee["weights"], exp["block0_route_weights"])
    aa_route = route_cmp(aa["idx"], act["block0_route_ids"], aa["weights"], act["block0_route_weights"])
    ee_final = cmp_bf16(ee["final"], exp["block0_moe_output"])
    aa_final = cmp_bf16(aa["final"], act["block0_moe_output"])
    aa_vs_ee = cmp_bf16(aa["final"], ee["final"])

    ee_ok = ee_route["route_ids"]["within_contract"] and ee_route["route_weights"]["within_contract"] and ee_final["within_contract"]
    aa_ok = aa_route["route_ids"]["within_contract"] and aa_route["route_weights"]["within_contract"] and aa_final["within_contract"]

    if not ee_ok:
        classification = "BOUNDARY13D_MOE_EXPECTED_GENERATION_INCONSISTENT"
        next_frontier = "repair Boundary13d MoE expected generation"
        ok = False
    elif aa_ok:
        classification = "MOE_COMPOSITIONAL_UPSTREAM_NUMERICAL_DRIFT"
        next_frontier = "Block0 final hc_post same-input qualification"
        ok = True
    elif not aa_route["route_ids"]["within_contract"]:
        classification = "BLOCK0_MOE_ROUTING_ID_DIVERGENCE"
        next_frontier = "localize same-input MoE routing IDs"
        ok = False
    elif not aa_route["route_weights"]["within_contract"]:
        classification = "BLOCK0_MOE_ROUTING_WEIGHT_DIVERGENCE"
        next_frontier = "localize same-input MoE routing weights"
        ok = False
    else:
        classification = "BLOCK0_MOE_OUTPUT_DIVERGENCE"
        next_frontier = "capture/localize same-input routed/shared MoE internals"
        ok = False

    rec = {
        "schema": SCHEMA,
        "checkpoint": str(ck),
        "omlx_path": str(omlx),
        "source_checkpoint_provenance": {
            "checkpoint": str(ck),
            "official_source": {
                "model_py_Gate_MoE": source_identity("inference/model.py", 827, 912, ck),
                "model_py_Expert": source_identity("inference/model.py", 790, 826, ck),
            },
            "helper": "tools.run_native_layer24_25_connected_validation.moe_layer reused for layer=0 same-input reconstruction",
            "selection_semantics": "route IDs selected from scores + bias; returned route weights gathered from un-biased scores, normalized, then route-scaled",
            "weights_not_stored_in_artifact": True,
        },
        "actual_loaded_omlx_source_identity": {
            "path": str(omlx),
            **git_identity(omlx),
            "language_py_sha256": file_sha(omlx / "omlx/patches/deepseek_v41/language.py"),
            "decode_topology": "sequence length 1: unsorted routed experts branch, row-local quantization if projections quantize input; not x.shape[1] >= 32 / idx.size >= 64 sorted branch",
        },
        "phase_a_rerun_required": False,
        "inputs": {
            "expected_boundary13d_moe_input": arr_info(exp["block0_moe_input"]),
            "actual_captured_moe_input": arr_info(act["block0_moe_input"]),
            "actual_vs_expected_moe_input": cmp_bf16(act["block0_moe_input"], exp["block0_moe_input"]),
        },
        "EE_endpoint_official_expected_input": {
            "summary": moe_summary(ee),
            "routing_vs_boundary13d": ee_route,
            "final_output_vs_boundary13d": ee_final,
            "reproduces_boundary13d": bool(ee_ok),
        },
        "AA_endpoint_official_actual_input": {
            "summary": moe_summary(aa),
            "routing_vs_actual_omlx_capture": aa_route,
            "final_output_vs_actual_omlx_capture": aa_final,
            "reproduces_actual_omlx": bool(aa_ok),
        },
        "same_input_qualification": {
            "route_ids_exact": bool(aa_route["route_ids"]["within_contract"]),
            "route_weights_within_contract": bool(aa_route["route_weights"]["within_contract"]),
            "routed_experts_covered_by_final_exact_reproduction": bool(aa_final["within_contract"]),
            "shared_expert_covered_by_final_exact_reproduction": bool(aa_final["within_contract"]),
            "final_moe_within_contract": bool(aa_final["within_contract"]),
            "note": "Existing capture records actual route IDs/weights/final MoE output, not separate actual routed/shared tensors. Because official same-input final output reproduces actual production exactly with matching routing, routed/shared/merge are qualified as a composed same-input MoE boundary without another Phase A run.",
        },
        "connected_trajectory_diagnostic": {
            "AA_official_actual_input_vs_EE_official_expected_input": aa_vs_ee,
            "classification": "trajectory drift only; not used to widen tolerance or declare same-input MoE defect",
        },
        "methodology": {
            "same_input_operator_qualification": "primary correctness question for MoE",
            "connected_ideal_trajectory_comparison": "diagnostic only after upstream drift is established",
            "tolerance_changed": False,
        },
        "classification": classification,
        "current_32ulp_difference_is_moe_defect": bool(not ok),
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
