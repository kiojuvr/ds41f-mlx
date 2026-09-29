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
from tools.run_official_hyper_connections_fixture import hc_post
from tools.source_identity import source_identity

SCHEMA = "ds41f.m4.block0-final-hc-post-same-input.v1"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def post(x: np.ndarray, residual: np.ndarray, post_mix: np.ndarray, comb: np.ndarray) -> np.ndarray:
    return hc_post(x, residual, post_mix, comb)


def endpoint_summary(name: str, output: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    return {"name": name, "output": arr_info(output), "target": arr_info(target), "comparison": cmp_bf16(output, target)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    ap.add_argument("--omlx-path", default=os.environ.get("DS41F_OMLX", str(DEFAULT_OMLX)))
    ap.add_argument("--actual-npz", default="artifacts/m4/actual-layer2-capture/actual-boundaries.npz")
    ap.add_argument("--expected-npz", default="artifacts/m4/actual-layer2-capture/expected-boundaries.npz")
    ap.add_argument("--out", default="artifacts/m4/block0-final-hc-post-same-input/result.json")
    args = ap.parse_args()

    ck = Path(args.checkpoint)
    omlx = Path(args.omlx_path)
    act = load_npz(ROOT / args.actual_npz)
    exp = load_npz(ROOT / args.expected_npz)

    ex = {
        "moe": exp["block0_moe_output"],
        "residual": exp["block0_post_attention_h"],
        "fo": exp["block0_ffn_fo"],
        "fc": exp["block0_ffn_fc"],
        "xout": exp["block0_exit_h"],
    }
    ac = {
        "moe": act["block0_moe_output"],
        "residual": act["block0_post_attention_h"],
        "fo": act["block0_ffn_fo"],
        "fc": act["block0_ffn_fc"],
        "xout": act["block0_exit_h"],
    }

    EE = post(ex["moe"], ex["residual"], ex["fo"], ex["fc"])
    AA = post(ac["moe"], ac["residual"], ac["fo"], ac["fc"])

    one_parent = {
        "MEEE_actual_moe_only_vs_EE": cmp_bf16(post(ac["moe"], ex["residual"], ex["fo"], ex["fc"]), EE),
        "EREE_actual_residual_only_vs_EE": cmp_bf16(post(ex["moe"], ac["residual"], ex["fo"], ex["fc"]), EE),
        "EEPE_actual_post_only_vs_EE": cmp_bf16(post(ex["moe"], ex["residual"], ac["fo"], ex["fc"]), EE),
        "EEEC_actual_comb_only_vs_EE": cmp_bf16(post(ex["moe"], ex["residual"], ex["fo"], ac["fc"]), EE),
    }
    aa_vs_ee = cmp_bf16(AA, EE)
    ee_endpoint = endpoint_summary("EE_vs_Boundary13d_block0_x_out", EE, ex["xout"])
    aa_endpoint = endpoint_summary("AA_vs_captured_actual_block0_x_out", AA, ac["xout"])
    ee_ok = bool(ee_endpoint["comparison"]["within_contract"])
    aa_ok = bool(aa_endpoint["comparison"]["within_contract"])

    if not ee_ok:
        classification = "BOUNDARY13D_FINAL_HC_POST_EXPECTED_GENERATION_INCONSISTENT"
        block0_complete = False
        next_frontier = "repair Boundary13d final HC-post expected generation"
        ok = False
    elif aa_ok:
        classification = "FINAL_HC_POST_COMPOSITIONAL_UPSTREAM_NUMERICAL_DRIFT"
        block0_complete = True
        next_frontier = "Engram@1 actual-input same-input qualification"
        ok = True
    else:
        classification = "BLOCK0_FINAL_HC_POST_SAME_INPUT_DIVERGENCE"
        block0_complete = False
        next_frontier = "localize final HC-post same-input arithmetic"
        ok = False

    dominant = sorted(((k, v["max_bf16_ulp"], v["num_gt_1_ulp"]) for k, v in one_parent.items()), key=lambda x: (x[1], x[2]), reverse=True)

    rec = {
        "schema": SCHEMA,
        "checkpoint": str(ck),
        "omlx_path": str(omlx),
        "source_checkpoint_provenance": {
            "checkpoint": str(ck),
            "official_source": {
                "model_py_Block_hc_post": source_identity("inference/model.py", 965, 969, ck),
                "model_py_Block_forward": source_identity("inference/model.py", 971, 994, ck),
            },
            "helper": "tools.run_official_hyper_connections_fixture.hc_post",
            "operator": "BF16(post[...,None] * x[...,None,:] + einsum('...ij,...id->...jd', comb, residual_as_f32))",
            "weights_not_stored_in_artifact": True,
        },
        "actual_loaded_omlx_source_identity": {
            "path": str(omlx),
            **git_identity(omlx),
            "language_py_sha256": file_sha(omlx / "omlx/patches/deepseek_v41/language.py"),
            "supporting_existing_evidence": "completed-comparison shows fused_hc_post(actual inputs) equals oMLX _hc_post_reference(actual inputs) at 0 BF16 ULP",
        },
        "phase_a_rerun_required": False,
        "inputs": {
            "expected": {k: arr_info(v) for k, v in ex.items()},
            "actual": {k: arr_info(v) for k, v in ac.items()},
            "actual_vs_expected_parent_comparisons": {
                "moe_output": cmp_bf16(ac["moe"], ex["moe"]),
                "post_attention_h_residual": cmp_bf16(ac["residual"], ex["residual"]),
                "ffn_fo_post": {
                    "actual": arr_info(ac["fo"]), "expected": arr_info(ex["fo"]),
                    "note": "FP32 parent previously qualified in actual-layer2-capture completed-comparison",
                },
                "ffn_fc_comb": {
                    "actual": arr_info(ac["fc"]), "expected": arr_info(ex["fc"]),
                    "note": "FP32 parent previously qualified in actual-layer2-capture completed-comparison",
                },
            },
        },
        "endpoints": {
            "EE": arr_info(EE),
            "AA": arr_info(AA),
            "EE_vs_Boundary13d": ee_endpoint,
            "AA_vs_actual_production": aa_endpoint,
        },
        "same_input_qualification": {
            "official_hc_post_reproduces_actual_block0_output": bool(aa_ok),
            "bf16_contract_changed": False,
            "same_input_bf16_result": aa_endpoint["comparison"],
        },
        "parent_contribution_diagnostics": {
            **one_parent,
            "dominant_one_parent_by_max_ulp": dominant,
            "interpretation": "One-parent substitutions are connected-trajectory diagnostics only; they do not assign semantic fault to a parent.",
        },
        "connected_trajectory_diagnostic": {
            "AA_vs_EE": aa_vs_ee,
            "classification": "diagnostic only; not used to widen tolerance or declare same-input final HC-post defect",
        },
        "methodology": {
            "same_input_operator_qualification": "primary correctness question for final HC post",
            "connected_ideal_trajectory_comparison": "diagnostic only after upstream drift is established",
            "tolerance_changed": False,
        },
        "classification": classification,
        "final_hc_post_is_operation_defect": bool(not ok),
        "block0_same_input_correctness": "COMPLETE" if block0_complete else "INCOMPLETE",
        "block0_same_input_complete": bool(block0_complete),
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
