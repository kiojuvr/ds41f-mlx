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
from tools.run_m4_block1_remainder_and_layer2_entry import ordered_bf16
from tools.run_native_layer0_25_transformer_entry_validation import DIM, cfg
from tools.run_official_hyper_connections_fixture import bf16_to_f32, f32_to_bf16, hc_pre, mmap, shard
from tools.source_identity import source_identity

SCHEMA = "ds41f.m4.block0-ffn-pre-norm-composition.v1"


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def arr_info(a: np.ndarray, *, values: bool = False) -> dict[str, Any]:
    out = {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": sha(a)}
    if values:
        out["values"] = a.tolist()
    return out


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def rms_eps(x_bf16: np.ndarray, w_bf16: np.ndarray, eps: float) -> np.ndarray:
    x = bf16_to_f32(x_bf16)
    w = bf16_to_f32(w_bf16)
    var = np.mean(np.square(x, dtype=np.float32), axis=-1, keepdims=True, dtype=np.float32)
    y = x * (1.0 / np.sqrt(var + np.float32(eps), dtype=np.float32)) * w
    return f32_to_bf16(y.astype(np.float32))


def ffn_pre_norm(h_bf16: np.ndarray, pre_f32: np.ndarray, weight_bf16: np.ndarray, eps: float) -> np.ndarray:
    return rms_eps(hc_pre(h_bf16, pre_f32), weight_bf16, eps)


def cmp_bf16(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if a.shape != b.shape or a.dtype != np.uint16 or b.dtype != np.uint16:
        return {"a": arr_info(a), "b": arr_info(b), "within_contract": False, "reason": "shape/dtype mismatch"}
    ulp = np.abs(ordered_bf16(a) - ordered_bf16(b))
    af = bf16_to_f32(a)
    bf = bf16_to_f32(b)
    diff = np.abs(af - bf)
    unique, counts = np.unique(ulp, return_counts=True)
    hist = {str(int(k)): int(v) for k, v in zip(unique, counts)}
    return {
        "a": arr_info(a),
        "b": arr_info(b),
        "exact": bool(np.array_equal(a, b)),
        "max_bf16_ulp": int(ulp.max()) if ulp.size else 0,
        "num_elements": int(ulp.size),
        "num_differing": int(np.count_nonzero(ulp)),
        "num_gt_1_ulp": int(np.count_nonzero(ulp > 1)),
        "ulp_histogram": hist,
        "ulp_histogram_bins": {
            "0": int(np.count_nonzero(ulp == 0)),
            "1": int(np.count_nonzero(ulp == 1)),
            "2": int(np.count_nonzero(ulp == 2)),
            ">2": int(np.count_nonzero(ulp > 2)),
        },
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "contract": "BF16 max ULP <= 1",
        "within_contract": bool((int(ulp.max()) if ulp.size else 0) <= 1),
    }


def cmp_f32(a: np.ndarray, b: np.ndarray, tol: float = 1e-4) -> dict[str, Any]:
    d = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return {
        "a": arr_info(a),
        "b": arr_info(b),
        "exact": bool(np.array_equal(a, b)),
        "max_abs_diff": float(d.max()) if d.size else 0.0,
        "mean_abs_diff": float(d.mean()) if d.size else 0.0,
        "contract": f"FP32 max abs <= {tol}",
        "within_contract": bool((float(d.max()) if d.size else 0.0) <= tol),
    }


def file_sha(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return None


def git_identity(path: Path) -> dict[str, Any]:
    try:
        rev = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(path), "status", "--short"], text=True).strip())
        return {"git_head": rev, "dirty": dirty}
    except Exception as exc:
        return {"error": repr(exc)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    ap.add_argument("--omlx-path", default=os.environ.get("DS41F_OMLX", str(DEFAULT_OMLX)))
    ap.add_argument("--actual-npz", default="artifacts/m4/actual-layer2-capture/actual-boundaries.npz")
    ap.add_argument("--expected-npz", default="artifacts/m4/actual-layer2-capture/expected-boundaries.npz")
    ap.add_argument("--out", default="artifacts/m4/block0-ffn-pre-norm-composition/result.json")
    args = ap.parse_args()

    ck = Path(args.checkpoint)
    omlx = Path(args.omlx_path)
    act = load_npz(ROOT / args.actual_npz)
    exp = load_npz(ROOT / args.expected_npz)
    c = cfg(ck)
    eps = float(c["rms_norm_eps"])
    weight = np.ascontiguousarray(mmap(shard(ck, "layers.0.ffn_norm.weight"), "layers.0.ffn_norm.weight", np.uint16, (DIM,)))

    expected_h = exp["block0_post_attention_h"]
    actual_h = act["block0_post_attention_h"]
    expected_pre = exp["block0_attn_ap"]
    actual_pre = act["block0_attn_ap"]

    outs = {
        "EE": ffn_pre_norm(expected_h, expected_pre, weight, eps),
        "AE": ffn_pre_norm(actual_h, expected_pre, weight, eps),
        "EA": ffn_pre_norm(expected_h, actual_pre, weight, eps),
        "AA": ffn_pre_norm(actual_h, actual_pre, weight, eps),
    }

    current_expected = exp["block0_ffn_pre_norm_output"]
    current_actual = act["block0_ffn_pre_norm_output"]
    same_input_expected_ok = cmp_bf16(outs["EE"], current_expected)["within_contract"]
    same_input_actual_ok = cmp_bf16(outs["AA"], current_actual)["within_contract"]
    aa_vs_ee = cmp_bf16(outs["AA"], outs["EE"])

    if same_input_expected_ok and same_input_actual_ok:
        classification = "COMPOSITIONAL_UPSTREAM_NUMERICAL_DRIFT" if not aa_vs_ee["within_contract"] else "FFN_PRE_NORM_OPERATOR_AND_TRAJECTORY_QUALIFY"
        operator_defect = False
    elif not same_input_expected_ok:
        classification = "BOUNDARY13D_EXPECTED_GENERATION_INCONSISTENT"
        operator_defect = None
    else:
        classification = "FFN_PRE_NORM_SAME_INPUT_OPERATOR_DIVERGENCE"
        operator_defect = True

    ae = cmp_bf16(outs["AE"], outs["EE"])
    ea = cmp_bf16(outs["EA"], outs["EE"])
    contribution = {
        "post_attention_h_only_AE_vs_EE": ae,
        "attn_ap_only_EA_vs_EE": ea,
        "combined_AA_vs_EE": aa_vs_ee,
        "interpretation": "AE isolates actual post_attention_h with expected attn_ap; EA isolates actual attn_ap with expected post_attention_h. These are diagnostics only and do not assign fault.",
    }
    if ae["num_gt_1_ulp"] and not ea["num_gt_1_ulp"]:
        contribution["dominant_gt1_source"] = "post_attention_h"
    elif ea["num_gt_1_ulp"] and not ae["num_gt_1_ulp"]:
        contribution["dominant_gt1_source"] = "attn_ap"
    elif ae["num_gt_1_ulp"] and ea["num_gt_1_ulp"]:
        contribution["dominant_gt1_source"] = "both_individually"
    else:
        contribution["dominant_gt1_source"] = "combination_or_none_gt1_in_single_parent_counterfactuals"

    rec = {
        "schema": SCHEMA,
        "checkpoint": str(ck),
        "omlx_path": str(omlx),
        "source_checkpoint_provenance": {
            "checkpoint": str(ck),
            "ffn_norm_weight": arr_info(weight),
            "rms_norm_eps": eps,
            "official_source": {
                "model_py_Block_forward": source_identity("inference/model.py", 971, 994, ck),
                "model_py_RMSNorm": source_identity("inference/model.py", 215, 224, ck),
            },
            "operator": "N(h, pre) = RMSNorm(hc_pre(h, pre), layers.0.ffn_norm.weight, rms_norm_eps)",
            "precision": "BF16 h, FP32 HC pre weights/accumulation via existing official helper hc_pre, RMSNorm in FP32, BF16 RNE output",
        },
        "actual_loaded_omlx_source_identity": {
            "path": str(omlx),
            **git_identity(omlx),
            "language_py_sha256": file_sha(omlx / "omlx/patches/deepseek_v41/language.py"),
        },
        "inputs": {
            "expected_h_boundary13d_post_attention_h": arr_info(expected_h),
            "actual_h_captured_post_attention_h": arr_info(actual_h),
            "expected_attn_ap_boundary13d": arr_info(expected_pre),
            "actual_attn_ap_captured": arr_info(actual_pre),
            "direct_parent_comparisons": {
                "post_attention_h_actual_vs_expected": cmp_bf16(actual_h, expected_h),
                "attn_ap_actual_vs_expected": cmp_f32(actual_pre, expected_pre, 1e-4),
            },
        },
        "outputs": {k: arr_info(v) for k, v in outs.items()},
        "decisive_checks": {
            "EE_vs_current_boundary13d_expected": cmp_bf16(outs["EE"], current_expected),
            "AA_vs_captured_actual_production": cmp_bf16(outs["AA"], current_actual),
        },
        "counterfactual_comparisons": {
            "AE_vs_EE": ae,
            "EA_vs_EE": ea,
            "AA_vs_EE": aa_vs_ee,
        },
        "contribution_diagnostic": contribution,
        "methodology": {
            "same_input_operator_qualification": "Given the same semantically valid inputs, compare operator output to production/reference output under the existing reviewed contract.",
            "connected_trajectory_comparison": "After prior qualified numerical operations, actual trajectory drift vs ideal trajectory is diagnostic and is not by itself a new operation defect if same-input semantics reproduce both endpoints.",
            "bf16_tolerance_changed": False,
            "bf16_contract_remains": "max ULP <= 1 for same-input/operator boundaries",
        },
        "classification": classification,
        "operator_defect": operator_defect,
        "block0_ffn_pre_norm_divergence_remains_valid": bool(classification not in {"COMPOSITIONAL_UPSTREAM_NUMERICAL_DRIFT", "FFN_PRE_NORM_OPERATOR_AND_TRAJECTORY_QUALIFY"}),
        "next_frontier": "actual-input MoE same-input qualification" if classification == "COMPOSITIONAL_UPSTREAM_NUMERICAL_DRIFT" else "resolve FFN pre-norm same-input/expected-generation inconsistency",
        "ok": bool(classification == "COMPOSITIONAL_UPSTREAM_NUMERICAL_DRIFT"),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print(out)
    print("classification", classification, "ok", rec["ok"])
    return 0 if rec["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
