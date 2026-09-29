#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/m4/block0-ffn-pre-norm-composition/result.json"

def main() -> int:
    assert ART.exists(), ART
    r = json.loads(ART.read_text())
    assert r["schema"] == "ds41f.m4.block0-ffn-pre-norm-composition.v1"
    assert r["classification"] == "COMPOSITIONAL_UPSTREAM_NUMERICAL_DRIFT"
    assert r["ok"] is True
    assert r["methodology"]["bf16_tolerance_changed"] is False
    assert r["block0_ffn_pre_norm_divergence_remains_valid"] is False
    ee = r["decisive_checks"]["EE_vs_current_boundary13d_expected"]
    aa = r["decisive_checks"]["AA_vs_captured_actual_production"]
    assert ee["exact"] is True and ee["max_bf16_ulp"] == 0
    assert aa["exact"] is True and aa["max_bf16_ulp"] == 0
    assert r["counterfactual_comparisons"]["AA_vs_EE"]["max_bf16_ulp"] == 2
    assert r["counterfactual_comparisons"]["AA_vs_EE"]["num_gt_1_ulp"] == 1
    assert r["counterfactual_comparisons"]["AE_vs_EE"]["max_bf16_ulp"] == 2
    assert r["counterfactual_comparisons"]["EA_vs_EE"]["max_bf16_ulp"] == 0
    assert r["contribution_diagnostic"]["dominant_gt1_source"] == "post_attention_h"
    assert r["next_frontier"] == "actual-input MoE same-input qualification"
    print(f"Block0 FFN pre-norm composition check PASS: {ART}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
