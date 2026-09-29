#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/m4/block0-moe-same-input/result.json"

def main() -> int:
    assert ART.exists(), ART
    r = json.loads(ART.read_text())
    assert r["schema"] == "ds41f.m4.block0-moe-same-input.v1"
    assert r["classification"] == "MOE_COMPOSITIONAL_UPSTREAM_NUMERICAL_DRIFT"
    assert r["ok"] is True
    assert r["phase_a_rerun_required"] is False
    assert r["methodology"]["tolerance_changed"] is False
    ee = r["EE_endpoint_official_expected_input"]
    aa = r["AA_endpoint_official_actual_input"]
    assert ee["reproduces_boundary13d"] is True
    assert ee["routing_vs_boundary13d"]["route_ids"]["exact"] is True
    assert ee["routing_vs_boundary13d"]["route_weights"]["max_abs_diff"] == 0.0
    assert ee["final_output_vs_boundary13d"]["max_bf16_ulp"] == 0
    assert aa["reproduces_actual_omlx"] is True
    assert aa["routing_vs_actual_omlx_capture"]["route_ids"]["exact"] is True
    assert aa["routing_vs_actual_omlx_capture"]["route_weights"]["within_contract"] is True
    assert aa["final_output_vs_actual_omlx_capture"]["max_bf16_ulp"] == 0
    q = r["same_input_qualification"]
    assert q["route_ids_exact"] and q["route_weights_within_contract"] and q["final_moe_within_contract"]
    drift = r["connected_trajectory_diagnostic"]["AA_official_actual_input_vs_EE_official_expected_input"]
    assert drift["max_bf16_ulp"] == 32
    assert r["current_32ulp_difference_is_moe_defect"] is False
    assert r["next_frontier"] == "Block0 final hc_post same-input qualification"
    print(f"Block0 MoE same-input check PASS: {ART}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
