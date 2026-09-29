#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/m4/block0-final-hc-post-same-input/result.json"

def main() -> int:
    assert ART.exists(), ART
    r = json.loads(ART.read_text())
    assert r["schema"] == "ds41f.m4.block0-final-hc-post-same-input.v1"
    assert r["classification"] == "FINAL_HC_POST_COMPOSITIONAL_UPSTREAM_NUMERICAL_DRIFT"
    assert r["ok"] is True
    assert r["phase_a_rerun_required"] is False
    assert r["methodology"]["tolerance_changed"] is False
    assert r["endpoints"]["EE_vs_Boundary13d"]["comparison"]["exact"] is True
    assert r["endpoints"]["EE_vs_Boundary13d"]["comparison"]["max_bf16_ulp"] == 0
    assert r["endpoints"]["AA_vs_actual_production"]["comparison"]["exact"] is True
    assert r["endpoints"]["AA_vs_actual_production"]["comparison"]["max_bf16_ulp"] == 0
    assert r["same_input_qualification"]["official_hc_post_reproduces_actual_block0_output"] is True
    assert r["connected_trajectory_diagnostic"]["AA_vs_EE"]["max_bf16_ulp"] == 196
    assert r["final_hc_post_is_operation_defect"] is False
    assert r["block0_same_input_correctness"] == "COMPLETE"
    assert r["block0_same_input_complete"] is True
    assert r["next_frontier"] == "Engram@1 actual-input same-input qualification"
    print(f"Block0 final HC post same-input check PASS: {ART}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
