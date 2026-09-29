#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/m4/engram1-same-input/result.json"

def main() -> int:
    assert ART.exists(), ART
    r = json.loads(ART.read_text())
    assert r["schema"] == "ds41f.m4.engram1-same-input.v1"
    assert r["classification"] == "ENGRAM1_COMPOSITIONAL_UPSTREAM_NUMERICAL_DRIFT"
    assert r["ok"] is True
    assert r["phase_a_rerun_required"] is False
    assert r["methodology"]["tolerance_changed"] is False
    assert r["inputs"]["hash_id_comparison"]["exact"] is True
    assert r["endpoints"]["EE_vs_Boundary13d"]["comparison"]["exact"] is True
    assert r["endpoints"]["EE_vs_Boundary13d"]["comparison"]["max_bf16_ulp"] == 0
    assert r["endpoints"]["AA_vs_actual_production"]["comparison"]["exact"] is True
    assert r["endpoints"]["AA_vs_actual_production"]["comparison"]["max_bf16_ulp"] == 0
    assert r["same_input_qualification"]["official_engram1_reproduces_actual_output"] is True
    assert r["same_input_qualification"]["hash_ids_exact"] is True
    assert r["connected_trajectory_diagnostic"]["incoming_block0_output_max_bf16_ulp"] == 196
    assert r["connected_trajectory_diagnostic"]["outgoing_engram1_output_max_bf16_ulp"] == 48
    assert r["connected_trajectory_diagnostic"]["max_distance_effect"] == "attenuates"
    assert r["engram1_is_operation_defect"] is False
    assert r["engram1_same_input_correctness"] == "COMPLETE"
    assert r["engram1_same_input_complete"] is True
    assert r["next_frontier"] == "Block1 actual-input same-input qualification"
    print(f"Engram@1 same-input check PASS: {ART}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
