#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/m4/actual-layer2-capture"
COMPLETE = BASE / "completed-comparison.json"
DIAG = BASE / "corrected-attempt-diagnostic.json"
FAILED = BASE / "result.json"
PROGRESS = BASE / "phase-a-progress.jsonl"

def main():
    assert FAILED.exists(), "historical exit137 diagnostic artifact must be preserved"
    failed = json.loads(FAILED.read_text())
    assert failed["attempt_result"]["exit_code"] == 137
    assert failed["layer2_incremental_block"] == "INCOMPLETE"
    if COMPLETE.exists():
        r = json.loads(COMPLETE.read_text())
        assert r["schema"] == "ds41f.m4.actual-layer2-capture.v2"
        assert r["attempt_status"] == "COMPLETE"
        assert r["historical_failed_attempt_preserved"] is True
        assert r["classification"] in {"LAYER2_ENTRY_DIVERGENCE", "BLOCK0_ENTRY_DIVERGENCE", "BLOCK0_EXECUTION_DIVERGENCE", "ENGRAM1_HASH_DIVERGENCE", "ENGRAM1_EXECUTION_DIVERGENCE", "BLOCK1_ENTRY_DIVERGENCE", "BLOCK1_EXECUTION_DIVERGENCE", "UPSTREAM_PREFIX_COMPLETE", "SPARSE_OUTPUT_DIVERGENCE", "INVERSE_ROPE_DIVERGENCE", "INVERSE_ROPE_COMPLETE / NEXT PROJECTION FRONTIER"}
        assert r.get("execution_order_classification") == r["classification"]
        assert "upstream_comparisons" in r
        assert "internal_identity_actual_block1_exit_eq_layer2_entry" in r
        for k in ["pre_inverse_rope", "inverse_rope", "attention_return", "post_attention_hc", "moe_input", "route_ids", "route_weights", "moe_output", "block_output_h", "block_returned_pre"]:
            assert k in r["comparisons"], k
            assert r["comparisons"][k]["actual"] is not None, k
        assert all(r["post_layer2_state"].values())
        if r["ok"]:
            assert r["layer2_incremental_block"] == "COMPLETE"
            assert all(v["within_contract"] for v in r["comparisons"].values())
            print(f"Completed actual Layer2 capture qualified: {COMPLETE}")
        else:
            assert r["layer2_incremental_block"] == "INCOMPLETE"
            assert r["first_unresolved_boundary"]
            if r["classification"] == "LAYER2_ENTRY_DIVERGENCE":
                assert r["first_unresolved_boundary"] in {"layer2_entry_h", "layer2_entry_pre"}
            print(f"Completed actual Layer2 capture recorded with mismatch: {COMPLETE}")
        return
    assert DIAG.exists(), "neither completed capture nor corrected-attempt diagnostic exists"
    r = json.loads(DIAG.read_text())
    assert r["schema"] == "ds41f.m4.actual-layer2-capture.v2"
    assert r["attempt_status"] == "KILLED"
    assert r["exit_code"] == 137
    assert r["classification"] in {"REAL MODEL LOAD MEMORY PRESSURE", "PREFILL-PLUS-MODEL RESIDENCY STILL OVERLAPS", "CAPTURE MATERIALIZATION MEMORY PRESSURE", "FORWARD EXECUTION MEMORY PRESSURE"}
    assert r["corrected_harness_properties"]["phase_a_excludes_independent_source_reconstruction"] is True
    assert r["model_load_completed"] is False
    assert r["admission_completed"] is False
    assert r["forward_began"] is False
    assert PROGRESS.exists()
    stages = [json.loads(l)["stage"] for l in PROGRESS.read_text().splitlines() if l.strip()]
    for s in ["process_start", "prefill_state_begin", "prefill_state_complete", "prefill_scaffolding_released", "omlx_runtime_construct_begin", "omlx_runtime_construct_complete", "model_load_begin"]:
        assert s in stages, s
    assert stages[-1] == r["last_durable_stage_marker"]["stage"] == "model_load_begin"
    print(f"Corrected actual Layer2 capture diagnostic recorded (killed): {DIAG}")

if __name__ == "__main__":
    main()
