#!/usr/bin/env python3
"""Summarize diagnostic evidence without promoting blocked lifecycle gates."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("artifacts/m25"))
    args = ap.parse_args()
    root = args.root
    on, off = [json.loads((root/f"{s}.json").read_text()) for s in ("on", "off")]
    assert all(r["status"] == "COMPLETED_DIAGNOSTIC" for r in (on, off))
    assert on["omlx_revision"] == off["omlx_revision"]
    assert on["checkpoint_config_sha256"] == off["checkpoint_config_sha256"]
    assert on["packages"] == off["packages"]
    assert on["source_sha256"] == off["source_sha256"]
    comparisons = []
    for a in off["cases"]:
        b = next(c for c in on["cases"] if (c["context"], c["scenario"]) == (a["context"], a["scenario"]))
        pair = {"context": a["context"], "scenario": a["scenario"], "sides": {}}
        for side, c in (("OFF", a), ("MTP", b)):
            windows = {}
            if len(c["steps"]) == 64:
                for label, steps in (("early", c["steps"][:21]), ("mid", c["steps"][21:42]), ("late", c["steps"][42:])):
                    seconds = sum(s["seconds"] for s in steps)
                    windows[label] = {"responses": len(steps), "step_seconds": seconds, "step_tok_s": len(steps)/seconds}
            skew = [s["target_offsets"][0]-(c["context"]+1+i+1)
                    for i,s in enumerate(c["steps"]) if s.get("target_offsets")]
            pair["sides"][side] = {
                "decode_tok_s_including_probe_observations": c["decode_tok_s"],
                "synchronized_step_tok_s": len(c["steps"])/sum(s["seconds"] for s in c["steps"]),
                "first_token_s": c["first_token_seconds"], "bootstrap_s": c["bootstrap_seconds"],
                "bootstrap_plus_first_token_s": c["bootstrap_seconds"]+c["first_token_seconds"],
                "prefill_s": c["prefill_seconds"], "idle_boundary": c["idle_boundary"],
                "upstream_last_observed_stats": c.get("last_stats"),
                "upstream_counter_acceptance_rate": c.get("acceptance_rate"),
                "accepted_per_last_observed_cycle": c.get("accepted_per_cycle"),
                "early_mid_late_decode_windows_NOT_session_windows": windows,
                "observed_cache_minus_emitted_frontier_range": [min(skew), max(skew)] if skew else None,
                "observed_adaptive_depths": sorted({s["adaptive_depth"] for s in c["steps"] if s.get("adaptive_depth") is not None}),
                "memory_before": c["memory_before"], "memory_after": c["memory_after"],
                "cleanup_empty": c["cleanup_empty"], "host_prime_context_retained": c["host_prime_context_retained"]}
        pair["diagnostic_mtp_decode_ratio"] = b["decode_tok_s"]/a["decode_tok_s"]
        comparisons.append(pair)
    rollback = [e for e in on["events"] if e["event"] == "target_partial_rollback"]
    attempted_replay = [e for e in on["events"] if e["event"] == "forbidden_full_history_reconcile"]
    assert rollback and any(e["accepted"] < e["drafts"] and e["returned"] for e in rollback)
    assert all(len(e["before"]) == len(e["after"]) == 40
               and all(a-b == e["drafts"]-e["accepted"] for a,b in zip(e["before"], e["after"]))
               for e in rollback)
    assert any(not c["idle_boundary"]["coherent"] for c in on["cases"])
    assert all(c["idle_boundary"]["coherent"] for c in off["cases"])
    quick = json.loads((root/"quick-final.json").read_text())
    soak = json.loads((root/"off-operational-soak.json").read_text())
    acceptance = json.loads((root/"release-acceptance.json").read_text())
    traces = [t["diagnostics_tail"] for t in soak["turns"] if t.get("ok") and t.get("diagnostics_tail")]
    soak_pass = (soak["turn_count"] == soak["turn_target"] == 10 and soak["tool_cycles"] > 0
                 and soak["restored"] and soak["close"]["ok"]
                 and soak["cancellation"].get("server_committed_after_client_timeout")
                 and not soak["final_diagnostics"]["lock_locked"] and len(traces) == 10
                 and all(t["all_cache_offsets_equal_frontier"] and t["cache_layer_count"] == 40
                         and t["prompt_replay_count"] == t["full_cache_repack_count"] == 0 for t in traces))
    assert soak_pass and quick["status"] == "QUICK_RUNTIME_QUALIFIED" and acceptance["status"] == "PASS"
    soak_windows = {}
    for label, turns in (("early", traces[:3]), ("mid", traces[3:7]), ("late", traces[7:])):
        soak_windows[label] = {k: statistics.median(t[k] for t in turns) for k in
                               ("decode_tok_s", "first_token_latency_s", "append_seconds")}
    result = {"schema": "ds41f.m25.mtp-decision.v1", "decision": "REJECT_DEFER_MTP",
              "production_default": "OFF", "optional_serving_qualified": False,
              "summary_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "reason": "Pinned ordinary upstream extraction fails arbitrary committed idle frontier; history-replay reconciliation forbidden",
              "tested_runtime_identity": quick["tested_runtime_identity"],
              "probe_base_commit": on["base_commit"], "configuration": on["configuration"],
              "comparisons": comparisons,
              "rollback_summary": {"calls": len(rollback), "successful_partial_rejections": sum(e["returned"] and e["accepted"] < e["drafts"] for e in rollback),
                                   "failed_calls": sum(not e["returned"] for e in rollback),
                                   "all40_offset_reductions_equal_rejected_suffix": True,
                                   "forbidden_reconcile_attempts": len(attempted_replay)},
              "telemetry_limits": ["last_stats is last live-state snapshot, potentially omits final-cycle/final-response updates",
                                   "upstream depth_drafted counts tested prefix positions through rejection, not all physical draft work",
                                   "component times omit initialization and are upstream counter times, not total wall time",
                                   "memory peak is process cumulative; scenarios reuse frozen diagnostic prefix views",
                                   "synthetic token fixture and cold shape compilation are not agent-session practical benefit evidence"],
              "gates": {"P5_bootstrap": "PASS_DIAGNOSTIC_BOTH", "ordinary_decode": "PASS_DIAGNOSTIC_BOTH",
                        "actual_accept_reject_rollback": "OBSERVED_MTP", "idle_extraction": "FAIL_MTP_PASS_OFF",
                        "MTP_repeated_continuation": "BLOCKED_IDLE_PREREQUISITE", "MTP_tool_EOS": "NOT_RUN_BLOCKED",
                        "MTP_cancel_reentry": "FAIL_EXTRACTION_REENTRY_BLOCKED", "MTP_persist_restart_restore": "FAIL_CLOSED_POLICY",
                        "MTP_agent_soak": "NOT_RUN_BLOCKED", "MTP_long_session_performance": "NOT_MEASURABLE_WITH_VALID_CONTINUATION",
                        "long_context": "200K_DIAGNOSTIC_BOTH_NOT_MTP_PRODUCTION_QUALIFICATION",
                        "cleanup": "PASS_BOUNDED_PROBE", "OFF_agent_soak": "PASS_FRESH_10_TURNS_RESTART_RESTORE", "release_acceptance": acceptance["status"],
                        "quick_release": quick["status"],
                        "unrestricted_pytest": "5_PREEXISTING_REFERENCE_RUNNER_FAILURES_137_PASS",
                        "production_tests": "137_PASS_32_SUBTESTS", "rust": "7_PASS", "native_ctest": "5_PASS"},
              "off_soak_result": soak,
              "reproduction": {"python": on["executable"], "commands": [
                  "tools/run_m25_mtp_boundary_probe.py --mtp ON --contexts 2048,200000 --out <new-dir>/on.json",
                  "tools/run_m25_mtp_boundary_probe.py --mtp OFF --contexts 2048,200000 --out <new-dir>/off.json"],
                  "policy": "separate sequential processes; exact pinned environment; never overwrite canonical artifacts"},
              "evidence_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir())
                                   if p.is_file() and p.name != "decision.json"},
              "scope": "Lifecycle rejection with bounded diagnostics; no completed optional production integration or long-session MTP qualification"}
    # Keep the soak artifact separate rather than duplicating large transcript bodies.
    result["off_soak_result"] = {"artifact": "off-operational-soak.json", "passed": soak_pass,
                                 "turns": soak["turn_count"], "tool_cycles": soak["tool_cycles"],
                                 "restored": soak["restored"], "final_frontier": traces[-1]["frontier"],
                                 "cancellation": soak["cancellation"], "early_mid_late": soak_windows,
                                 "rss_range_bytes": [min(t["rss_bytes"] for t in soak["turns"]), max(t["rss_bytes"] for t in soak["turns"])]}
    (root/"decision.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
