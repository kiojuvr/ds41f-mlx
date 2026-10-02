#!/usr/bin/env python3
"""Validate M28 raw evidence and emit its scoped decision; no MLX import."""
import collections
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = [ROOT/"artifacts/m28"/f"{name}-boundaries.json" for name in ("code", "synthetic")]
    documents = [json.loads(p.read_text()) for p in paths]
    cases = [c for j in documents for c in j["cases"]]
    for j in documents:
        assert j["status"] == "COMPLETED_DIAGNOSTIC"
        assert j["omlx_revision"] == "4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40"
        assert j["totals"]["history_replay_tokens"] == j["totals"]["repack_calls"] == j["totals"]["reconcile_attempts"] == 0
        assert j["window_size"] == 128
        probe = ROOT/"tools/run_m28_mtp_quiesce.py"
        assert j["source_sha256"][str(probe)] == hashlib.sha256(probe.read_bytes()).hexdigest()
    for c in cases:
        assert c["status"] == "PASS", (c.get("error"), c["context"], c["interrupt_after"])
        n = c["needed_commits"]
        assert -1 <= n <= 5
        assert len(c["before"]["queue"]) == n+1
        delta = c["counter_delta"]
        for name in ("verify_cycles", "dspark_proposals", "history_replay_tokens", "cache_constructions", "reconcile_attempts", "repack_calls"):
            assert delta[name] == 0
        assert delta["target_forwards"] == delta["dspark_appends"] == int(n == -1)
        after = c["after"]
        h = len(after["canonical_history"])
        assert after["target"] == [h]*40
        assert after["dspark"] == [h]*3
        assert h == c["before"]["history_frontier"]+max(n, 0)
        assert len(after["extra_server_committed_tokens_NOT_delivered"]) == max(n, 0)
        assert not after["rollback_stash_present"] and not after["scheduler_uids"] and after["queue_len"] == 0
        if n >= 0:
            assert c["committed_queue_tokens_match_execution_ledger"]
            assert c["dspark_ring_unchanged_during_drain"]
            assert len(after["discarded_uncommitted_queue"]) == 1
        for i, step in enumerate(c["ordinary_target_continuation"], 1):
            assert step["target"] == [h+i]*40 and step["dspark"] == [h+i]*3
        if c["mode"] in ("stop", "stop_sequence"):
            assert c["topology"][-1]["finish"] == "stop"
        if c["mode"] == "length":
            assert c["topology"][-1]["finish"] == "length"
    acceptance = collections.Counter((cy["depth"], cy["accepted_final"]) for c in cases for cy in c["cycles"])
    assert any(k == m == 5 for k, m in acceptance)
    assert any(0 < m < k for k, m in acceptance)
    assert any(m == 0 for k, m in acceptance)
    assert {c["needed_commits"] for c in cases} == {-1, 0, 1, 2, 3, 4}
    latency = [c["quiescence_seconds"]*1000 for c in cases if c["needed_commits"] >= 0]
    lag_latency = [c["quiescence_seconds"]*1000 for c in cases if c["needed_commits"] < 0]
    summary = {
        "schema": "ds41f.m28.quiesce-decision.v1",
        "decision": "IMMEDIATE_ABORT_ONLY_BLOCKED",
        "canonical_quiescence_viable": True,
        "scope": "native singleton DeepSeek-V4.1 DSpark, depth<=5, commit_align=0, matcher-known token stops and max_tokens; NOT serving qualification",
        "production_mtp": "OFF_UNCHANGED",
        "base_commit": documents[0]["base_commit"],
        "omlx_revision": documents[0]["omlx_revision"],
        "packages": documents[0]["packages"],
        "checkpoint": documents[0]["checkpoint"],
        "config_sha256": documents[0]["config_sha256"],
        "checkpoint_index_sha256": documents[0]["checkpoint_index_sha256"],
        "device": documents[0]["device"],
        "window_size": 128,
        "invariant": "At cycle commit B=H-1; verify retains 1+m input positions, T=H+m, Q=[accepted drafts[:m], final unforwarded correction/bonus]. After r emissions from that queue, T-H=m-r and len(Q)=T-H+1. First max(T-H,0) queued tokens equal actual target input ledger positions [H,T). DSpark offset equals T.",
        "bound": {"before_first_cycle_emission": "needed<=m<=configured_depth (5)", "external_post_response_bound": "max(0,needed)<=depth-1 (4); post-init first response needs zero", "observed_extra_tokens_max": max(len(c["after"]["extra_server_committed_tokens_NOT_delivered"]) for c in cases)},
        "algorithm": "Freeze verification/drafting. For T>=H, validate queue/frontier/matcher/ledger contract and emit exactly T-H existing queue entries via _emit_response (never _mtp_next). Discard final not-forwarded token. For T=H-1 and Q empty, forward last already-emitted token once with hidden capture and append to DSpark. Clear stashes/queues, native row extract before removing UID. No sampling or new speculative cycle.",
        "results": {"cases_passed": len(cases), "cases_failed": 0,
                    "code_cases": len(documents[0]["cases"]), "synthetic_cases": len(documents[1]["cases"]),
                    "initial_contexts": sorted({c["context"] for c in cases}),
                    "queue_lengths": sorted({len(c["before"]["queue"]) for c in cases}),
                    "needed_values": sorted({c["needed_commits"] for c in cases}),
                    "owning_cycle_acceptance_shapes": [{"depth": k, "accepted": m, "observations_including_repeated_runs": count} for (k, m), count in sorted(acceptance.items())],
                    "cache_ahead_new_verifies": 0, "cache_ahead_target_forwards": 0, "history_replay_tokens": 0, "full_cache_repack": 0,
                    "ordinary_target_continuation_passed": len(cases),
                    "drain_quiescence_ms": {"median": statistics.median(latency), "max": max(latency)},
                    "lagging_quiescence_ms": {"median": statistics.median(lag_latency), "max": max(lag_latency)}},
        "stop_safety": "Upstream clamps final m with min(remaining-1, first matcher-hit index) before target rollback and DSpark append. Stop-completing/length-final token remains unforwarded. Copied immutable-trie matcher and length checks independently guard draining; 1-token and 3-token stop sequences plus length 1/2/3/7/16 passed.",
        "limitations": [
            "Immediate token-exact abort remains M27-blocked for target-ahead states. Canonical recovery must report extra server-committed output separately from transport-delivered output.",
            "Actual recipe EOS/tool generation not induced. Configured token matcher semantics exercised, including a three-token stop. Boundaries enforced only by external text parsers/tool detection are unsupported: fail closed before drain unless registered into the cycle matcher or equivalently clamped before target commit.",
            "No P7/P6 priming seam, HTTP/Rust recovery API, MTP re-entry, processors/thinking budgets, batching, commit-alignment overrides, or persistence qualification. Ordinary target continuation only; integration deferred.",
            "Execution ledger is diagnostic evidence, not a proposed second executable target authority. A future adapter must enforce the pinned state-machine contract and supported protocol before mutation.",
            "Quiescence latency includes cleanup/extraction instrumentation, excludes subsequent continuation and model load. No broad throughput benchmark performed."
        ],
        "regressions": {"production_unittest": "108 tests PASS; artifacts/m28/production-regression.log", "production_runtime_changed": False, "self_containment": "artifacts/m28/self-containment.log", "syntax": "py_compile run_m28_mtp_quiesce.py / check_m28_quiesce.py"},
        "evidence_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "next_milestone": "Integrate full-quality P7/P6 DSpark taps plus canonical-session MTP lifecycle and recovery, then operational qualification; no retained rollback snapshot project needed for this semantic model."
    }
    target = ROOT/"artifacts/m28/decision.json"
    target.write_text(json.dumps(summary, indent=2)+"\n")
    print(f"PASS {len(cases)} native MTP quiescence/continuation cases; decision={summary['decision']}")


if __name__ == "__main__": main()
