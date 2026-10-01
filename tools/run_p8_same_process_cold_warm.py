#!/usr/bin/env python3
"""P8 same-process cold/warm qualification harness.

Loads the checkpoint once, then executes cold + warm fresh-cache requests for a
single shape family in the same process/model instance.  It does not add timing
barriers inside the production command stream; prefill timing ends at the
existing commit/P6 materialization boundaries reached by the command stream.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ds41f_mlx.native_prefill import compile_native_prefill_library, load_native_prefill_library
from ds41f_mlx.prefill_fp8_mlx import DeferredPrefillAppend, DwarfStarFP8MLXPrefillExecutorSetup, P8ExecutionOptimizer, SweepPlanner


def deterministic_tokens(n: int) -> list[int]:
    return [1 + (i % 251) for i in range(int(n))]


def summarize(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"median": None, "min": None, "max": None, "range": None}
    return {"median": float(statistics.median(xs)), "min": float(min(xs)), "max": float(max(xs)), "range": float(max(xs) - min(xs))}


def run_family(args: argparse.Namespace) -> dict:
    import mlx.core as mx
    from omlx.patches.deepseek_v41.loading import load

    report = {
        "schema": "ds41f.p8.same_process_cold_warm.v1",
        "checkpoint": str(args.checkpoint),
        "tokens": int(args.tokens),
        "sequence_policy": "single shape family per process recommended; cold is first request after model load",
        "fresh_cache_each_request": True,
        "p7_enabled": True,
        "runs": [],
    }
    model = load(args.checkpoint, preserve_mtp=False, engram_ssd_offload=True)[0]
    try:
        lm = model.language_model
        lm._p7_enable_overlap = True
        with tempfile.TemporaryDirectory(prefix="ds41f-p8-coldwarm-") as tmp:
            planner = SweepPlanner(load_native_prefill_library(compile_native_prefill_library(Path(tmp))))
            plan = planner.build_from_model(lm, ctx=max(32768, int(args.tokens)), remaining=int(args.tokens))
        tokens = deterministic_tokens(args.tokens)
        for i in range(args.warm_repeats + 1):
            label = "cold" if i == 0 else f"warm{i}"
            print(json.dumps({"event": "progress", "mode": f"p8-{args.tokens}", "count": int(args.tokens), "stage": f"start-{label}", "elapsed_wall_s": 0.0}), flush=True)
            p8 = P8ExecutionOptimizer(enabled=bool(args.p8_verify), mx=mx)
            lm._p8_optimizer = p8
            setup_t0 = time.perf_counter()
            if int(args.tokens) == int(plan.count):
                setup = DwarfStarFP8MLXPrefillExecutorSetup(lm, mx=mx).prepare(plan, tokens)
                setup_s = time.perf_counter() - setup_t0
                arena, runner = setup.arena, setup.block_runner
                prefill_t0 = time.perf_counter()
                with p8.verification_context():
                    runner.execute_batch(plan.commands, arena)
                prefill_s = time.perf_counter() - prefill_t0
                cache = runner.working_cache
                schedulers = [getattr(runner, "scheduling_coordinator", None)]
                close_runner = runner.close
                segment_plan = [(0, int(plan.count), "COMPLETE")]
            else:
                app = DeferredPrefillAppend.create(lm, lm.make_cache(), tokens, committed_frontier=0, mx=mx)
                setup_s = time.perf_counter() - setup_t0
                prefill_t0 = time.perf_counter()
                with p8.verification_context():
                    app.execute_all()
                prefill_s = time.perf_counter() - prefill_t0
                cache = app.live_cache
                segment_record_events = [e for r in getattr(app, "segment_records", []) for e in getattr(r, "scheduling_events", ())]
                schedulers = []
                if getattr(app, "final_execution", None) is not None:
                    schedulers.append(getattr(app.final_execution.runner, "scheduling_coordinator", None))
                close_runner = lambda: None
                segment_plan = [(s.start, s.count, s.mode.value) for s in app.plan.segments]
            frontiers = []
            for c in cache:
                try:
                    v = c[0]
                    frontiers.append(int(v.item()) if hasattr(v, "item") else int(v[0]) if hasattr(v, "__getitem__") else int(v))
                except Exception:
                    try:
                        frontiers.append(int(c.size()))
                    except Exception:
                        frontiers.append(None)
            events = list(locals().pop("segment_record_events", []))
            for sched in schedulers:
                tel = getattr(sched, "telemetry", None) if sched is not None else None
                events.extend(list(getattr(tel, "events", [])))
            bg_reads = sum(1 for e in events if e.get("event") == "engram_prefetch_submit" and e.get("donor_issue_observed"))
            fg_fallback = sum(1 for e in events if e.get("event") == "engram_consume" and not e.get("logical_match"))
            run = {
                "label": label,
                "setup_s": setup_s,
                "prefill_s": prefill_s,
                "final_seal_s": None,
                "p5_bootstrap_s": None,
                "decode_token1_s": None,
                "decode_token2_s": None,
                "segment_plan": segment_plan,
                "cache_object_id_sample": [id(c) for c in cache[:3]],
                "cache_frontiers": frontiers,
                "foreground_engram_fallback": fg_fallback,
                "background_microtile_reads": bg_reads,
                "p8": p8.to_json() if args.p8_verify else {"enabled": False},
            }
            report["runs"].append(run)
            print(json.dumps({"event": "progress", "mode": f"p8-{args.tokens}", "count": int(args.tokens), "stage": f"done-{label}", "elapsed_wall_s": prefill_s}), flush=True)
            close_runner()
        warms = [r["prefill_s"] for r in report["runs"] if r["label"].startswith("warm")]
        report["summary"] = {
            "cold_prefill_s": report["runs"][0]["prefill_s"],
            "warm": summarize(warms),
            "cold_warm_ratio": (report["runs"][0]["prefill_s"] / statistics.median(warms)) if warms else None,
        }
        report["status"] = "PASS"
    finally:
        try:
            delattr(model.language_model, "_p8_optimizer")
        except Exception:
            pass
        model.close()
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"))
    p.add_argument("--tokens", type=int, choices=(8192, 16384), required=True)
    p.add_argument("--warm-repeats", type=int, default=3)
    p.add_argument("--p8-verify", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    report = {"status": "FAIL"}
    try:
        report = run_family(args)
    except Exception as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps({"status": report.get("status"), "out": str(args.out), "summary": report.get("summary")}), flush=True)


if __name__ == "__main__":
    main()
