#!/usr/bin/env python3
"""Diagnostic-only pinned upstream MTP boundary probe; never a serving selector.

Uses existing DENSE_P0_P7 prefill and upstream BatchGenerator directly. Frozen
prefix row views are diagnostic forks (no tensor reconstruction), NOT a normal
continuation mechanism. Any upstream full-history reconciliation is blocked.
A failed idle boundary stops lifecycle qualification, not the diagnostic decode.
"""
from pathlib import Path
import argparse
import dataclasses
import hashlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PIN = "4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40"


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def boundary(cache, history):
    offsets = [int(c.size()) for c in cache]
    return {"history_tokens": len(history), "all40_offsets": offsets,
            "coherent": len(offsets) == 40 and all(x == len(history) for x in offsets)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mtp", choices=("OFF", "ON"), required=True)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--contexts", default="2048,200000")
    ap.add_argument("--omlx-path", type=Path, default=Path.home()/"omlx-0.7.0.release")
    ap.add_argument("--checkpoint", type=Path, default=Path("/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert git(args.omlx_path, "rev-parse", "HEAD") == PIN
    assert not git(args.omlx_path, "diff", "HEAD", "--name-only")
    assert 1 <= args.depth <= 5
    sys.path.insert(0, str(args.omlx_path))
    import mlx.core as mx
    import psutil
    import omlx.scheduler  # exact upstream GenerationBatch patches
    from omlx.patches.mlx_lm_mtp import batch_generator as mtp, cache_rollback
    # Importing scheduler alone does NOT install MTP dispatch. V4.1 supplies
    # native model hooks; install the two upstream loop/cache patches explicitly.
    assert cache_rollback.apply() and mtp.apply()
    from omlx.patches.deepseek_v41.loading import load
    from mlx_lm.generate import BatchGenerator, generation_stream
    from ds41f_mlx.runtime.dwarfstar_prefill import DenseP0P7PrefillSession
    from tools.run_m6_performance_qualification import deterministic_tokens

    result = {"schema": "ds41f.m25.mtp-boundary-probe.v1", "qualification": "DIAGNOSTIC_ONLY",
              "base_commit": git(ROOT, "rev-parse", "HEAD"), "mtp": args.mtp, "depth": args.depth,
              "omlx_revision": PIN, "omlx_path": str(args.omlx_path), "checkpoint": str(args.checkpoint),
              "checkpoint_config_sha256": hashlib.sha256((args.checkpoint/"config.json").read_bytes()).hexdigest(),
              "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "packages": {n: md.version(n) for n in ("omlx", "mlx", "mlx-lm", "deepseek-recipe")},
              "environment": {k:v for k,v in os.environ.items() if k.startswith(("OMLX_", "DS41F_"))},
              "configuration": {"prefill": "DENSE_P0_P7", "engram_ssd_offload": True,
                                "moe_expert_offload_resident_fraction": None, "sampler": "argmax",
                                "prompt_mtp_priming": "OFF during unchanged P7; terminal capture only",
                                "full_history_reconcile": "FORBIDDEN", "fixture": "M6 deterministic valid token IDs",
                                "serving_enabled": False}, "cases": [], "events": [], "status": "RUNNING"}
    files = [args.omlx_path/"omlx/patches/deepseek_v41"/n for n in ("mtp.py", "cache.py", "loading.py", "language.py", "dspark.py")]
    files += [Path(mtp.__file__), Path(sys.modules["mlx_lm.generate"].__file__), Path(__file__)]
    result["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2)+"\n")
    def memory():
        return {"rss_bytes": psutil.Process().memory_info().rss, "mlx_active_bytes": mx.get_active_memory(),
                "mlx_peak_bytes": mx.get_peak_memory(), "mlx_cache_bytes": mx.get_cache_memory()}
    original_reconcile = mtp._reconcile_mtp_to_standard
    def forbid_reconcile(*a, **kw):
        result["events"].append({"event": "forbidden_full_history_reconcile"})
        return False
    mtp._reconcile_mtp_to_standard = forbid_reconcile
    model = None
    try:
        t = time.perf_counter()
        model, _ = load(args.checkpoint, preserve_mtp=args.mtp == "ON", engram_ssd_offload=True)
        mx.synchronize()
        lm = model.language_model
        result["load"] = {"seconds": time.perf_counter()-t, **memory()}
        result["model_config"] = {k: getattr(lm._config, k) for k in
                                  ("preserve_mtp", "dspark_block_size", "dspark_target_layer_ids", "n_mtp_layers")}
        original_rollback = lm.mtp_partial_rollback
        def rollback(cache, accepted, drafts):
            before = [c.size() for c in cache]
            ok = original_rollback(cache, accepted, drafts)
            result["events"].append({"event": "target_partial_rollback", "accepted": accepted,
                                     "drafts": drafts, "returned": ok, "before": before,
                                     "after": [c.size() for c in cache]})
            return ok
        lm.mtp_partial_rollback = rollback
        for count in map(int, args.contexts.split(",")):
            ids = deterministic_tokens(count+1)
            pre = DenseP0P7PrefillSession(model, omlx_path=args.omlx_path, mx=mx)
            t = time.perf_counter()
            pref = pre.prefill(ids[:-1])
            prefix = pref.live_result.live_cache
            pre_seconds = time.perf_counter()-t
            # Direct upstream probe forks frozen prefix views, never publishes them
            # as a ds41f session or persistence artifact.
            for name, limit, cancel_at in (("length1", 1, None), ("length2", 2, None),
                                          ("ordinary64", 64, None), ("cancel1", 64, 1), ("cancel7", 64, 7)):
                lm.configure_mtp(args.mtp == "ON", args.depth)
                bg = BatchGenerator(lm, max_tokens=limit, sampler=lambda x: mx.argmax(x, axis=-1),
                                    completion_batch_size=1, prefill_batch_size=1,
                                    prefill_step_size=2048, stream=generation_stream)
                rec = {"context": count, "scenario": name, "prefill_seconds": pre_seconds,
                       "prefill_boundary": boundary(prefix, ids[:-1]), "steps": [], "memory_before": memory()}
                result["cases"].append(rec)
                try:
                    with mx.stream(generation_stream):
                        cache = [c.extract(0) for c in prefix]
                        uid = bg.insert([[ids[-1]]], max_tokens=[limit], caches=[cache], all_tokens=[ids[:-1]])[0]
                        t = time.perf_counter()
                        pr, gr = bg.next(); mx.synchronize(generation_stream)
                        rec["bootstrap_seconds"] = time.perf_counter()-t
                        assert not gr
                        rec["bootstrap_boundary"] = boundary(bg._generation_batch.prompt_cache, ids)
                        rec["prompt_responses"] = [{"end_of_prompt": r.end_of_prompt, "progress": r.progress} for r in pr]
                        assert all(r.end_of_prompt for r in pr)
                        total = 0
                        tdecode = time.perf_counter()
                        while total < limit:
                            t = time.perf_counter()
                            _, gr = bg.next(); mx.synchronize(generation_stream)
                            latency = time.perf_counter()-t
                            assert len(gr) == 1, "singleton contract changed"
                            r = gr[0]; total += 1
                            state = getattr(bg._generation_batch, "_omlx_mtp_state", None)
                            step = {"token": int(r.token), "seconds": latency, "finish_reason": r.finish_reason}
                            if state is not None:
                                rec["last_stats"] = dataclasses.asdict(state.stats)
                                step.update(queue_tokens=[int(q[0]) for q in state.queue],
                                            depth=state.depth, adaptive_depth=getattr(state.controller, "cur", None),
                                            target_offsets=[c.size() for c in bg._generation_batch.prompt_cache])
                            rec["steps"].append(step)
                            if r.finish_reason is not None:
                                rec["idle_boundary"] = boundary(r.prompt_cache, r.all_tokens)
                                break
                            if cancel_at == total:
                                extracted, history = bg.extract_cache([uid])[uid]
                                rec["idle_boundary"] = boundary(extracted, history)
                                break
                        if args.mtp == "ON" and limit >= 64 and cancel_at is None:
                            assert rec.get("last_stats", {}).get("cycles", 0) > 0, "enabled flag without actual MTP verification"
                        rec["decode_seconds"] = time.perf_counter()-tdecode
                        rec["decode_tok_s"] = total/rec["decode_seconds"]
                        rec["first_token_seconds"] = rec["steps"][0]["seconds"]
                        rec["memory_after"] = memory()
                        stats = rec.get("last_stats", {})
                        drafted = sum(stats.get("depth_drafted", []))
                        rec["acceptance_rate"] = stats.get("accepts", 0)/drafted if drafted else None
                        rec["accepted_per_cycle"] = stats.get("accepts", 0)/stats["cycles"] if stats.get("cycles") else None
                finally:
                    bg.remove(list(bg._generation_batch.uids))
                    bg.close()
                    rec["cleanup_empty"] = not (bg._generation_batch.uids or bg._prompt_batch.uids or bg._unprocessed_sequences)
                    rec["host_prime_context_retained"] = getattr(lm, "_omlx_mtp_prime_ctx", None) is not None
                    save()
                print(json.dumps({"context": count, "scenario": name, "mtp": args.mtp,
                                  "tok_s": rec.get("decode_tok_s"), "idle": rec.get("idle_boundary")}), flush=True)
            del prefix, pref, pre
            mx.clear_cache()
        result["status"] = "COMPLETED_DIAGNOSTIC"
    except Exception as exc:
        result["status"] = "ERROR"
        result["error"] = repr(exc)
        raise
    finally:
        mtp._reconcile_mtp_to_standard = original_reconcile
        if model is not None:
            model.close()
        save()


if __name__ == "__main__":
    main()
