#!/usr/bin/env python3
"""Diagnostic-only canonical MTP quiescence; never a production selector.

Uses pinned oMLX directly, not M27's incomplete prompt/finish harness.  Every
prompt chunk and response is accounted for.  A position/token execution ledger
checks that the existing queue prefix is exactly the target-committed suffix.
"""
import argparse
import copy
import dataclasses
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PIN = "4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40"


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--contexts", default="4096,65536")
    ap.add_argument("--interruptions", default="1,2,3,4,5,6,7,8,9,10,12,16")
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--fixture", choices=("code", "synthetic"), default="code")
    ap.add_argument("--protocol-only", action="store_true")
    ap.add_argument("--omlx-path", type=Path, default=Path.home()/"omlx-0.7.0.release")
    ap.add_argument("--checkpoint", type=Path, default=Path("/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"))
    args = ap.parse_args()
    assert git(args.omlx_path, "rev-parse", "HEAD") == PIN
    assert not git(args.omlx_path, "status", "--short")
    sys.path.insert(0, str(args.omlx_path))
    import mlx.core as mx
    import omlx.scheduler
    from mlx_lm.generate import BatchGenerator, generation_stream
    from mlx_lm.sample_utils import make_sampler
    from omlx.patches.deepseek_v41.loading import load
    from omlx.patches.mlx_lm_mtp import batch_generator as mtp, cache_rollback
    from omlx.patches.mlx_lm_mtp import prompt_priming
    assert mtp.apply() and cache_rollback.apply()
    from tools.run_m26_upstream_mtp_bench import make_prompt
    from tools.run_m6_performance_qualification import deterministic_tokens
    from ds41f_mlx.runtime.omlx_decode import OMLXDecodeStateAdapter

    result = {"schema": "ds41f.m28.quiesce.v1", "scope": "DIAGNOSTIC_ONLY; canonical-session quiescence, NOT immediate token-exact abort",
              "base_commit": git(ROOT, "rev-parse", "HEAD"), "omlx_revision": PIN,
              "checkpoint": str(args.checkpoint), "config_sha256": hashlib.sha256((args.checkpoint/"config.json").read_bytes()).hexdigest(),
              "packages": {n: md.version(n) for n in ("omlx", "mlx", "mlx-lm")},
              "executable": sys.executable, "python": sys.version,
              "settings": {"depth": args.depth, "engram_ssd": True, "prefix_cache": False, "sampling": "mlx_lm make_sampler(temp=1, top_p=1, top_k=0)", "priming": "stock full native chunked prefill; terminal holdout once", "contexts": args.contexts, "fixture": args.fixture},
              "cases": [], "status": "RUNNING"}
    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2)+"\n")

    counters = {"target_forwards": 0, "verify_cycles": 0, "dspark_proposals": 0, "cache_constructions": 0, "reconcile_attempts": 0, "repack_calls": 0, "history_replay_tokens": 0, "dspark_appends": 0}
    quiesce_frontier = None
    quiesce_forward_spans = []
    original_pack = OMLXDecodeStateAdapter._pack_cache_array
    original_admit = OMLXDecodeStateAdapter.admit
    def forbid_repack(*a, **kw):
        counters["repack_calls"] += 1
        raise AssertionError("ds41f portable-state admission/repack forbidden")
    OMLXDecodeStateAdapter._pack_cache_array = forbid_repack
    OMLXDecodeStateAdapter.admit = forbid_repack
    ledger = {}
    cycles = []
    frozen = False
    last_emit = {}
    original_cycle = mtp._run_verify_cycle_chain
    original_emit = mtp._emit_response
    original_reconcile = mtp._reconcile_mtp_to_standard
    def no_replay(*a, **kw):
        counters["reconcile_attempts"] += 1
        raise AssertionError("full-history reconciliation forbidden")
    mtp._reconcile_mtp_to_standard = no_replay
    def track_cycle(gb, state, *a, **kw):
        if frozen:
            raise AssertionError("NEW VERIFY DURING QUIESCENCE")
        counters["verify_cycles"] += 1
        before_accepts = state.stats.accepts
        k = int(state.drafts.shape[0])
        ret = original_cycle(gb, state, *a, **kw)
        cycles.append({"depth": k, "accepted_final": state.stats.accepts-before_accepts,
                       "target_frontier": gb.prompt_cache[0].size(),
                       "history_frontier": len(gb.tokens[0]),
                       "queue": [[int(t), src] for t, lp, src in state.queue],
                       "next_main": state.next_main.tolist()})
        return ret
    mtp._run_verify_cycle_chain = track_cycle
    def track_emit(gb, token, lp, stats=None):
        # Natural finish filters the batch and drops state: retain the ring and
        # counters BEFORE calling the original epilogue, use its returned cache.
        last_emit.clear()
        last_emit.update(state=getattr(gb, "_omlx_mtp_state", None), cache=gb.prompt_cache)
        return original_emit(gb, token, lp, stats)
    mtp._emit_response = track_emit
    model = None
    try:
        model, processor = load(args.checkpoint, preserve_mtp=True, engram_ssd_offload=True)
        lm = model.language_model
        result["device"] = mx.device_info()
        result["window_size"] = lm._config.window_size
        result["checkpoint_index_sha256"] = hashlib.sha256((args.checkpoint/"model.safetensors.index.json").read_bytes()).hexdigest() if (args.checkpoint/"model.safetensors.index.json").exists() else None
        result["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(mtp.__file__), Path(__file__), args.omlx_path/"omlx/patches/deepseek_v41/mtp.py", args.omlx_path/"omlx/patches/mlx_lm_mtp/deepseek_v4_dspark.py"]}
        original_forward = lm._forward
        original_make = lm.make_cache
        original_ds = lm.dspark_forward
        original_append = lm.dspark_append_context
        def append_context(*a, **kw):
            counters["dspark_appends"] += 1
            return original_append(*a, **kw)
        lm.dspark_append_context = append_context
        def forward(ids, cache=None, **kw):
            counters["target_forwards"] += 1
            start = cache[0].size() if cache else 0
            tokens = ids.tolist()[0]
            if frozen:
                quiesce_forward_spans.append({"start": start, "tokens": tokens})
                counters["history_replay_tokens"] += sum(start+i < quiesce_frontier for i in range(len(tokens)))
                assert start >= quiesce_frontier, "history replay during quiescence"
            ledger.update({start+i: int(t) for i, t in enumerate(tokens)})
            return original_forward(ids, cache=cache, **kw)
        def make_cache(*a, **kw):
            if frozen: raise AssertionError("TARGET CACHE RECONSTRUCTION DURING QUIESCENCE")
            counters["cache_constructions"] += 1
            return original_make(*a, **kw)
        def ds_forward(*a, **kw):
            if frozen: raise AssertionError("NEW DRAFT DURING QUIESCENCE")
            counters["dspark_proposals"] += 1
            return original_ds(*a, **kw)
        lm._forward, lm.make_cache, lm.dspark_forward = forward, make_cache, ds_forward
        sampler = make_sampler(temp=1.0, top_p=1.0, top_k=0)
        def frontiers(cache): return [int(c.size()) for c in cache]
        def rings(state): return [int(c.offset) for c in state.mtp_cache]
        def queue(state): return [[int(t), src] for t, lp, src in state.queue]

        def run(context, interrupt, mode="cancel", stop_token=None):
            nonlocal frozen, quiesce_frontier
            frozen = False
            ledger.clear(); cycles.clear(); last_emit.clear()
            prompt_priming.drop_ctx(lm)
            mx.random.seed(12345+context)
            lm.configure_mtp(True, args.depth)
            ids = make_prompt(processor.tokenizer, context) if args.fixture=="code" else list(map(int, deterministic_tokens(context)))
            prefix_cache = lm.make_cache()
            with mx.stream(generation_stream):
                for begin in range(0, len(ids)-1, 2048):
                    logits = lm(mx.array([ids[begin:min(begin+2048, len(ids)-1)]]), cache=prefix_cache)
                    mx.eval(logits)
                bg = BatchGenerator(lm, max_tokens=interrupt if mode=="length" else 128,
                                    stop_tokens=[stop_token if isinstance(stop_token, list) else [stop_token]] if stop_token is not None else None,
                                    sampler=sampler, completion_batch_size=1, prefill_batch_size=1,
                                    prefill_step_size=2048, stream=generation_stream)
                uid = bg.insert([[ids[-1]]], caches=[prefix_cache], all_tokens=[ids[:-1]])[0]
                pr, gr = bg.next(); mx.synchronize(generation_stream)
                assert not gr and any(r.end_of_prompt for r in pr)
                delivered = []
                response = None
                topology = []
                for i in range(interrupt):
                    _, gr = bg.next(); mx.synchronize(generation_stream)
                    assert len(gr)==1
                    response = gr[0]
                    delivered.append(int(response.token))
                    st = getattr(bg._generation_batch, "_omlx_mtp_state", None) or last_emit.get("state")
                    assert st is not None, "enabled but inactive MTP"
                    ca = response.prompt_cache if response.finish_reason else bg._generation_batch.prompt_cache
                    topology.append({"emitted": len(delivered), "frontier": frontiers(ca), "queue": queue(st), "dspark": rings(st), "finish": response.finish_reason})
                    if response.finish_reason: break
                gb = bg._generation_batch
                state = getattr(gb, "_omlx_mtp_state", None) or last_emit["state"]
                cache = response.prompt_cache if response.finish_reason else gb.prompt_cache
                history = list(response.all_tokens) if response.finish_reason else list(gb.tokens[0])
                assert history == ids + delivered, "canonical emitted history mismatch"
                rec = {"context": context, "interrupt_after": interrupt, "mode": mode, "stop_token": stop_token,
                       "delivered_tokens": delivered, "owning_cycle": copy.deepcopy(cycles[-1]) if cycles else None,
                       "cycles": copy.deepcopy(cycles), "topology": topology,
                       "before": {"history_frontier": len(history), "target": frontiers(cache), "dspark": rings(state), "queue": queue(state), "stats": dataclasses.asdict(state.stats)}}
                result["cases"].append(rec); save()
                frozen = True
                quiesce_frontier = cache[0].size()
                quiesce_forward_spans.clear()
                before_counts = dict(counters)
                t0 = time.perf_counter()
                try:
                    fs = frontiers(cache)
                    assert len(fs)==40 and len(set(fs))==1
                    target = fs[0]
                    needed = target-len(history)
                    rec["needed_commits"] = needed
                    assert all(ledger[p] == token for p, token in enumerate(history[:target])), "target execution token ledger mismatch"
                    ring_keys_before = [c.keys for c in state.mtp_cache]
                    extra = []
                    discarded = []
                    forward_tokens = []
                    if needed >= 0:
                        assert needed <= args.depth, "drain exceeds depth"
                        assert rings(state)==[target]*len(state.mtp_cache), "DSpark mismatch"
                        assert len(state.queue)==needed+1, "unsupported queue/target relationship"
                        expected = [ledger[p] for p in range(len(history), target)]
                        actual = [int(t) for t, lp, src in list(state.queue)[:needed]]
                        assert actual==expected, "queue prefix differs from actual target inputs"
                        rec["committed_queue_tokens_match_execution_ledger"] = True
                        # Known matcher/length boundaries may NOT be crossed by canonical drain.
                        if not response.finish_reason:
                            matcher = copy.copy(gb._matchers[0])
                            assert gb._num_tokens[0]+needed < gb.max_tokens[0]
                            assert not any(matcher.advance(t) for t in actual)
                        for _ in range(needed):
                            assert state.queue, "empty queue: never call _mtp_next"
                            token, lp, source = state.queue.popleft()
                            emitted = mtp._emit_response(gb, token, lp, state.stats)
                            assert len(emitted)==1 and emitted[0].finish_reason is None
                            history.append(int(token)); extra.append(int(token))
                        discarded = queue(state)
                        assert len(discarded)==1
                        assert all(c.keys is keys for c, keys in zip(state.mtp_cache, ring_keys_before)), "drain changed DSpark ring"
                        rec["dspark_ring_unchanged_during_drain"] = True
                    else:
                        assert needed == -1 and not state.queue, "unsupported lagging topology"
                        # Only last ALREADY emitted token is missing. Never sample a successor.
                        forward_tokens = history[target:]
                        logits, hidden = lm(mx.array([forward_tokens]), cache=cache, return_dspark_hidden=True)
                        lm.dspark_append_context(hidden, state.mtp_cache, start_offset=target)
                        mx.eval(logits, *[c.keys for c in state.mtp_cache]); mx.synchronize(generation_stream)
                    assert frontiers(cache)==[len(history)]*40
                    assert rings(state)==[len(history)]*len(state.mtp_cache)
                    state.queue.clear()
                    mtp._clear_rollback(cache)
                    # Use native row extraction, never convert/reconstruct cache tensors.
                    idle_cache = cache if response.finish_reason else gb.extract_cache(0)
                    idle_rings = state.mtp_cache
                    if hasattr(gb, "_omlx_mtp_state"): delattr(gb, "_omlx_mtp_state")
                    bg.remove([uid]); bg.close()
                    prompt_priming.drop_ctx(lm)
                    state.drafts=None; state.next_main=None; state.mtp_cache=None
                    state.draft_lps=[]; state.draft_accept_lps=[]
                    rec["quiescence_seconds"] = time.perf_counter()-t0
                    rec["counter_delta"] = {k: counters[k]-before_counts[k] for k in counters}
                    assert rec["counter_delta"]["verify_cycles"]==0
                    assert rec["counter_delta"]["dspark_proposals"]==0
                    assert rec["counter_delta"]["cache_constructions"]==0
                    assert rec["counter_delta"]["reconcile_attempts"]==0
                    assert rec["counter_delta"]["history_replay_tokens"]==0
                    assert rec["counter_delta"]["repack_calls"]==0
                    assert rec["counter_delta"]["dspark_appends"] == (1 if needed < 0 else 0)
                    assert rec["counter_delta"]["target_forwards"] == (1 if needed < 0 else 0)
                    rec["quiesce_target_forward_spans"] = list(quiesce_forward_spans)
                    rec["after"] = {"canonical_history": list(history), "canonical_history_sha256": hashlib.sha256(json.dumps(history).encode()).hexdigest(),
                                    "target": frontiers(idle_cache), "dspark": [c.offset for c in idle_rings],
                                    "extra_server_committed_tokens_NOT_delivered": extra, "discarded_uncommitted_queue": discarded,
                                    "known_missing_tokens_forwarded": forward_tokens,
                                    "rollback_stash_present": any(getattr(c, "_mtp_draft_stash", None) is not None for c in idle_cache),
                                    "scheduler_uids": list(bg._generation_batch.uids), "queue_len": len(state.queue)}
                    assert not rec["after"]["rollback_stash_present"]
                    # Strong proof: continue from idle with only NEW suffix tokens.
                    frozen = False
                    lm.configure_mtp(False, 1)
                    continuation = []
                    for token in [ids[-1], ids[-2]]:
                        start = len(history)
                        logits, hidden = lm(mx.array([[token]]), cache=idle_cache, return_dspark_hidden=True)
                        lm.dspark_append_context(hidden, idle_rings, start_offset=start)
                        mx.eval(logits, *[c.keys for c in idle_rings])
                        history.append(token)
                        continuation.append({"new_input_token": token, "target": frontiers(idle_cache), "dspark": [c.offset for c in idle_rings]})
                    rec["ordinary_target_continuation"] = continuation
                    assert frontiers(idle_cache)==[len(history)]*40
                    rec["status"] = "PASS"
                except Exception as exc:
                    rec["status"] = "FAIL_CLOSED"
                    rec["error"] = repr(exc)
                    frozen = False
                    bg.close()
                finally:
                    frozen = False
                    save()
                print(json.dumps({"context": context, "interrupt": interrupt, "mode": mode, "status": rec["status"], "needed": rec.get("needed_commits"), "extra": len(rec.get("after", {}).get("extra_server_committed_tokens_NOT_delivered", []))}), flush=True)
                return delivered

        contexts = [int(x) for x in args.contexts.split(",")]
        for context in ([] if args.protocol_only else contexts):
            positions = [int(x) for x in args.interruptions.split(",")]
            if context>4096: positions = [1,2,3,5,7,8]
            for point in positions: run(context, point)
        if not args.protocol_only:
            for point in [1,2,3,7,16]: run(contexts[0], point, "length")
        # Known single-token stop: derive token from deterministic seeded native path,
        # then use upstream stop matcher (not a new protocol/parser implementation).
        tokens = run(contexts[0], 8, "stop_seed")
        for token in list(dict.fromkeys(tokens))[:3]: run(contexts[0], 32, "stop", token)
        run(contexts[0], 32, "stop_sequence", tokens[2:5])
        result["status"] = "COMPLETED_DIAGNOSTIC"
    except Exception as exc:
        result["status"] = "ERROR"; result["error"] = repr(exc)
        raise
    finally:
        OMLXDecodeStateAdapter._pack_cache_array = original_pack
        OMLXDecodeStateAdapter.admit = original_admit
        mtp._run_verify_cycle_chain = original_cycle
        mtp._emit_response = original_emit
        mtp._reconcile_mtp_to_standard = original_reconcile
        if model is not None: model.close()
        result["totals"] = counters
        save()


if __name__ == "__main__": main()
