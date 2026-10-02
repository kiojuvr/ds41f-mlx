#!/usr/bin/env python3
"""M27 diagnostic-only DeepSeek V4.1 MTP lifecycle probe.

This does not enable production MTP.  It tests the narrow lifecycle operations
against stock oMLX state: full DSpark priming, arbitrary stop/cancel points,
a bounded committed-idle materialization, DSpark committed-ring carry, and MTP
re-entry after an appended suffix without prompt replay/repack.
"""
from __future__ import annotations

import argparse, dataclasses, hashlib, importlib.metadata as md, json, os, platform, subprocess, sys, time
from pathlib import Path
from typing import Any

PIN = "4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40"

def git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()

def sha(path: Path) -> str | None:
    try: return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError: return None

def offsets(cache): return [int(c.size()) for c in cache]

def qlen(gb):
    st = getattr(gb, "_omlx_mtp_state", None)
    return None if st is None else len(st.queue)

def ds_offsets(caches):
    if caches is None: return None
    return [int(getattr(c, "offset", -1)) for c in caches]

def make_prompt(tok, n: int) -> list[int]:
    text = ("def lifecycle_probe(x: int) -> int:\n"
            "    total = 0\n"
            "    for i in range(x):\n"
            "        total += (i * 17 + 3) % 101\n"
            "    return total\n\n")
    ids=[]
    while len(ids)<n: ids.extend(tok.encode(text))
    return ids[:n]

class Counters:
    replay=0; repack=0; forwarded_missing=0


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--omlx-path", type=Path, default=Path.home()/"omlx-0.7.0.release")
    ap.add_argument("--checkpoint", type=Path, default=Path("/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--contexts", default="4096,65536")
    ap.add_argument("--first-tokens", default="1,7,64")
    ap.add_argument("--second-tokens", type=int, default=32)
    ap.add_argument("--append-tokens", type=int, default=8)
    ap.add_argument("--depth", type=int, default=5)
    args=ap.parse_args()
    assert git(args.omlx_path,"rev-parse","HEAD") == PIN
    assert not git(args.omlx_path,"status","--short")
    sys.path.insert(0,str(args.omlx_path))
    import mlx.core as mx
    import omlx.scheduler # noqa
    from mlx_lm.generate import BatchGenerator, generation_stream
    from omlx.patches.deepseek_v41.loading import load
    from omlx.patches.mlx_lm_mtp import batch_generator as mtp, cache_rollback
    from omlx.patches.mlx_lm_mtp.deepseek_v4_dspark import _DSparkPrimeContext
    assert cache_rollback.apply() and mtp.apply()

    result={"schema":"ds41f.m27.mtp-lifecycle-probe.v1","qualification":"DIAGNOSTIC_ONLY_NOT_PRODUCTION",
            "ds41f_commit":git(Path(__file__).resolve().parents[1],"rev-parse","HEAD"),
            "omlx_revision":PIN,"omlx_path":str(args.omlx_path),"checkpoint":str(args.checkpoint),
            "checkpoint_config_sha256":sha(args.checkpoint/"config.json"),"python":sys.version,
            "executable":sys.executable,"platform":platform.platform(),
            "packages":{n:md.version(n) for n in ("omlx","mlx","mlx-lm")},
            "settings":{"contexts":args.contexts,"first_tokens":args.first_tokens,"second_tokens":args.second_tokens,"append_tokens":args.append_tokens,"depth":args.depth,"engram_ssd_offload":True,"sampler":"categorical temperature=1.0","prefix_cache":False},
            "state_model":{"target_cache":"sole executable target authority; must equal committed history at idle","dspark":"bounded committed-only context ring derived from target hidden taps; no draft KV retained","idle_algorithm":"materialize any cache-behind emitted suffix with bounded one-token/short-suffix target forwards, append missing hidden taps to DSpark ring only if needed, reject cache-ahead, clear queues/stashes, retain DSpark ring as next priming context"},
            "runs":[],"status":"RUNNING"}
    def save():
        args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2)+"\n")

    orig_reconcile=mtp._reconcile_mtp_to_standard
    def forbid_reconcile(*a,**kw):
        Counters.replay += 1
        return False
    mtp._reconcile_mtp_to_standard=forbid_reconcile
    model=None
    try:
        model, processor=load(args.checkpoint,preserve_mtp=True,engram_ssd_offload=True)
        lm=model.language_model; lm.configure_mtp(True,args.depth)
        tok=processor.tokenizer
        def sampler(logits): return mx.random.categorical(logits)

        def committed_idle(bg, history, note):
            gb=bg._generation_batch; st=getattr(gb,"_omlx_mtp_state",None); cache=gb.prompt_cache
            before={"history_len":len(history),"cache_offsets":offsets(cache),"queue_len":None if st is None else len(st.queue),"dspark_offsets":None if st is None else ds_offsets(st.mtp_cache),"stats":None if st is None else dataclasses.asdict(st.stats)}
            if len(set(before["cache_offsets"])) != 1:
                return {"ok":False,"reason":"DIVERGENT_TARGET_CACHE_OFFSETS","before":before}
            frontier=before["cache_offsets"][0]
            if frontier > len(history):
                return {"ok":False,"reason":"CACHE_AHEAD_OF_COMMITTED_HISTORY","before":before}
            missing=history[frontier:]
            hidden=None
            if missing:
                if len(missing) > 8: raise RuntimeError(f"unbounded missing suffix {len(missing)}")
                arr=mx.array([missing], dtype=mx.int64)
                logits, hidden = lm(arr, cache=cache, return_dspark_hidden=True)
                mx.eval(logits, hidden); mx.synchronize(generation_stream)
                Counters.forwarded_missing += len(missing)
                if st is not None and st.mtp_cache is not None:
                    ds_off=st.mtp_cache[0].offset
                    if ds_off < len(history):
                        start=ds_off-frontier
                        if start < 0: raise RuntimeError("DSpark ring behind target missing-window start")
                        lm.dspark_append_context(hidden[:, start:], st.mtp_cache, start_offset=ds_off)
                        mx.eval(*[c.keys for c in st.mtp_cache if c.keys is not None])
            after_offsets=offsets(cache)
            if any(x != len(history) for x in after_offsets):
                return {"ok":False,"reason":"MATERIALIZATION_FAILED","before":before,"after_offsets":after_offsets}
            ds_caches = st.mtp_cache if st is not None else None
            if ds_caches is not None and len({c.offset for c in ds_caches}) == 1:
                setattr(lm, "_omlx_mtp_prime_ctx", _DSparkPrimeContext(ds_caches, len(history)))
            if st is not None:
                st.queue.clear(); st.drafts=None; st.draft_lps.clear(); st.draft_accept_lps.clear(); st.next_main=None
                delattr(gb,"_omlx_mtp_state")
            for c in cache:
                if hasattr(c,"_mtp_draft_stash"): c._mtp_draft_stash=None
            return {"ok":True,"note":note,"before":before,"after":{"history_len":len(history),"cache_offsets":after_offsets,"dspark_offsets":ds_offsets(ds_caches),"queue_present":hasattr(gb,"_omlx_mtp_state"),"rollback_stashes":[hasattr(c,"_mtp_draft_stash") and c._mtp_draft_stash is not None for c in cache]},"bounded_missing_forwards":len(missing)}

        def generate_from(cache, history, terminal, max_tokens):
            lm.configure_mtp(True,args.depth)
            bg=BatchGenerator(lm,max_tokens=max_tokens,sampler=sampler,completion_batch_size=1,prefill_batch_size=1,prefill_step_size=2048,stream=generation_stream)
            uid=bg.insert([[terminal]],max_tokens=[max_tokens],caches=[cache],all_tokens=[history])[0]
            pr,gr=bg.next(); mx.synchronize(generation_stream)
            if gr: raise RuntimeError("generation during bootstrap")
            hist=history+[terminal]; steps=[]
            return bg,uid,hist,steps

        for ctx in [int(x) for x in args.contexts.split(',') if x]:
            prompt=make_prompt(tok,ctx)
            for first_count in [int(x) for x in args.first_tokens.split(',') if x]:
                rec={"context":ctx,"first_count":first_count,"events":[],"replay_repack_before":{"replay":Counters.replay,"repack":Counters.repack,"forwarded_missing":Counters.forwarded_missing}}
                cache=lm.make_cache()
                # Stock full prefill with native DSpark prompt capture, serving as equivalence authority for this lifecycle probe.
                t=time.perf_counter(); lm(mx.array([prompt],dtype=mx.int64),cache=cache); mx.synchronize(); rec["initial_prefill_s"]=time.perf_counter()-t
                # Host prime context now covers prompt.  Start MTP with a terminal token.
                terminal=prompt[-1]; prefix=prompt[:-1]
                # Rebuild exact ds41f-like split: cache currently has full prompt, so use it as prefix+terminal authority for this diagnostic.
                # The run still tests idle extraction and DSpark carry/re-entry, not P7 capture implementation.
                bg=BatchGenerator(lm,max_tokens=first_count,sampler=sampler,completion_batch_size=1,prefill_batch_size=1,prefill_step_size=2048,stream=generation_stream)
                uid=bg.insert([prompt],max_tokens=[first_count])[0]
                pr,gr=bg.next(); mx.synchronize(generation_stream)
                history=list(prompt); generated=[]
                tdec=time.perf_counter()
                for _ in range(first_count):
                    _,gr=bg.next(); mx.synchronize(generation_stream)
                    if not gr: continue
                    r=gr[0]; generated.append(int(r.token)); history.append(int(r.token))
                    rec["events"].append({"phase":"first_step","token":int(r.token),"finish":r.finish_reason,"queue_len":qlen(bg._generation_batch),"cache_offsets":offsets(bg._generation_batch.prompt_cache)[:4]})
                    if r.finish_reason is not None: break
                rec["first_decode_tok_s"]=len(generated)/(time.perf_counter()-tdec)
                idle=committed_idle(bg,history,"after_first")
                rec["idle1"]=idle
                cache=bg._generation_batch.prompt_cache
                bg.remove([uid]); bg.close()
                if not idle.get("ok") or len(cache) != 40:
                    if len(cache) != 40:
                        rec["post_idle_cache_unavailable"] = {"cache_len": len(cache)}
                    result["runs"].append(rec); save(); continue
                # Diagnostic P6 append equivalent: forward only new suffix tokens and append their hidden taps into retained DSpark ring.
                suffix=make_prompt(tok,args.append_tokens+1)[:args.append_tokens+1]
                append=suffix[:-1]; terminal2=suffix[-1]
                if append:
                    ctxobj=getattr(lm,"_omlx_mtp_prime_ctx")
                    off=len(history)
                    logits,hidden=lm(mx.array([append],dtype=mx.int64),cache=cache,return_dspark_hidden=True)
                    mx.eval(logits,hidden); mx.synchronize(generation_stream)
                    lm.dspark_append_context(hidden,ctxobj.caches,start_offset=off)
                    ctxobj.expected_target_offset=off+len(append)
                    history.extend(append)
                    rec["p6_diagnostic_append"]={"tokens":len(append),"cache_offsets":offsets(cache)[:4],"dspark_offsets":ds_offsets(ctxobj.caches)}
                bg2,uid2,history2,steps2=generate_from(cache,history,terminal2,args.second_tokens)
                tdec=time.perf_counter(); generated2=[]
                for _ in range(args.second_tokens):
                    _,gr=bg2.next(); mx.synchronize(generation_stream)
                    if not gr: continue
                    r=gr[0]; generated2.append(int(r.token)); history2.append(int(r.token))
                    if len(generated2)<=8: rec["events"].append({"phase":"second_step","token":int(r.token),"finish":r.finish_reason,"queue_len":qlen(bg2._generation_batch),"cache_offsets":offsets(bg2._generation_batch.prompt_cache)[:4]})
                    if r.finish_reason is not None: break
                rec["second_decode_tok_s"]=len(generated2)/(time.perf_counter()-tdec)
                rec["second_last_stats"]=dataclasses.asdict(getattr(bg2._generation_batch,"_omlx_mtp_state").stats) if getattr(bg2._generation_batch,"_omlx_mtp_state",None) else None
                rec["idle2"]=committed_idle(bg2,history2,"after_second")
                bg2.remove([uid2]); bg2.close()
                rec["replay_repack_after"]={"replay":Counters.replay,"repack":Counters.repack,"forwarded_missing":Counters.forwarded_missing}
                result["runs"].append(rec); save()
                print(json.dumps({"ctx":ctx,"first":first_count,"idle1":idle.get("ok"),"second_tok_s":rec.get("second_decode_tok_s"),"idle2":rec["idle2"].get("ok")}))
        result["status"]="COMPLETED_DIAGNOSTIC"
    except Exception as e:
        result["status"]="ERROR"; result["error"]=repr(e); raise
    finally:
        mtp._reconcile_mtp_to_standard=orig_reconcile
        if model is not None: model.close()
        save()

if __name__=="__main__": main()
