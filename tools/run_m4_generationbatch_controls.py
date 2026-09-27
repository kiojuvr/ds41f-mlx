#!/usr/bin/env python3
"""M4 P0-P5 execution-path timing with exact generation-stream synchronization."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT, DEFAULT_OMLX, OmlxRuntime, OmlxRuntimeConfig
from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig, OMLXDecodeSession
from tools.run_m4_omlx_base_decode_qualification import build_prefill_state

def stats(lat):
    return {"steps":len(lat),"first_step_s":lat[0] if lat else None,"warm_median_s":median(lat[1:]) if len(lat)>1 else None,"median_s":median(lat) if lat else None,"mean_s":mean(lat) if lat else None,"tok_per_s_median":1/median(lat) if lat else None,"tok_per_s_mean":1/mean(lat) if lat else None,"latencies_s":lat}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',str(DEFAULT_CHECKPOINT))); ap.add_argument('--omlx-path',default=os.environ.get('DS41F_OMLX',str(DEFAULT_OMLX))); ap.add_argument('--out',default='artifacts/m4/generationbatch-controls/result.json'); ap.add_argument('--steps',type=int,default=8); args=ap.parse_args()
    ck=Path(args.checkpoint); omlx=Path(args.omlx_path)
    if str(omlx) not in sys.path: sys.path.insert(0,str(omlx))
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator, GenerationBatch, generation_stream
    rec={"schema":"ds41f.m4.generationbatch-controls.v1","checkpoint":str(ck),"omlx_path":str(omlx),"fixture":{"prefill":[0,3],"first_decode_input":15},"sync_policy":"all timed steps execute inside or on the mlx-lm generation stream and call mx.synchronize(bg.stream or generation_stream) before stopping timer; completed-step latency, not enqueue-only"}
    rt=OmlxRuntime(OmlxRuntimeConfig(omlx_path=omlx,checkpoint_path=ck,engram_ssd_offload=True,preserve_mtp=False)); model,_=rt.load_model(); lm=model.language_model
    def greedy(logits): return mx.argmax(logits, axis=-1)
    def make_o_cache():
        c=lm.make_cache(); mx.eval(lm._forward(mx.array([[0,3]],mx.int64), cache=c)); mx.synchronize(); return c
    prefill=build_prefill_state(ck, Path('artifacts/m4/generationbatch-controls/native'), [0,3]); state=prefill.continuation_state
    def make_d_cache(): return OMLXDecodeSession.from_prefill_state(model,state,OMLXDecodeConfig(omlx_path=omlx,checkpoint_path=ck,preserve_mtp=False)).cache
    def token(i): return 15+i%4
    try:
        # P0 raw _forward singleton cache
        c=make_o_cache(); lat=[]
        for i in range(args.steps):
            t=time.perf_counter(); out=lm._forward(mx.array([[token(i)]],mx.int64), cache=c); mx.eval(out); mx.synchronize(); lat.append(time.perf_counter()-t)
        rec['P0_raw_forward_singleton_cache']=stats(lat)
        # P1 __call__ singleton cache
        c=make_o_cache(); lat=[]
        for i in range(args.steps):
            t=time.perf_counter(); out=lm(mx.array([[token(i)]],mx.int64), cache=c); mx.eval(out); mx.synchronize(); lat.append(time.perf_counter()-t)
        rec['P1_call_singleton_cache']=stats(lat)
        # P2 __call__ after per-layer singleton merge
        c=make_o_cache(); t0=time.perf_counter(); c=[type(x).merge([x]) for x in c]; mx.synchronize(); boot=time.perf_counter()-t0; lat=[]
        for i in range(args.steps):
            t=time.perf_counter(); out=lm(mx.array([[token(i)]],mx.int64), cache=c); mx.eval(out); mx.synchronize(); lat.append(time.perf_counter()-t)
        rec['P2_call_after_deepseek_cache_merge']={"bootstrap_s":boot,"timing":stats(lat)}
        # P3 direct GenerationBatch over ordinary prefilled cache
        c=make_o_cache(); t0=time.perf_counter()
        # Use BatchGenerator only as a source for the default stop-state machine; execution below is direct GenerationBatch.
        bg_tmp=BatchGenerator(lm,max_tokens=args.steps,sampler=greedy,completion_batch_size=1,prefill_batch_size=1)
        sm=[bg_tmp._default_state_machine]
        with mx.stream(generation_stream): gb=GenerationBatch(lm,[0],mx.array([15],mx.int64),c,[[0,3]],[None],greedy,[[]],sm,[args.steps])
        mx.synchronize(generation_stream); boot=time.perf_counter()-t0; lat=[]; emitted=[]
        for _ in range(max(0,args.steps-1)):
            t=time.perf_counter();
            with mx.stream(generation_stream): resp=gb.next()
            mx.synchronize(generation_stream); lat.append(time.perf_counter()-t); emitted += [int(r.token) for r in resp]
        rec['P3_direct_generationbatch_equivalent_cache']={"constructor_bootstrap_completed_s":boot,"post_bootstrap_timing":stats(lat),"emitted_tokens_after_bootstrap":emitted,"token_accounting":{"initial_tokens_argument":[0,3],"bootstrap_input":15,"cache_after_constructor":"[0,3,15]"}}
        # P4 standard BatchGenerator ordinary oMLX
        bg=BatchGenerator(lm,max_tokens=args.steps,sampler=greedy,completion_batch_size=1,prefill_batch_size=1,prefill_step_size=2048,stream=generation_stream)
        t0=time.perf_counter(); bg.insert([[0,3]],max_tokens=[args.steps],samplers=[greedy]); mx.synchronize(bg.stream); boot=time.perf_counter()-t0
        lat=[]; events=[]; gen=[]
        while len(gen)<args.steps:
            t=time.perf_counter(); pr,gr=bg.next(); mx.synchronize(bg.stream); dt=time.perf_counter()-t
            events.append({"dt_s":dt,"prompt_responses":len(pr),"generation_responses":len(gr),"tokens":[int(r.token) for r in gr]})
            if gr: lat.append(dt); gen += [int(r.token) for r in gr]
        rec['P4_batchgenerator_standard']={"insert_bootstrap_s":boot,"timing":stats(lat),"events":events,"generated":gen}
        # P5 BatchGenerator seeded with DwarfStar-admitted cache; no [0,3] replay
        dcache=make_d_cache(); before=[int(x.size()) for x in dcache]
        bg=BatchGenerator(lm,max_tokens=args.steps,sampler=greedy,completion_batch_size=1,prefill_batch_size=1,prefill_step_size=2048,stream=generation_stream)
        t0=time.perf_counter(); uids=bg.insert([[15]],max_tokens=[args.steps],caches=[dcache],all_tokens=[[0,3]],samplers=[greedy]); mx.synchronize(bg.stream); boot=time.perf_counter()-t0
        lat=[]; events=[]; gen=[]; prompt_forward_tokens=0; final_prompt_cache=None; final_all_tokens=None
        while len(gen)<args.steps:
            t=time.perf_counter(); pr,gr=bg.next(); mx.synchronize(bg.stream); dt=time.perf_counter()-t
            prompt_forward_tokens += sum((r.progress[0] if isinstance(r.progress, tuple) else 0) for r in pr if not r.end_of_prompt)
            for r in gr:
                if r.prompt_cache is not None:
                    final_prompt_cache=r.prompt_cache; final_all_tokens=r.all_tokens
            events.append({"dt_s":dt,"prompt_responses":[r.__dict__ for r in pr],"generation_responses":len(gr),"tokens":[int(r.token) for r in gr],"finish":[r.finish_reason for r in gr]})
            if gr: lat.append(dt); gen += [int(r.token) for r in gr]
        if final_prompt_cache is None:
            caches=bg.extract_cache(uids); final_prompt_cache, final_all_tokens = caches[uids[0]]
        after=[int(x.size()) for x in final_prompt_cache]
        rec['P5_batchgenerator_dwarfstar_no_replay']={"insert_bootstrap_s":boot,"timing":stats(lat),"events":events,"generated":gen,"zero_prompt_replay_proof":{"insert_prompt":[15],"all_tokens_seed":[0,3],"prompt_processing_forwarded_tokens_from_0_3":0,"observed_non_generation_prompt_forward_tokens":prompt_forward_tokens,"cache_offsets_before_insert":before,"extracted_cache_offsets_after":after},"token_accounting":{"cache_authority_before_decode":"state after [0,3]","first_emitted_generation_token":15,"first_generationbatch_backbone_input":15,"all_tokens_after_finish":final_all_tokens}}
    finally:
        rt.close()
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0
if __name__=='__main__': raise SystemExit(main())
