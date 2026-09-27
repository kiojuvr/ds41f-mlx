#!/usr/bin/env python3
"""M4 oMLX standard-generation and prefill/admission state diagnostics."""
from __future__ import annotations

import argparse, json, os, sys, time, hashlib, importlib
from pathlib import Path
from statistics import median, mean
from typing import Any

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT, DEFAULT_OMLX, OmlxRuntime, OmlxRuntimeConfig
from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig, OMLXDecodeSession
from tools.run_m4_omlx_base_decode_qualification import build_prefill_state


def arr_digest(x: Any) -> dict[str, Any] | None:
    if x is None: return None
    import mlx.core as mx
    try:
        mx.eval(x)
        a=np.asarray(x)
        return {"shape":list(a.shape),"dtype":str(a.dtype),"sha256":hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()}
    except Exception as e:
        return {"type":type(x).__name__,"repr":repr(x)[:200],"error":repr(e)}


def cache_summary(cache: list[Any]) -> list[dict[str, Any]]:
    out=[]
    for li,c in enumerate(cache):
        slots={str(i):arr_digest(v) for i,v in enumerate(getattr(c,'cache',[]))}
        out.append({"layer":li,"compress_ratio":getattr(c,'compress_ratio',None),"meta_state":list(getattr(c,'meta_state',())),"slots":slots})
    return out


def compare_summaries(a,b):
    diffs=[]
    for la,lb in zip(a,b):
        for si in range(7):
            x=la['slots'].get(str(si)); y=lb['slots'].get(str(si))
            if x!=y:
                diffs.append({"layer":la['layer'],"slot":si,"ordinary":x,"admitted":y})
                break
    return diffs


def perf_stats(lat):
    return {"steps":len(lat),"first_step_latency_s":lat[0] if lat else None,"median_s":median(lat) if lat else None,"mean_s":mean(lat) if lat else None,"tok_per_s_median":(1/median(lat) if lat else None),"latencies_s":lat}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',str(DEFAULT_CHECKPOINT)))
    ap.add_argument('--omlx-path',default=os.environ.get('DS41F_OMLX',str(DEFAULT_OMLX)))
    ap.add_argument('--out',default='artifacts/m4/omlx-generation-state-controls/result.json')
    ap.add_argument('--prompt-tokens',default='0,3')
    ap.add_argument('--max-tokens',type=int,default=8)
    args=ap.parse_args()
    omlx=Path(args.omlx_path); ck=Path(args.checkpoint); prompt=[int(x) for x in args.prompt_tokens.split(',') if x]
    if str(omlx) not in sys.path: sys.path.insert(0,str(omlx))
    rec={"schema":"ds41f.m4.omlx-generation-state-controls.v1","checkpoint":str(ck),"omlx_path":str(omlx),"prompt_tokens":prompt,"mtp_dspark_off":True}
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator
    # Import scheduler for normal oMLX GenerationBatch monkey patches without starting HTTP engine.
    importlib.import_module('omlx.scheduler')
    from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast
    rec['native']={"is_native_available":glm_fast.is_native_available(),"import_error":repr(glm_fast.import_error()),"native_symbols":list(glm_fast.native_symbols())}
    rt=OmlxRuntime(OmlxRuntimeConfig(omlx_path=omlx,checkpoint_path=ck,engram_ssd_offload=True,preserve_mtp=False))
    model,_=rt.load_model(); lm=model.language_model
    try:
        def greedy(logits): return mx.argmax(logits, axis=-1)
        bg=BatchGenerator(lm,max_tokens=args.max_tokens,sampler=greedy,completion_batch_size=1,prefill_batch_size=1,prefill_step_size=2048)
        bg.insert([prompt],max_tokens=[args.max_tokens],samplers=[greedy])
        events=[]; gen_lat=[]; gen=[]; prefill_calls=0
        while len(gen)<args.max_tokens:
            t=time.perf_counter(); pr,gr=bg.next(); mx.synchronize(); dt=time.perf_counter()-t
            events.append({"dt_s":dt,"prompt_responses":len(pr),"generation_responses":len(gr),"tokens":[int(r.token) for r in gr],"finish":[r.finish_reason for r in gr]})
            if pr: prefill_calls+=1
            if gr:
                gen_lat.append(dt); gen.extend(int(r.token) for r in gr)
                if any(r.finish_reason for r in gr): break
        rec['A_standard_batchgenerator_mtp_off']={"generated_tokens":gen,"prefill_next_events":events,"prefill_calls":prefill_calls,"decode_timing":perf_stats(gen_lat)}

        ordinary_cache=lm.make_cache(); mx.eval(lm._forward(mx.array([prompt],mx.int64),cache=ordinary_cache)); mx.synchronize()
        rec['ordinary_prefill_cache_summary']=cache_summary(ordinary_cache)
        prefill=build_prefill_state(ck,Path('artifacts/m4/omlx-generation-state-controls/native'),prompt)
        admitted=OMLXDecodeSession.from_prefill_state(model,prefill.continuation_state,OMLXDecodeConfig(omlx_path=omlx,checkpoint_path=ck,engram_ssd_offload=True,preserve_mtp=False))
        rec['admitted_cache_summary']=cache_summary(admitted.cache)
        diffs=compare_summaries(rec['ordinary_prefill_cache_summary'],rec['admitted_cache_summary'])
        rec['ordinary_vs_admitted']={"all_layer_first_differences":diffs,"first_difference":diffs[0] if diffs else None,"equivalent_by_digest":not diffs}
    finally:
        rt.close()
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0
if __name__=='__main__': raise SystemExit(main())
