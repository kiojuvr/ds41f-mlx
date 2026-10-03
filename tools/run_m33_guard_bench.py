"""Bounded paired M26-shape benchmark (4K/128), not operational qualification."""
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),'/tmp/ds41f-m33-omlx']
import deepseek_recipe as d
import deepseek_recipe._native as native
import mlx.core as mx
import omlx.scheduler
from mlx_lm.generate import BatchGenerator,generation_stream
from omlx.patches.deepseek_v41.loading import load
from omlx.patches.mlx_lm_mtp import batch_generator as mtp,cache_rollback,prompt_priming
from ds41f_mlx.runtime.recipe_semantic_guard import RecipeSemanticGuard
from ds41f_mlx.runtime.mtp_lifecycle import CanonicalTransportHistory,canonical_quiesce_native_singleton
from tools.run_m26_upstream_mtp_bench import make_prompt
OUT=ROOT/'artifacts/m33/performance.json'
result=dict(status='RUNNING',scope='One warmup + paired 4K/128 diagnostic; not broad soak',native_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),runs=[])
def save():OUT.write_text(json.dumps(result,indent=2)+'\n')
assert mtp.apply() and cache_rollback.apply()
model=None
try:
    model,processor=load(Path('/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash'),preserve_mtp=True,engram_ssd_offload=True)
    lm=model.language_model
    t=d.Tokenizer.from_file('/tmp/ds41f-m32-recipe/static/tokenizers/v41/tokenizer.json')
    ids=make_prompt(processor.tokenizer,4096)
    def sampler(logits):return mx.random.categorical(logits)
    for mode in ['warmup_guarded','OFF','ON_no_guard','ON_guarded']:
        prompt_priming.drop_ctx(lm);lm.configure_mtp(mode!='OFF',5);mx.random.seed(12345+4096)
        p=d.StreamProcessor(d.ChatCompletionChunkGenerator('bench','v41',True,False),d.ParsingOptions(),t)
        guard=RecipeSemanticGuard(p,diagnostic=False)
        history=CanonicalTransportHistory(tuple(ids));steps=[];bg=None
        try:
            with mx.stream(generation_stream):
                bg=BatchGenerator(lm,max_tokens=128,sampler=sampler,completion_batch_size=1,prefill_batch_size=1,prefill_step_size=2048,stream=generation_stream)
                uid=bg.insert([ids],max_tokens=[128])[0]
                while True:
                    pr,gr=bg.next();mx.synchronize(generation_stream)
                    assert not gr
                    if any(r.end_of_prompt for r in pr):break
                gb=bg._generation_batch
                guarded=mode in ['warmup_guarded','ON_guarded']
                if guarded:gb._omlx_semantic_guard=guard
                start=time.perf_counter();first=None;last_state=None
                for i in range(128):
                    pr,gr=bg.next();mx.synchronize(generation_stream)
                    assert len(gr)==1
                    response=gr[0];token=int(response.token)
                    if first is None:first=time.perf_counter()-start
                    if not guarded:guard.events.extend(p.push(d.InferenceChunk.token(token)))
                    history.record_delivered(token)
                    h=getattr(gb,'_omlx_semantic_horizon',None)
                    last_state=getattr(gb,'_omlx_mtp_state',None) or (None if h is None else h.last_state) or last_state
                    steps.append(token)
                    if response.finish_reason:
                        guard.finish_backend(response.finish_reason);break
                seconds=time.perf_counter()-start
                times=sorted(guard.latencies_ns)
                state_stats=None if last_state is None else dataclasses.asdict(last_state.stats)
                drafted=0 if state_stats is None else sum(state_stats['depth_drafted'])
                cycles=0 if state_stats is None else state_stats['cycles']
                rec=dict(mode=mode,context=4096,tokens=steps,decode_seconds=seconds,tok_s=len(steps)/seconds,first_token_seconds=first,
                         stats=state_stats,acceptance=None if not drafted else state_stats['accepts']/drafted,
                         accepted_per_cycle=None if not cycles else state_stats['accepts']/cycles,
                         preview=dict(calls=guard.calls,init_calls=2 if guarded else 0,calls_per_cycle=None if not cycles or not guarded else (guard.calls-2)/cycles,
                                      candidate_ids=guard.candidate_ids,median_ns=None if not times else times[len(times)//2],p95_ns=None if not times else times[int(len(times)*.95)],max_ns=None if not times else times[-1],
                                      metadata='one owned terminal, <=16 bounded diagnostic decisions; no terminal on this performance corpus'))
                result['runs'].append(rec);save()
                if last_state is not None:
                    cache=response.prompt_cache if response.finish_reason else gb.prompt_cache
                    quiet=canonical_quiesce_native_singleton(language_model=lm,target_cache=cache,mtp_state=last_state,history=history,mx=mx,queue_pop=lambda q:q.popleft())
                    mx.synchronize(generation_stream);rec['quiescence']=quiet.to_json()
                bg.remove([uid]);bg.close();bg=None
        finally:
            if bg is not None:
                with mx.stream(generation_stream):bg.close()
            p.close();save()
    plain=next(r for r in result['runs'] if r['mode']=='ON_no_guard');guarded=next(r for r in result['runs'] if r['mode']=='ON_guarded')
    assert plain['tokens']==guarded['tokens'], 'no-boundary guard changed model tokens'
    assert plain['stats']['accepts']==guarded['stats']['accepts']
    result['guard_vs_no_guard_ratio']=guarded['tok_s']/plain['tok_s']
    result['M26_class_materially_intact']=guarded['tok_s']>=34 and result['guard_vs_no_guard_ratio']>=.9
    result['status']='PASS' if result['M26_class_materially_intact'] else 'PERFORMANCE_REGRESSION'
finally:save()
print(json.dumps(dict(status=result['status'],runs=[dict(mode=r['mode'],tok_s=r['tok_s'],acceptance=r['acceptance'],accepted_per_cycle=r['accepted_per_cycle'],preview=r['preview']) for r in result['runs']]),indent=2))
