"""Small matched installed-candidate 4096/128 greedy OFF/guarded MTP control.

This is a private qualification control, not runtime profile switching. Uses M26's
frozen prompt fixture and stock GenerationBatch; HTTP costs are in composed.json.
"""
import argparse
import dataclasses
import json
from pathlib import Path
import time

from ds41f_mlx.mtp_identity import config, inspect


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    cfg=config();identity=inspect(cfg)
    import deepseek_recipe as d
    import mlx.core as mx
    import omlx.scheduler
    from mlx_lm.generate import BatchGenerator,generation_stream
    from omlx.patches.deepseek_v41.loading import load
    from omlx.patches.mlx_lm_mtp import batch_generator,cache_rollback,prompt_priming
    from ds41f_mlx.runtime.recipe_semantic_guard import RecipeSemanticGuard
    from ds41f_mlx.runtime.mtp_lifecycle import CanonicalTransportHistory,canonical_quiesce_native_singleton
    from tools.run_m26_upstream_mtp_bench import make_prompt
    assert batch_generator.apply() and cache_rollback.apply()
    result=dict(schema='ds41f.m41.performance.v1',status='RUNNING',identity_sha256=identity['identity_sha256'],
                scope='same installed candidate; private greedy stock-JIT math control, not preserved OFF HTTP',runs=[])
    def save():args.output.write_text(json.dumps(result,indent=2)+'\n')
    start=time.perf_counter()
    model,processor=load(cfg.checkpoint_path,preserve_mtp=True,engram_ssd_offload=True)
    result['load_s']=time.perf_counter()-start
    lm=model.language_model
    t=d.Tokenizer.from_file(str(cfg.recipe_path/'static/tokenizers/v41/tokenizer.json'))
    ids=make_prompt(processor.tokenizer,4096)
    try:
        for mode in ('warmup_mtp','OFF','guarded_MTP'):
            prompt_priming.drop_ctx(lm);lm.configure_mtp(mode!='OFF',5)
            processor_native=d.StreamProcessor(d.ChatCompletionChunkGenerator('bench','v41',True,False),d.ParsingOptions(),t)
            guard=RecipeSemanticGuard(processor_native,diagnostic=False)
            history=CanonicalTransportHistory(tuple(ids));tokens=[];bg=None
            try:
                with mx.stream(generation_stream):
                    bg=BatchGenerator(lm,max_tokens=128,sampler=lambda logits:mx.argmax(logits,axis=-1),
                        completion_batch_size=1,prefill_batch_size=1,prefill_step_size=2048,stream=generation_stream)
                    uid=bg.insert([list(ids)],max_tokens=[128])[0]
                    prefill=time.perf_counter()
                    while True:
                        pr,gr=bg.next();mx.synchronize(generation_stream);assert not gr
                        if any(r.end_of_prompt for r in pr):break
                    prefill_s=time.perf_counter()-prefill
                    gb=bg._generation_batch
                    if mode!='OFF':gb._omlx_semantic_guard=guard
                    start=time.perf_counter();first=None;state=None
                    for _ in range(128):
                        pr,gr=bg.next();mx.synchronize(generation_stream);assert len(gr)==1
                        r=gr[0];token=int(r.token)
                        if first is None:first=time.perf_counter()-start
                        if mode=='OFF':guard.events.extend(processor_native.push(d.InferenceChunk.token(token)))
                        history.record_delivered(token);tokens.append(token)
                        h=getattr(gb,'_omlx_semantic_horizon',None)
                        state=getattr(gb,'_omlx_mtp_state',None) or (None if h is None else h.last_state) or state
                        if r.finish_reason:
                            guard.finish_backend(r.finish_reason);break
                    elapsed=time.perf_counter()-start
                    stats=None if state is None else dataclasses.asdict(state.stats)
                    drafted=0 if stats is None else sum(stats['depth_drafted'])
                    row=dict(mode=mode,context=4096,output=len(tokens),prefill_s=prefill_s,decode_s=elapsed,
                        tok_s=len(tokens)/elapsed,first_response_s=first,tokens=tokens,stats=stats,
                        considered_draft_acceptance=None if not drafted else stats['accepts']/drafted,
                        preview_calls=guard.calls)
                    if state is not None:
                        cache=r.prompt_cache if r.finish_reason else gb.prompt_cache
                        quiet=canonical_quiesce_native_singleton(language_model=lm,target_cache=cache,
                            mtp_state=state,history=history,mx=mx,queue_pop=lambda q:q.popleft())
                        quiet.counters.require_m28_zeroes();row['quiescence']=quiet.to_json()
                    bg.remove([uid]);bg.close();bg=None
                    result['runs'].append(row);save()
            finally:
                if bg:
                    with mx.stream(generation_stream):bg.close()
                processor_native.close()
        off,mtp=result['runs'][1:]
        result.update(status='PASS',mtp_off_ratio=mtp['tok_s']/off['tok_s'],
                      token_identity_observed=off['tokens']==mtp['tokens'],
                      policy='workload-dependent practical measurement; no arbitrary speedup threshold')
    except BaseException as exc:result.update(status='FAIL',error=repr(exc));raise
    finally:save()
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))


if __name__=='__main__':main()
