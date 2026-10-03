"""Internal M34 singleton qualification; no public selector or persistence seam.
Run with the M33 native environment. Atomic JSON snapshots preserve failed runs.
"""
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), '/tmp/ds41f-m33-omlx']
import deepseek_recipe as d
import deepseek_recipe._native as native
import mlx.core as mx
import omlx.scheduler
from mlx_lm.generate import generation_stream
from mlx_lm.sample_utils import make_sampler
from omlx.patches.deepseek_v41.loading import load
from omlx.patches.mlx_lm_mtp import batch_generator as mtp, cache_rollback, prompt_priming
from ds41f_mlx.prefill_fp8_mlx import DeferredPrefillAppend, LivePrefillResult, handoff_to_generation
from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig, OMLXDecodeStateAdapter
from ds41f_mlx.runtime.mtp_lifecycle import OMLXMTPGenerationSession, DSparkCommittedContext
from ds41f_mlx.runtime.recipe_semantic_guard import RecipeSemanticGuard
from tools.run_m11_tool_boundary_qualification import initial_body, tool_stub
OUT = ROOT / 'artifacts/m34/soak.json'
OUT.parent.mkdir(parents=True, exist_ok=True)
identities = json.loads((ROOT/'artifacts/m33/runtime-identities.json').read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(native.__file__) == identities['recipe_native_sha256']
assert subprocess.check_output(['git','-C',identities['omlx_path'],'rev-parse','HEAD'],text=True).strip() == identities['omlx_candidate']
for path, digest in identities['omlx_source_hashes'].items():
    assert sha(Path(identities['omlx_path'])/path) == digest
correction=json.loads((ROOT/'artifacts/m34/runtime-correction.json').read_text())
current_runtime_sources=dict(identities['ds41f_source_hashes'])
assert correction['m33_sha256']==current_runtime_sources[correction['path']]
current_runtime_sources[correction['path']]=correction['m34_sha256']
assert sha(ROOT/correction['patch'])==correction['patch_sha256']
for path in ['ds41f_mlx/runtime/recipe_semantic_guard.py','ds41f_mlx/runtime/mtp_lifecycle.py','ds41f_mlx/prefill_fp8_mlx/handoff.py']:
    assert sha(ROOT/path) == current_runtime_sources[path]
result = dict(schema='ds41f.m34.soak.v1', status='RUNNING', base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), runtime_identities=identities, current_runtime_sources=current_runtime_sources, runtime_correction=correction, harness_sha256=sha(__file__), turns=[], resources=[], started=time.time())
def save():
    tmp=OUT.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');tmp.replace(OUT)
def memory(label):
    rss=int(subprocess.check_output(['ps','-p',str(os.getpid()),'-o','rss='],text=True).strip())
    result['resources'].append(dict(label=label,elapsed=time.time()-result['started'],rss_kib=rss,active_bytes=mx.get_active_memory(),cache_bytes=mx.get_cache_memory(),peak_bytes=mx.get_peak_memory(),maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
counts=dict(verify=0,proposal=0,replay=0,repack=0)
def forbidden(*a,**kw):
    counts['replay']+=1;raise AssertionError('history replay forbidden')
def repack(*a,**kw):
    counts['repack']+=1;raise AssertionError('cache reconstruction/repack forbidden')
mtp._reconcile_mtp_to_standard=forbidden
OMLXDecodeStateAdapter._pack_cache_array=repack
OMLXDecodeStateAdapter.admit=repack
assert mtp.apply() and cache_rollback.apply()
old_cycle=mtp._run_verify_cycle_chain
cycle_rows=[]
def cycle(gb,state,*a,**kw):
    counts['verify']+=1;before=state.stats.accepts
    ret=old_cycle(gb,state,*a,**kw)
    target=[int(c.size()) for c in gb.prompt_cache];rings=[int(c.offset) for c in state.mtp_cache]
    assert len(set(target+rings))==1
    cycle_rows.append(dict(accepted=state.stats.accepts-before,target=target[0],history=len(gb.tokens[0]),queue=len(state.queue)))
    return ret
mtp._run_verify_cycle_chain=cycle
session=None
save()
try:
    model,_=load(Path(identities['checkpoint']),preserve_mtp=True,engram_ssd_offload=True)
    lm=model.language_model;lm.configure_mtp(True,5);lm._p7_enable_overlap=True
    old_proposal=lm.dspark_forward
    def proposal(*a,**kw):
        counts['proposal']+=1;return old_proposal(*a,**kw)
    lm.dspark_forward=proposal
    tokenizer=d.Tokenizer.from_file('/tmp/ds41f-m32-recipe/static/tokenizers/v41/tokenizer.json')
    control=tuple(tokenizer.encode('<｜end▁of▁sentence｜>'))
    memory('model_loaded')
    make_cache=lm.make_cache
    with mx.stream(generation_stream):
        for fresh in range(3):
            prompt_priming.drop_ctx(lm)
            cache=make_cache();rings=lm.make_mtp_cache();canonical=[]
            lm.make_cache=repack
            ring_ids=[id(c) for c in rings]
            body=initial_body('Paris');body['stop']=None
            for turn in range(30):
                kind=turn%3
                limit=128 if kind<2 else 768
                mx.random.seed(3400+fresh*100+turn)
                converted=d.ChatCompletionRequest(body).convert(d.ConversionOptions(default_thinking_mode=False))
                ids=d.DeepseekV41Encoding().with_tokenizer(tokenizer).encode(converted.conversation)
                assert ids[:len(canonical)]==canonical, 'ordinary recipe continuation must preserve canonical prefix'
                C=len(canonical);target=len(ids)-1
                taps={i:[] for i in lm._config.dspark_target_layer_ids}
                original={i:lm.layers[i] for i in taps}
                class Tap:
                    def __init__(self,index,layer):self.index,self.layer=index,layer
                    def __getattr__(self,name):return getattr(self.layer,name)
                    def __call__(self,h,*a,**kw):
                        reduced=mx.mean(h,axis=-2)
                        if reduced.ndim==2:reduced=reduced[None]
                        taps[self.index].append(reduced)
                        return self.layer(h,*a,**kw)
                for i,layer in original.items():lm.layers[i]=Tap(i,layer)
                prestart=time.perf_counter()
                try:
                    app=DeferredPrefillAppend.create(lm,cache,ids[:-1],committed_frontier=C,mx=mx);app.execute_all()
                finally:
                    for i,layer in original.items():lm.layers[i]=layer
                hidden=mx.concatenate([mx.concatenate(taps[i],axis=1) for i in taps],axis=-1)
                lm.dspark_append_context(hidden,rings,start_offset=C);mx.eval(*[c.keys for c in rings]);mx.synchronize(generation_stream)
                context=DSparkCommittedContext.from_native(rings,frontier=target,target_layer_ids=lm._config.dspark_target_layer_ids)
                prefill=LivePrefillResult.from_committed(app.commit_certificate,prefix_token_ids=ids[:-1]);prefill.dspark_committed_context=context
                assert app.final_execution.runner.full_cache_repack_count==0
                p=d.StreamProcessor(d.ChatCompletionRequest.chunk_generator(converted,'m34','v41'),converted.parsing_options,tokenizer)
                reference=d.StreamProcessor(d.ChatCompletionRequest.chunk_generator(converted,'m34','v41'),converted.parsing_options,tokenizer)
                guard=RecipeSemanticGuard(p,diagnostic=False,control_token_ids=control)
                cfg=OMLXDecodeConfig(omlx_path=Path(identities['omlx_path']),checkpoint_path=Path(identities['checkpoint']),preserve_mtp=True,speculation_enabled=True,stop_token_ids=control)
                class Factory:
                    @classmethod
                    def from_prefilled_cache(cls,model_arg,cache_arg,prefix,config,*,max_tokens,sampler):
                        return OMLXMTPGenerationSession(model_arg,cache_arg,np.asarray(prefix),context,config=config,sampler=sampler,max_tokens=max_tokens,semantic_guard=guard)
                session=handoff_to_generation(prefill,model,terminal_prompt_token=ids[-1],config=cfg,max_tokens=limit,sampler=make_sampler(temp=1.,top_p=1.,top_k=0),session_factory=Factory)
                preseconds=time.perf_counter()-prestart
                start=time.perf_counter();first=None;latencies=[];ack=[];last_state=None
                # A final long stream is cancelled at a returned-response boundary;
                # intervening turns exercise temporary transport stalls and exact ACKs.
                cancel_at=193 if turn==29 else None
                for step in range(limit):
                    pending=step%47 in (20,21,22)
                    tick=time.perf_counter();token=session.next_token(transport_delivered=not pending);latencies.append(time.perf_counter()-tick)
                    assert token is not None
                    if first is None:first=time.perf_counter()-start
                    if token not in control:reference.push(d.InferenceChunk.token(token))
                    if guard.finished and not reference.finished:reference.push(d.InferenceChunk.finish(d.InferenceFinishReason.Stop))
                    assert p.semantic_snapshot()==reference.semantic_snapshot()
                    if pending and (step%47==22 or guard.finished or session.last_response.finish_reason):
                        ordinal=len(session.history.transport_delivered_tokens);suffix=session.history.recovery_suffix_tokens
                        before=dict(counts);snapshot=p.semantic_snapshot();tick=time.perf_counter()
                        session.confirm_delivery(suffix,start_ordinal=ordinal)
                        assert before==counts and snapshot==p.semantic_snapshot()
                        ack.append(dict(ordinal=ordinal,tokens=len(suffix),seconds=time.perf_counter()-tick,metadata_only=True))
                    gb=session._bg._generation_batch;h=getattr(gb,'_omlx_semantic_horizon',None)
                    last_state=getattr(gb,'_omlx_mtp_state',None) or (None if h is None else h.last_state) or last_state
                    if guard.finished or session.last_response.finish_reason or step==cancel_at:break
                seconds=time.perf_counter()-start
                stats=dataclasses.asdict(last_state.stats) if last_state is not None else None
                native_cache=h.last_cache if h is not None and h.last_state is not None else gb.prompt_cache
                native_slots=tuple(native_cache)
                extracted=[];old_extract=gb.extract_cache
                def extract_row(index):
                    assert index==0
                    row=old_extract(index);extracted.append(tuple(row));return row
                gb.extract_cache=extract_row
                before=dict(counts);tick=time.perf_counter();quiet=session.cancel() if cancel_at is not None else session.quiesce();quietseconds=time.perf_counter()-tick
                for token in quiet.queue_drained_tokens:
                    if token not in control:reference.push(d.InferenceChunk.token(token))
                assert p.semantic_snapshot()==reference.semantic_snapshot()
                delta={k:counts[k]-before[k] for k in counts}
                assert not any(delta.values())
                assert all(c.size()==len(quiet.canonical_tokens) for c in quiet.target_cache)
                assert set(quiet.dspark_context.offsets)=={len(quiet.canonical_tokens)}
                # Native finish extraction returns row views, not the admission wrappers.
                expected_slots=extracted[0] if extracted else native_slots
                assert len(extracted)<=1
                assert len(quiet.target_cache)==40 and all(a is b for a,b in zip(quiet.target_cache,expected_slots))
                if extracted:assert quiet.target_cache is not native_cache
                assert [id(c) for c in quiet.dspark_context.caches]==ring_ids
                assert h is None or h.pending is None
                assert last_state is None or not last_state.queue
                if quiet.recovery_suffix_tokens:
                    session.confirm_delivery(quiet.recovery_suffix_tokens,start_ordinal=len(session.history.transport_delivered_tokens))
                response=d.ChatCompletionResponse('m34','v41',0,0,0)
                for event in guard.events:response.append(event)
                protocol=json.loads(response.to_json());times=sorted(guard.latencies_ns)
                rec=dict(fresh_session=fresh,turn=turn,kind=['tool','tool_result','long_text'][kind],prompt_frontier=len(ids),suffix_start=C,canonical_frontier=len(quiet.canonical_tokens),prefix_preserving=True,cache_ownership_preserved=True,native_idle_row_extractions=len(extracted),retired_prediction=True,queue_empty=True,target_offsets=[c.size() for c in quiet.target_cache],dspark_offsets=list(quiet.dspark_context.offsets),generated=step+1,decode_seconds=seconds,tok_s=(step+1)/seconds,first_response_seconds=first,prefill_handoff_seconds=preseconds,stats=stats,preview_calls=guard.calls,preview_median_ns=times[len(times)//2] if times else None,preview_max_ns=max(times) if times else None,delivery_acks=ack,quiescence_seconds=quietseconds,quiescence=quiet.to_json(),quiescence_counts=delta,protocol=protocol,terminal_matches=list(guard.matches),cycles=list(cycle_rows),latency_quarters=[sum(latencies[i:i+64])/len(latencies[i:i+64]) for i in range(0,len(latencies),64)],cancelled=cancel_at is not None and step==cancel_at,prompt_replay=session.prompt_replay_count)
                result['turns'].append(rec);cycle_rows.clear()
                cache=quiet.target_cache;rings=list(quiet.dspark_context.caches);canonical=list(quiet.canonical_tokens)
                session.close();session=None;p.close();reference.close()
                assert len(cache)==40 and all(c.size()==len(canonical) for c in cache), 'idle authority must survive owner close'
                memory(f'session{fresh}_turn{turn}');save()
                if turn==29:break
                choice=protocol['choices'][0];message=choice['message']
                body['messages'].append(message)
                if kind==0:
                    assert choice['finish_reason']=='tool_calls' and len(message['tool_calls'])==1
                    call=message['tool_calls'][0]
                    tool_result=tool_stub(call['function']['name'],call['function']['arguments'])
                    body['messages'].append(dict(role='tool',tool_call_id=call['id'],content=tool_result));body['tool_choice']='auto'
                elif kind==1:
                    assert choice['finish_reason']=='stop'
                    body['messages'].append(dict(role='user',content='Write a detailed practical guide to designing a reliable local weather assistant. Cover tool errors, caching, testing and user experience in eight numbered sections, with examples. Aim for about 450 words. Do not call a tool.'))
                    body['tool_choice']='auto'
                else:
                    assert choice['finish_reason']=='stop', 'long response must close before ordinary user re-entry'
                    body['messages'].append(dict(role='user',content='Now use lookup_weather for Berlin, and then answer concisely.'))
                    body['tool_choice']='auto'
            del cache,rings,canonical,hidden,taps,app,prefill,context,quiet
            prompt_priming.drop_ctx(lm);mx.synchronize(generation_stream);mx.clear_cache();memory(f'fresh{fresh}_cleanup');save()
    result.update(status='PASS',counts=counts,elapsed_seconds=time.time()-result['started'])
except Exception as exc:
    result.update(status='FAIL',error=repr(exc));raise
finally:
    if session is not None:
        with mx.stream(generation_stream):session.close()
    save()
print(json.dumps(dict(status=result['status'],turns=len(result['turns']),counts=counts)))
