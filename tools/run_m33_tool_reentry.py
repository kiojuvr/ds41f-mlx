"""One guarded native tool-result P6/P5 re-entry, not an operational soak."""
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),'/tmp/ds41f-m33-omlx']
import deepseek_recipe as d
import deepseek_recipe._native as native
import mlx.core as mx
import omlx.scheduler
from mlx_lm.generate import generation_stream
from mlx_lm.sample_utils import make_sampler
from omlx.patches.deepseek_v41.loading import load
from omlx.patches.mlx_lm_mtp import batch_generator as mtp,cache_rollback
from ds41f_mlx.prefill_fp8_mlx import DeferredPrefillAppend,LivePrefillResult,handoff_to_generation
from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig,OMLXDecodeStateAdapter
from ds41f_mlx.runtime.mtp_lifecycle import OMLXMTPGenerationSession,DSparkCommittedContext
from ds41f_mlx.runtime.recipe_semantic_guard import RecipeSemanticGuard
from tools.run_m11_tool_boundary_qualification import initial_body,continuation_body,tool_stub
from types import SimpleNamespace as NS
OUT=ROOT/'artifacts/m33/tool-reentry.json'
checkpoint=Path('/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash')
t=d.Tokenizer.from_file('/tmp/ds41f-m32-recipe/static/tokenizers/v41/tokenizer.json')
result=dict(status='RUNNING',stage='load',native_path=native.__file__,native_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),turns=[],suffixes=[])
def save():OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
counts=dict(verify=0,proposal=0,replay=0,repack=0,forward=0)
def forbidden(*args,**kwargs):
    counts['replay']+=1
    raise AssertionError('history reconciliation/reconstruction forbidden')
def forbidden_repack(*args,**kwargs):
    counts['repack']+=1
    raise AssertionError('portable admission/repack forbidden')
mtp._reconcile_mtp_to_standard=forbidden
OMLXDecodeStateAdapter._pack_cache_array=forbidden_repack
OMLXDecodeStateAdapter.admit=forbidden_repack
assert mtp.apply() and cache_rollback.apply()
old_cycle=mtp._run_verify_cycle_chain
cycles=[]
def cycle(gb,state,*args,**kwargs):
    counts['verify']+=1
    before=state.stats.accepts
    ret=old_cycle(gb,state,*args,**kwargs)
    cycles.append(dict(k=int(state.stats.depth_drafted.__len__()),accepted=state.stats.accepts-before,history=len(gb.tokens[0]),target=[int(c.size()) for c in gb.prompt_cache],dspark=[int(c.offset) for c in state.mtp_cache]))
    return ret
mtp._run_verify_cycle_chain=cycle
model=session=None
try:
    model,_=load(checkpoint,preserve_mtp=True,engram_ssd_offload=True)
    lm=model.language_model;lm.configure_mtp(True,5)
    lm._p7_enable_overlap=True
    old_proposal=lm.dspark_forward
    def proposal(*args,**kw):
        counts['proposal']+=1
        return old_proposal(*args,**kw)
    lm.dspark_forward=proposal
    forward_spans=[]
    old_forward=lm._forward
    def forward(ids,cache=None,**kw):
        counts['forward']+=1
        forward_spans.append(dict(start=cache[0].size(),ids=ids.tolist()[0]))
        return old_forward(ids,cache=cache,**kw)
    lm._forward=forward
    cache=lm.make_cache();rings=lm.make_mtp_cache()
    def no_new_target_cache(*args,**kwargs):
        counts['repack']+=1
        raise AssertionError('target cache reconstruction after initial ownership forbidden')
    lm.make_cache=no_new_target_cache
    canonical=[]
    tool_body=initial_body('Paris');tool_body['stop']=None
    body=tool_body
    with mx.stream(generation_stream):
        for turn in range(2):
            mx.random.seed(3301+turn)
            converted=d.ChatCompletionRequest(body).convert(d.ConversionOptions(default_thinking_mode=False))
            ids=d.DeepseekV41Encoding().with_tokenizer(t).encode(converted.conversation)
            assert ids[:len(canonical)]==canonical, 'recipe re-encoding is not an exact canonical prefix extension'
            result['stage']=f'turn{turn}_P7_P6'
            C=len(canonical);target=len(ids)-1
            assert 0 < target-C <= 8192, 'bounded ordinary P6 tap qualification only'
            taps={i:[] for i in lm._config.dspark_target_layer_ids}
            original_layers={i:lm.layers[i] for i in taps}
            class Tap:
                def __init__(self,index,layer):self.index,self.layer=index,layer
                def __getattr__(self,name):return getattr(self.layer,name)
                def __call__(self,h,*args,**kw):
                    reduced=mx.mean(h,axis=-2)
                    if reduced.ndim==2:reduced=reduced[None]
                    taps[self.index].append(reduced)
                    return self.layer(h,*args,**kw)
            for i,layer in original_layers.items():lm.layers[i]=Tap(i,layer)
            try:
                app=DeferredPrefillAppend.create(lm,cache,ids[:-1],committed_frontier=C,mx=mx)
                app.execute_all()
            finally:
                for i,layer in original_layers.items():lm.layers[i]=layer
            hidden=mx.concatenate([mx.concatenate(taps[i],axis=1) for i in lm._config.dspark_target_layer_ids],axis=-1)
            assert hidden.shape[1]==target-C
            lm.dspark_append_context(hidden,rings,start_offset=C)
            mx.eval(*[c.keys for c in rings]);mx.synchronize(generation_stream)
            context=DSparkCommittedContext.from_native(rings,frontier=target,target_layer_ids=lm._config.dspark_target_layer_ids)
            prefill=LivePrefillResult.from_committed(app.commit_certificate,prefix_token_ids=ids[:-1])
            prefill.dspark_committed_context=context
            result['suffixes'].append(dict(C=C,T=target,ids=ids[C:-1],P6_plan=app.plan.to_json(),P7_enabled=True, same_forward_tap_rows=int(hidden.shape[1]),
                                           P7_events=[dict(e) for r in app.segment_records for e in r.scheduling_events],
                                           target=[c.size() for c in cache],dspark=list(context.offsets),cache_id=id(cache),ring_ids=[id(c) for c in rings],
                                           full_cache_repack=app.final_execution.runner.full_cache_repack_count))
            processor=d.StreamProcessor(d.ChatCompletionRequest.chunk_generator(converted,f'm33-reentry-{turn}','v41'),converted.parsing_options,t)
            control_ids=tuple(t.encode('<｜end▁of▁sentence｜>'))
            guard=RecipeSemanticGuard(processor,diagnostic=True,control_token_ids=control_ids)
            cfg=OMLXDecodeConfig(omlx_path=Path('/tmp/ds41f-m33-omlx'),checkpoint_path=checkpoint,preserve_mtp=True,speculation_enabled=True,stop_token_ids=control_ids)
            class Factory:
                @classmethod
                def from_prefilled_cache(cls,model_arg,cache_arg,prefix,config,*,max_tokens,sampler):
                    assert cache_arg is cache
                    return OMLXMTPGenerationSession(model=model_arg,initial_cache=cache_arg,initial_token_ids=np.asarray(prefix),dspark_context=context,config=config,sampler=sampler,max_tokens=max_tokens,semantic_guard=guard)
            result['stage']=f'turn{turn}_P5_MTP'
            session=handoff_to_generation(prefill,model,terminal_prompt_token=ids[-1],config=cfg,max_tokens=128,sampler=make_sampler(temp=1.,top_p=1.,top_k=0),session_factory=Factory)
            for _ in range(128):
                token=session.next_token()
                assert token is not None
                if guard.finished or session.last_response.finish_reason:
                    if not guard.finished:guard.finish_backend(session.last_response.finish_reason)
                    break
            else:raise AssertionError('re-entry response did not finish')
            before=dict(counts);before_spans=len(forward_spans)
            quiet=session.quiesce();mx.synchronize(generation_stream)
            delta={k:counts[k]-before[k] for k in counts}
            assert delta['verify']==delta['proposal']==delta['replay']==delta['repack']==0
            cache=quiet.target_cache;rings=list(quiet.dspark_context.caches);canonical=list(quiet.canonical_tokens)
            response=d.ChatCompletionResponse(f'm33-reentry-{turn}','v41',0,0,0)
            for event in guard.events:response.append(event)
            protocol=json.loads(response.to_json())
            result['turns'].append(dict(protocol=protocol,canonical_tokens=canonical,quiescence=quiet.to_json(),quiescence_counts=delta,quiescence_forwards=forward_spans[before_spans:],predictions=list(guard.matches),prompt_replay=session.prompt_replay_count,cycles=list(cycles),ring_ids=[id(c) for c in rings],target=[c.size() for c in cache]))
            cycles.clear();session.close();session=None;processor.close();save()
            if turn==0:
                choice=protocol['choices'][0]
                assert choice['finish_reason']=='tool_calls'
                calls=choice['message']['tool_calls'];assert len(calls)==1
                parsed=calls[0];call=NS(id=parsed['id'],name=parsed['function']['name'],arguments=parsed['function']['arguments'])
                tool_result=tool_stub(call.name,call.arguments)
                body=continuation_body(tool_body,call,tool_result);body['stop']=None
                result['tool']=dict(name=call.name,arguments=json.loads(call.arguments),result=json.loads(tool_result))
            else:
                assert protocol['choices'][0]['finish_reason']=='stop'
                assert not protocol['choices'][0]['message'].get('tool_calls')
                assert protocol['choices'][0]['message'].get('content')
        result.update(status='PASS',stage='complete',counts=counts,model_history_replay=0,full_cache_repack=0)
finally:
    if session is not None:
        with mx.stream(generation_stream):session.close()
    save()
print(json.dumps(dict(status=result['status'],stage=result['stage'],tool=result.get('tool')),indent=2))
