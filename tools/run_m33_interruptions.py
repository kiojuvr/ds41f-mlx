"""Bounded live cancellation and fail-closed synchronous init fault probes."""
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
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
from omlx.patches.mlx_lm_mtp import batch_generator as mtp,cache_rollback,prompt_priming
from ds41f_mlx.runtime.mtp_lifecycle import OMLXMTPGenerationSession,DSparkCommittedContext,MTPLifecycleError
from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
from ds41f_mlx.runtime.recipe_semantic_guard import RecipeSemanticGuard
from tokenizers import Tokenizer
OUT=ROOT/'artifacts/m33/interruptions.json'
t=d.Tokenizer.from_file('/tmp/ds41f-m32-recipe/static/tokenizers/v41/tokenizer.json')
decode=Tokenizer.from_file('/tmp/ds41f-m32-recipe/static/tokenizers/v41/tokenizer.json')
result=dict(status='RUNNING',native_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),cases=[])
def save():OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
counts=dict(verify=0,proposal=0,forward=0,reconcile=0)
old_cycle=mtp._run_verify_cycle_chain
old_backbone=mtp._call_backbone_captured
old_resolve=mtp._resolve_sampler
old_init=mtp._post_init_mtp
fault=None
forced_successor=None

def cycle(*args,**kwargs):
    counts['verify']+=1
    return old_cycle(*args,**kwargs)
def reconcile(*args,**kwargs):
    counts['reconcile']+=1
    raise AssertionError('history reconciliation forbidden')
def backbone(*args,**kwargs):
    out=old_backbone(*args,**kwargs)
    if fault=='after_safe_main_forward':raise TimeoutError(fault)
    return out
def resolve(gb):
    sampler=old_resolve(gb)
    def sample(logits):
        out=sampler(logits) if forced_successor is None else mx.array([forced_successor],dtype=mx.uint32)
        if fault=='after_successor_sample':raise TimeoutError(fault)
        return out
    return sample
def init(*args,**kwargs):
    out=old_init(*args,**kwargs)
    if fault=='after_init_construction':raise TimeoutError(fault)
    return out
mtp._run_verify_cycle_chain=cycle;mtp._reconcile_mtp_to_standard=reconcile
mtp._call_backbone_captured=backbone;mtp._resolve_sampler=resolve;mtp._post_init_mtp=init
assert mtp.apply() and cache_rollback.apply()
model=session=None
try:
    model,_=load(Path('/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash'),preserve_mtp=True,engram_ssd_offload=True)
    lm=model.language_model;lm.configure_mtp(True,5)
    old_forward,old_proposal=lm._forward,lm.dspark_forward
    spans=[]
    def forward(ids,cache=None,**kw):
        counts['forward']+=1;spans.append(dict(start=cache[0].size(),ids=ids.tolist()[0]))
        return old_forward(ids,cache=cache,**kw)
    def proposal(*args,**kw):
        counts['proposal']+=1
        return old_proposal(*args,**kw)
    lm._forward,lm.dspark_forward=forward,proposal
    names=['before_init','after_first_safe_emit','predicted_successor_before_emit','safe_undelivered_prefix','inside_DSML','before_DSML_closure','after_terminal_emit','after_safe_main_forward','after_successor_sample','after_init_construction','native_fixture_pending_main','native_fixture_pending_successor','native_fixture_stop_prefix_main','alignment_successor_terminal','alignment_prediction_interruption','canonical_terminal_transport_pending']
    for name in names:
        prompt_priming.drop_ctx(lm);mx.random.seed(3201);fault=None;forced_successor=None
        tool='DSML' in name or name=='safe_undelivered_prefix'
        req={'model':'v41','messages':[{'role':'user','content':('Call lookup for Paris and London, both in this response. Do not answer without both calls.' if tool else 'Say hello in one short sentence.')}], 'stop':None}
        if tool:req['tools']=[{'type':'function','function':{'name':'lookup','description':'Return weather for a city.','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}]
        if name in ['predicted_successor_before_emit']:req['stop']=[' there']
        if name in ['after_terminal_emit','canonical_terminal_transport_pending']:req['stop']=['Hello']
        if name.startswith('alignment_'):req['stop']=['!']
        converted=d.ChatCompletionRequest(req).convert(d.ConversionOptions(default_thinking_mode=False))
        ids=d.DeepseekV41Encoding().with_tokenizer(t).encode(converted.conversation)
        seed=[];forced_main=None;opts=converted.parsing_options
        if name.startswith('native_fixture_pending'):
            emoji=t.encode('🙂');assert len(emoji)==2
            opts.stop_sequences=['🙂']
            if name.endswith('_main'):seed=emoji[:-1];forced_main=emoji[-1];ids+=seed
            else:forced_main=emoji[0];forced_successor=emoji[1]
        elif name=='native_fixture_stop_prefix_main':
            seed=t.encode('HALT');ids+=seed;forced_main=t.encode(' NOW')[0]
            opts.stop_sequences=['HALT NOW']
        processor=d.StreamProcessor(d.ChatCompletionRequest.chunk_generator(converted,'cancel','v41'),opts,t)
        reference=d.StreamProcessor(d.ChatCompletionRequest.chunk_generator(converted,'cancel','v41'),opts,t)
        for token in seed:
            processor.push(d.InferenceChunk.token(token));reference.push(d.InferenceChunk.token(token))
        guard=RecipeSemanticGuard(processor,diagnostic=True,control_token_ids=t.encode('<｜end▁of▁sentence｜>'))
        rec=dict(name=name,status='RUNNING');result['cases'].append(rec);save()
        with mx.stream(generation_stream):
            cache=lm.make_cache()
            logits=lm(mx.array([ids[:-1]]),cache=cache);mx.eval(logits)
            rings,offset=lm.mtp_take_committed_context(cache)
            context=DSparkCommittedContext.from_native(rings,frontier=len(ids)-1,target_layer_ids=lm._config.dspark_target_layer_ids)
            cfg=OMLXDecodeConfig(omlx_path=Path('/tmp/ds41f-m33-omlx'),preserve_mtp=True,speculation_enabled=True,stop_token_ids=tuple(guard.control_token_ids))
            session=OMLXMTPGenerationSession(model,cache,np.asarray(ids[:-1]),context,config=cfg,sampler=make_sampler(temp=1.,top_p=1.,top_k=0),max_tokens=128,semantic_guard=guard)
            session.start(ids[-1])
            if forced_main is not None:
                session._bg._generation_batch._next_tokens=mx.array([forced_main],dtype=mx.uint32)
                rec['scope']='synthetic pending-token fixture through real model caches/DSpark/native init; not a sampler-distribution test'
            if name.startswith('after_') and name in ['after_safe_main_forward','after_successor_sample','after_init_construction']:
                before_parser=processor.semantic_snapshot();before_counts=dict(counts)
                fault=name
                try:session.next_token();raise AssertionError('fault injection did not fire')
                except TimeoutError:pass
                fault=None
                assert session._operation_failed and processor.semantic_snapshot()==before_parser
                for method in [session.next_token,session.quiesce]:
                    try:method();raise AssertionError('failed session resumed')
                    except MTPLifecycleError:pass
                rec.update(status='PASS_FAIL_CLOSED',parser_unchanged=True,externally_resumable=False,
                           counts_delta={k:counts[k]-before_counts[k] for k in counts}, target=[c.size() for c in session._bg._generation_batch.prompt_cache])
            else:
                if name!='before_init':
                    for step in range(128):
                        token=session.next_token(transport_delivered=name!='canonical_terminal_transport_pending');assert token is not None
                        if token not in guard.control_token_ids:reference.push(d.InferenceChunk.token(token))
                        if guard.finished and not reference.finished:reference.push(d.InferenceChunk.finish(d.InferenceFinishReason.Stop))
                        assert processor.semantic_snapshot()==reference.semantic_snapshot()
                        gb=session._bg._generation_batch;h=getattr(gb,'_omlx_semantic_horizon',None)
                        state=getattr(gb,'_omlx_mtp_state',None)
                        history=session.history.canonical_frontier
                        text=decode.decode(session.history.canonical_generated_tokens,skip_special_tokens=False)
                        if name.startswith('alignment_') and step==0:
                            before_align=dict(counts);before_parser=processor.semantic_snapshot()
                            mtp._materialize_mtp_boundary_emit(gb,state)
                            assert processor.semantic_snapshot()==before_parser
                            assert h.pending is not None and h.pending.region=='alignment-successor'
                            assert counts['proposal']==before_align['proposal']
                            rec['alignment']=dict(prediction=dataclasses.asdict(h.pending),counter_delta={k:counts[k]-before_align[k] for k in counts},parser_unchanged=True)
                            if name=='alignment_prediction_interruption':break
                        if name=='alignment_successor_terminal' and guard.finished:break
                        if name in ['after_first_safe_emit','predicted_successor_before_emit','after_terminal_emit','canonical_terminal_transport_pending'] or (name.startswith('native_fixture_') and guard.finished):break
                        if name=='safe_undelivered_prefix' and state is not None and gb.prompt_cache[0].size()>history:break
                        if name=='inside_DSML' and '<｜DSML｜ calls>' in text and '</｜DSML｜ calls>' not in text:break
                        if name=='before_DSML_closure' and h is not None and h.pending is not None and not h.completed:break
                    else:raise AssertionError('interruption point not reached')
                gb=session._bg._generation_batch;h=getattr(gb,'_omlx_semantic_horizon',None)
                rec['before']=dict(history=session.history.to_json(),parser=processor.semantic_snapshot(),
                                   prediction=None if h is None or h.pending is None else dataclasses.asdict(h.pending))
                before_counts=dict(counts);before_spans=len(spans)
                quiet=session.quiesce();mx.synchronize(generation_stream)
                for token in quiet.queue_drained_tokens:reference.push(d.InferenceChunk.token(token))
                assert processor.semantic_snapshot()==reference.semantic_snapshot()
                delta={k:counts[k]-before_counts[k] for k in counts}
                assert delta['verify']==delta['proposal']==delta['reconcile']==0
                assert all(c.size()==session.history.canonical_frontier for c in quiet.target_cache)
                assert quiet.dspark_context.offsets==(session.history.canonical_frontier,)*3
                if name=='predicted_successor_before_emit':
                    assert guard.processor.semantic_terminal is None and not guard.finished
                    assert quiet.discarded_future_token==rec['before']['prediction']['prediction']['token_id']
                if name in ['before_DSML_closure','alignment_prediction_interruption']:
                    assert guard.processor.semantic_terminal is None
                    assert len(quiet.canonical_tokens)-len(ids) <= rec['before']['prediction']['ordinal']
                if rec['before']['prediction'] is not None:
                    assert len(quiet.canonical_tokens)-len(ids) <= rec['before']['prediction']['ordinal']
                    assert processor.semantic_terminal is None
                if name=='canonical_terminal_transport_pending':
                    assert guard.finished and len(quiet.recovery_suffix_tokens)==1
                    ack_before=dict(counts);snapshot=processor.semantic_snapshot()
                    session.confirm_delivery(quiet.recovery_suffix_tokens,start_ordinal=0)
                    assert processor.semantic_snapshot()==snapshot and counts==ack_before
                    rec['delivery_ack']=dict(metadata_only=True,history=session.history.to_json(),counter_delta={k:counts[k]-ack_before[k] for k in counts})
                rec.update(status='PASS',quiescence=quiet.to_json(),counts_delta=delta,forward_spans=spans[before_spans:],parser_after=processor.semantic_snapshot(),parser_matches_emitted_only=True,prediction_matches=list(guard.matches),preview_rows=list(guard.preview_rows))
            session.close();session=None
        processor.close();reference.close();save()
    result['status']='PASS';save()
finally:
    fault=None
    if session is not None:
        with mx.stream(generation_stream):session.close()
    save()
print(json.dumps(dict(status=result['status'],cases=[dict(name=c['name'],status=c['status']) for c in result['cases']]),indent=2))
