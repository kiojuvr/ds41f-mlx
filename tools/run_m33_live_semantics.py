"""Fresh native singleton model semantics; isolated candidate, no serving/soak."""
import argparse
from collections import deque
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--omlx', type=Path, default=Path('/tmp/ds41f-m33-omlx'))
p.add_argument('--recipe', type=Path, default=Path('/tmp/ds41f-m32-recipe'))
p.add_argument('--checkpoint', type=Path, default=Path('/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash'))
p.add_argument('--cases', default='main_stop,successor_stop,text,one_tool,thinking_tool,multiple_tools')
p.add_argument('--out', type=Path, default=ROOT/'artifacts/m33/live.json')
p.add_argument('--diagnostic', action=argparse.BooleanOptionalAction, default=True)
a = p.parse_args()
sys.path[:0] = [str(ROOT), str(a.omlx)]
import deepseek_recipe as d
import deepseek_recipe._native as native
import mlx.core as mx
import omlx.scheduler
import omlx.api.tool_calling
from mlx_lm.generate import BatchGenerator, generation_stream
from mlx_lm.sample_utils import make_sampler
from omlx.patches.deepseek_v41.loading import load
from omlx.patches.mlx_lm_mtp import batch_generator as mtp, cache_rollback, prompt_priming
from ds41f_mlx.runtime.recipe_semantic_guard import RecipeSemanticGuard
from ds41f_mlx.runtime.mtp_lifecycle import canonical_quiesce_native_singleton, CanonicalTransportHistory
assert mtp.apply() and cache_rollback.apply()
t = d.Tokenizer.from_file(str(a.recipe/'static/tokenizers/v41/tokenizer.json'))
result = dict(scope='Fresh actual model native semantics; no public promotion or operational soak',
              native_path=native.__file__,native_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
              omlx_revision=subprocess.check_output(['git','-C',str(a.omlx),'rev-parse','HEAD'],text=True).strip(),
              cases=[],status='RUNNING')
def save():
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
counts = dict(verify=0,proposal=0,target_forward=0,reconcile=0,repack=0)
original_cycle = mtp._run_verify_cycle_chain
original_reconcile = mtp._reconcile_mtp_to_standard
cycles = []
def forbid_reconcile(*args,**kwargs):
    counts['reconcile'] += 1
    raise AssertionError('history reconstruction forbidden')
mtp._reconcile_mtp_to_standard = forbid_reconcile

def cycle(gb,state,*args,**kwargs):
    counts['verify'] += 1
    start = state.stats.accepts
    actual_depth = int(state.drafts.shape[0])
    guard=getattr(gb,'_omlx_semantic_guard',None)
    before_parser=None if guard is None else guard.processor.semantic_snapshot()
    ret = original_cycle(gb,state,*args,**kwargs)
    h = getattr(gb,'_omlx_semantic_horizon',None)
    if guard is not None:
        assert guard.processor.semantic_snapshot()==before_parser, 'verify cycle mutated canonical parser'
    cycles.append(dict(depth=actual_depth,accepted=state.stats.accepts-start,
                       target=[int(c.size()) for c in gb.prompt_cache],dspark=[int(c.offset) for c in state.mtp_cache],
                       history=len(gb.tokens[0]),queue=[[int(token),source] for token,lp,source in state.queue],
                       terminal=None if h is None or h.pending is None else dataclasses.asdict(h.pending)))
    return ret
mtp._run_verify_cycle_chain = cycle
model = None
try:
    model,_ = load(a.checkpoint,preserve_mtp=True,engram_ssd_offload=True)
    lm = model.language_model
    old_forward,old_proposal = lm._forward,lm.dspark_forward
    forward_spans = []
    def forward(ids,cache=None,**kw):
        counts['target_forward'] += 1
        forward_spans.append(dict(start=0 if not cache else int(cache[0].size()), ids=ids.tolist()[0]))
        return old_forward(ids,cache=cache,**kw)
    def proposal(*args,**kwargs):
        counts['proposal'] += 1
        return old_proposal(*args,**kwargs)
    lm._forward,lm.dspark_forward = forward,proposal
    tool = {'type':'function','function':{'name':'lookup','description':'Return weather for a city.',
            'parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}
    for name in a.cases.split(','):
        for on in ([True] if name.endswith('_stop') else [False,True]):
            prompt_priming.drop_ctx(lm)
            lm.configure_mtp(on,5)
            mx.random.seed(3201)
            is_tool = 'tool' in name
            content = 'Say hello in one short sentence.'
            if name == 'unicode_stop': content = 'Reply exactly with: café🙂 HALT NOW. Do not add anything else.'
            if name == 'cycle_stop': content = 'Reply exactly with: alpha beta gamma delta HALT NOW. Do not add anything else.'
            if is_tool:
                content = ('Call lookup for Paris and London, both in this response. Do not answer without both calls.' if name=='multiple_tools' else 'Use lookup to get the weather in Paris. You must call lookup; do not invent weather.')
            req = {'model':'v41','messages':[{'role':'user','content':content}],
                   'stop':None}
            if is_tool:
                req['tools'] = [tool]
            thinking = name=='thinking_tool'
            converted = d.ChatCompletionRequest(req).convert(d.ConversionOptions(default_thinking_mode=thinking))
            opts = converted.parsing_options
            if name=='main_stop': opts.stop_sequences=['Hello']
            if name=='text_aligned_stop': opts.stop_sequences=['!']
            if name=='successor_stop': opts.stop_sequences=[' there']
            if name=='multi_stop': opts.stop_sequences=['Hello there']
            if name=='shared_stop': opts.stop_sequences=['Hello there','Hello then']
            if name=='unicode_stop': opts.stop_sequences=['🙂']
            if name=='cycle_stop': opts.stop_sequences=['HALT NOW','HALT NEVER']
            ids = d.DeepseekV41Encoding().with_tokenizer(t).encode(converted.conversation)
            lm._omlx_mtp_commit_align = len(ids)+3 if name=='text_aligned_stop' else 0
            processor = d.StreamProcessor(d.ChatCompletionRequest.chunk_generator(converted,'m33','v41'),opts,t)
            reference = d.StreamProcessor(d.ChatCompletionRequest.chunk_generator(converted,'m33','v41'),opts,t)
            control_ids=t.encode('<｜end▁of▁sentence｜>')
            guard = RecipeSemanticGuard(processor,diagnostic=a.diagnostic,control_token_ids=control_ids)
            history = CanonicalTransportHistory(tuple(ids))
            rec = dict(name=name,mtp=on,prompt_ids=ids,status='RUNNING',emits=[],cycles=[],counts_before=dict(counts))
            result['cases'].append(rec);save()
            cycles.clear();forward_spans.clear()
            bg = None
            start = time.perf_counter();first_latency=None
            try:
                with mx.stream(generation_stream):
                    cache = lm.make_cache()
                    for begin in range(0,len(ids)-1,2048):
                        logits = lm(mx.array([ids[begin:min(begin+2048,len(ids)-1)]]),cache=cache)
                        mx.eval(logits)
                    bg = BatchGenerator(lm,max_tokens=256,stop_tokens=[t.encode('<｜end▁of▁sentence｜>')],sampler=make_sampler(temp=1.,top_p=1.,top_k=0),completion_batch_size=1,prefill_batch_size=1,prefill_step_size=2048,stream=generation_stream)
                    uid=bg.insert([[ids[-1]]],caches=[cache],all_tokens=[ids[:-1]])[0]
                    pr,gr=bg.next();mx.synchronize(generation_stream)
                    assert not gr and any(r.end_of_prompt for r in pr)
                    gb=bg._generation_batch
                    if on: gb._omlx_semantic_guard=guard
                    decode_start=time.perf_counter()
                    last_state=None; last_cache=None
                    for i in range(256):
                        pr,gr=bg.next();mx.synchronize(generation_stream)
                        assert len(gr)==1
                        response=gr[0];token=int(response.token)
                        if first_latency is None: first_latency=time.perf_counter()-decode_start
                        if not on and token not in control_ids:
                            guard.events.extend(processor.push(d.InferenceChunk.token(token)))
                        if token not in control_ids:
                            reference.push(d.InferenceChunk.token(token))
                        obs=processor.semantic_terminal
                        if obs is not None and not reference.finished:
                            reference.push(d.InferenceChunk.finish(d.InferenceFinishReason.Stop))
                        if not on and obs is not None:
                            guard.finish_backend()
                        elif response.finish_reason and not guard.finished:
                            guard.finish_backend(response.finish_reason)
                            if not reference.finished:
                                reference.push(d.InferenceChunk.finish(d.InferenceFinishReason.Length if response.finish_reason=='length' else d.InferenceFinishReason.Stop))
                        assert processor.semantic_snapshot()==reference.semantic_snapshot(), 'parser differs from emitted-only reference'
                        history.record_delivered(token)
                        semantic=getattr(gb,'_omlx_semantic_horizon',None)
                        state=(getattr(gb,'_omlx_mtp_state',None) or (None if semantic is None else semantic.last_state))
                        actual_cache=response.prompt_cache if response.finish_reason else gb.prompt_cache
                        rec['emits'].append(dict(token=token,backend_finish=response.finish_reason,parser=processor.semantic_snapshot(),
                                                semantic_terminal=None if obs is None else dict(kind=obs.kind,start=obs.source_start,end=obs.source_end),
                                                history=history.canonical_frontier,target=[int(c.size()) for c in actual_cache],
                                                dspark=None if state is None else [int(c.offset) for c in state.mtp_cache]))
                        if obs is not None or response.finish_reason:
                            last_state,last_cache=state,actual_cache
                            rec['backend_finish']=response.finish_reason
                            break
                    else: raise AssertionError('turn did not terminate')
                    rec['decode_seconds']=time.perf_counter()-decode_start
                    rec['first_token_seconds']=first_latency
                    rec['tok_s']=len(history.canonical_generated_tokens)/rec['decode_seconds']
                    rec['cycles']=list(cycles)
                    rec['prediction_matches']=list(guard.matches)
                    if on and name.endswith('_stop'):
                        assert len(guard.matches)==1 and guard.matches[0]['identity'][0]=='STOP_SEQUENCE', 'intended stop fixture was not induced'
                    if on:
                        assert last_state is not None
                        before_counts=dict(counts);before_forward=len(forward_spans)
                        quiet=canonical_quiesce_native_singleton(language_model=lm,target_cache=last_cache,mtp_state=last_state,
                            history=history,mx=mx,queue_pop=lambda q:q.popleft(),observe_canonical=lambda token:guard.observe_canonical_emit(token,None))
                        mx.synchronize(generation_stream)
                        rec['quiescence']=quiet.to_json()
                        rec['quiescence_actual_counter_delta']={k:counts[k]-before_counts[k] for k in counts}
                        rec['quiescence_forwards']=forward_spans[before_forward:]
                        assert counts['verify']==before_counts['verify'] and counts['proposal']==before_counts['proposal']
                        assert [int(c.size()) for c in last_cache]==[history.canonical_frontier]*40
                        assert all(c.offset==history.canonical_frontier for c in quiet.dspark_context.caches)
                    body=d.ChatCompletionResponse('m33','v41',0,0,0)
                    for event in guard.events:body.append(event)
                    rec['protocol']=json.loads(body.to_json())
                    rec['preview'] = dict(calls=guard.calls,candidate_ids=guard.candidate_ids,latencies_ns=list(guard.latencies_ns),pending_groups=list(guard.pending_sizes), rows=list(guard.preview_rows))
                    rec['forward_spans']=list(forward_spans)
                    rec['counts_delta']={k:counts[k]-rec['counts_before'][k] for k in counts}
                    rec['status']='PASS'
                    bg.remove([uid]);bg.close();bg=None
                    prompt_priming.drop_ctx(lm)
            finally:
                if bg is not None:
                    with mx.stream(generation_stream):bg.close()
                processor.close();reference.close();save()
    result['status']='PASS';save()
finally:
    save()
print(json.dumps(dict(status=result['status'],cases=[dict(name=c['name'],mtp=c['mtp'],status=c['status']) for c in result['cases']]),indent=2))
