"""Real recipe/MLX bindings, synthetic target fixture for native code seams.

This suite tests actual init/chain Python functions, not model acceptance math.
Fresh-model qualification is recorded separately by run_m33_live_semantics.py.
"""
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS
import os
import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[1]),os.environ.get('DS41F_OMLX_CANDIDATE','/tmp/ds41f-m33-omlx')]
import pytest
import mlx.core as mx
import deepseek_recipe as d
from omlx.patches.mlx_lm_mtp import batch_generator as mtp
from omlx.patches.mlx_lm_mtp.semantic_horizon import SemanticGuardError, SemanticHorizon, Prediction
from ds41f_mlx.runtime.recipe_semantic_guard import RecipeSemanticGuard
from ds41f_mlx.runtime.mtp_lifecycle import CanonicalTransportHistory, canonical_quiesce_native_singleton
TOKENIZER='/tmp/ds41f-m32-recipe/static/tokenizers/v41/tokenizer.json'

class Cache:
    def __init__(self,n):self.n=n
    def size(self):return self.n

class Model:
    def __init__(self,n):
        self.rings=[NS(offset=n) for _ in range(3)]
        self._config=NS(dspark_target_layer_ids=[7,17,27],dspark_block_size=5)
        self.forwards=self.proposals=self.samples=0
    def mtp_validate_committed_context(self,cache):
        assert all(c.offset==cache[0].size() for c in self.rings)
    def mtp_take_committed_context(self,cache):return self.rings,cache[0].size()
    def dspark_append_context(self,hidden,rings,**kw):
        for c in rings:c.offset+=int(hidden.shape[1])
    def __call__(self,ids,cache=None,**kw):
        self.forwards+=1
        for c in cache:c.n+=int(ids.shape[1])
        return mx.zeros((1,ids.shape[1],8)),mx.zeros((1,ids.shape[1],4))

class Batch:
    Response=NS
    def __init__(self,model):
        self.model=model;self.uids=[1];self.tokens=[[42]*10];self._num_tokens=[0];self.max_tokens=[100]
        self.prompt_cache=[Cache(10) for _ in range(40)]
        self._matchers=[NS(advance=lambda token:False)]
        self._next_logprobs=[mx.zeros(131072)]
    def extract_cache(self,i):return self.prompt_cache
    def filter(self,uids):self.uids=uids

@pytest.fixture
def rig(monkeypatch):
    t=d.Tokenizer.from_file(TOKENIZER)
    model=Model(10);gb=Batch(model)
    monkeypatch.setattr(mtp,'_proc_list',lambda gb:None)
    monkeypatch.setattr(mtp,'_resolve_sampler',lambda gb:lambda logits:mx.array([gb.sampled_successor],dtype=mx.uint32))
    monkeypatch.setattr(mtp,'_resolve_mtp_chain_depth',lambda model:(True,5,False))
    monkeypatch.setattr(mtp,'_mtp_depth_fixed',lambda model:True)
    monkeypatch.setattr(mtp,'_drafter_for',lambda model:None)
    monkeypatch.setattr(mtp,'_dspark_host',lambda model:model)
    monkeypatch.setattr(mtp,'_clear_rollback',lambda cache:None)
    monkeypatch.setattr(mtp,'_is_greedy',lambda gb:True)
    monkeypatch.setattr(mtp,'_drafts_before_commit',lambda model:False)
    monkeypatch.setattr(mtp,'_log_mtp_stats',lambda *args:None)
    def backbone(model,inputs,cache,**kw):
        model.forwards+=1
        for c in cache:c.n+=int(inputs.shape[1])
        return mx.zeros((1,inputs.shape[1],131072)),mx.zeros((1,inputs.shape[1],4)),None,None
    monkeypatch.setattr(mtp,'_call_backbone_captured',backbone)
    monkeypatch.setattr(mtp._prompt_priming,'take_primed',lambda *args,**kw:(model.rings,10))
    def drafts(gb,state,hidden,committed,buf):
        model.proposals+=1
        model.dspark_append_context(hidden,state.mtp_cache)
        state.hist_offset+=int(hidden.shape[1])
        state.drafts=mx.array([2,2],dtype=mx.uint32)
    monkeypatch.setattr(mtp,'_chain_next_drafts',drafts)
    def rollback(model,cache,m,k,states):
        for c in cache:c.n=10+m+1
        return True
    monkeypatch.setattr(mtp,'_chain_rollback',rollback)
    return t,model,gb

def attach(t,gb,stops=(),prefix=()):
    opts=d.ParsingOptions();opts.stop_sequences=list(stops)
    p=d.StreamProcessor(d.ChatCompletionChunkGenerator('unit','v41',True,False),opts,t)
    for token in prefix:p.push(d.InferenceChunk.token(token))
    guard=RecipeSemanticGuard(p,diagnostic=True)
    gb._omlx_semantic_guard=guard
    return guard

def emit(gb,state):
    token,lp,source=state.queue.popleft()
    return mtp._emit_response(gb,token,lp,state.stats)[0]

@pytest.mark.parametrize('which',['main','successor','none','pending_unicode','pending_unicode_successor','stop_prefix'])
def test_init_and_terminal_quiescence(rig,which):
    t,model,gb=rig
    hello=t.encode('Hello')[0];safe=t.encode('Hi')[0]
    prefix=[];stops=['Hello'];main=hello if which=='main' else safe;successor=hello
    if which=='none':stops=[]
    if which=='pending_unicode':
        ids=t.encode('🙂');prefix=ids[:-1];main=ids[-1];stops=['🙂']
    if which=='pending_unicode_successor':
        ids=t.encode('🙂');assert len(ids)==2
        main,successor=ids;stops=['🙂']
    if which=='stop_prefix':
        prefix=t.encode('HALT');main=t.encode(' NOW')[0];stops=['HALT NOW']
    guard=attach(t,gb,stops,prefix)
    gb._next_tokens=mx.array([main],dtype=mx.uint32);gb.sampled_successor=successor
    before=guard.processor.semantic_snapshot()
    mtp._post_init_mtp(gb)
    state=gb._omlx_mtp_state
    assert guard.processor.semantic_snapshot()==before
    terminal=which!='none'
    terminal_main=which in ['main','pending_unicode','stop_prefix']
    assert model.forwards==(0 if terminal_main else 1)
    assert model.proposals==(0 if terminal else 1)
    assert len(state.queue)==(1 if terminal_main else 2)
    assert gb.prompt_cache[0].size()==(10 if terminal_main else 11)
    assert all(c.offset==gb.prompt_cache[0].size() for c in state.mtp_cache)
    if terminal:
        history=CanonicalTransportHistory(tuple([42]*10))
        while state.queue:
            response=emit(gb,state);history.record_delivered(response.token)
        assert response.finish_reason=='stop'
        h=gb._omlx_semantic_horizon
        assert h.completed and len(guard.matches)==1
        before_work=(model.forwards,model.proposals)
        result=canonical_quiesce_native_singleton(language_model=model,target_cache=gb.prompt_cache,mtp_state=state,history=history,mx=mx,queue_pop=lambda q:q.popleft())
        assert result.counters.target_forwards==1
        assert model.forwards==before_work[0]+1 and model.proposals==before_work[1]
        assert all(c.size()==history.canonical_frontier for c in gb.prompt_cache)
    guard.processor.close()

@pytest.mark.parametrize('guarded',[False,True])
def test_no_boundary_init_call_shape_parity(rig,guarded):
    t,model,gb=rig
    if guarded:guard=attach(t,gb)
    gb._next_tokens=mx.array([t.encode('Hi')[0]],dtype=mx.uint32)
    gb.sampled_successor=t.encode(' there')[0]
    mtp._post_init_mtp(gb)
    state=gb._omlx_mtp_state
    assert [x[0] for x in state.queue]==[int(gb._next_tokens[0]),gb.sampled_successor]
    assert (model.forwards,model.proposals,state.hist_offset)==(1,1,11)
    assert [c.offset for c in state.mtp_cache]==[11]*3
    if guarded:guard.processor.close()

@pytest.mark.parametrize('topology',['full_inside','bonus','correction','reject_before','immediate_terminal'])
def test_chain_narrows_only_actual_candidate(rig,topology):
    t,model,gb=rig
    safe=t.encode('Hi')[0];hello=t.encode('Hello')[0];tail=t.encode(' tail')[0]
    guard=attach(t,gb,['Hello'])
    guard.observe_canonical_emit(safe,None)
    gb.tokens[0].append(safe);gb._num_tokens[0]=1
    state=mtp._MtpState(uid=1);state.chain=True;state.depth=2;state.mtp_cache=model.rings;state.hist_offset=10
    state.next_main=mx.array([safe],dtype=mx.uint32)
    m=2;drafts=[safe,hello];bonus=tail
    if topology=='bonus':drafts=[safe,safe];bonus=hello
    if topology=='correction':m=1;drafts=[safe,tail];bonus=hello
    if topology=='reject_before':m=0;drafts=[hello,tail];bonus=safe
    if topology=='immediate_terminal':m=0;drafts=[tail,tail];bonus=hello
    state.drafts=mx.array(drafts,dtype=mx.uint32)
    targets=[bonus]*3
    targets[m]=bonus
    for c in gb.prompt_cache:c.n=13
    gb._omlx_mtp_state=state
    before=guard.processor.semantic_snapshot()
    mtp._run_verify_cycle_chain(gb,state,verify_result=(mx.zeros((1,3,131072)),mx.zeros((1,3,4)),None),greedy_result=[m]+targets+drafts)
    h=gb._omlx_semantic_horizon
    assert guard.processor.semantic_snapshot()==before
    if topology=='reject_before':
        assert h.pending is None and model.proposals==1
        assert [x[0] for x in state.queue]==[safe]
    else:
        assert h.pending is not None and model.proposals==0
        assert state.queue[-1][0]==hello
        assert gb.prompt_cache[0].size()==10+len(state.queue)
        # Includes the already-canonical anchor at target position 10; terminal
        # is the next position after the committed safe draft prefix.
        assert h.pending.ordinal==len(state.queue)
        while state.queue:response=emit(gb,state)
        assert response.finish_reason=='stop' and len(guard.matches)==1
    guard.processor.close()


def test_outstanding_safe_queue_is_seen_only_by_fork(rig):
    t,model,gb=rig
    guard=attach(t,gb,['HALT NOW'])
    h=SemanticHorizon(guard)
    prefix=t.encode('HALT')
    candidate=t.encode(' NOW')
    before=guard.processor.semantic_snapshot()
    terminal=h.preview(gb,candidate,region='chain',queue=[(i,None,'draft') for i in prefix])
    assert terminal.queued_prefix==len(prefix)
    assert terminal.prediction.index==len(prefix)+len(candidate)-1
    assert guard.processor.semantic_snapshot()==before
    guard.processor.close()


def test_repeated_ids_are_owned_by_ordinal(rig):
    t,model,gb=rig
    hello=t.encode('Hello')[0]
    guard=attach(t,gb,['HelloHello'])
    h=SemanticHorizon(guard)
    terminal=h.preview(gb,[hello,hello],region='init-successor')
    h.bind(terminal)
    assert h.observe(gb,hello) is None
    gb._num_tokens[0]+=1
    assert h.observe(gb,hello)=='stop'
    guard.processor.close()


def test_disagreement_and_inexact_mapping_fail_closed(rig):
    t,model,gb=rig
    hello=t.encode('Hello')[0];safe=t.encode('Hi')[0]
    guard=attach(t,gb,['Hello'])
    h=SemanticHorizon(guard);h.bind(h.preview(gb,[hello],region='init-main'))
    with pytest.raises(SemanticGuardError):h.observe(gb,safe)
    assert guard.processor.semantic_terminal is None
    with pytest.raises(SemanticGuardError):h.preview(gb,[hello],region='chain')
    p=d.StreamProcessor(d.ChatCompletionChunkGenerator('x','v41',True,False),d.ParsingOptions())
    no_tokenizer=RecipeSemanticGuard(p)
    with pytest.raises(RuntimeError):no_tokenizer.preview([1])
    p.close();guard.processor.close()
