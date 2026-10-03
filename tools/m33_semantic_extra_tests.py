"""Additional horizon, alignment and serialized interruption regressions."""
import ast
from dataclasses import replace
import inspect
import textwrap
from threading import Event,RLock,Thread
from tools.m33_semantic_tests import rig,attach,emit,mtp,mx,d
from omlx.patches.mlx_lm_mtp.semantic_horizon import horizon,Prediction,SemanticGuardError
from ds41f_mlx.runtime.mtp_lifecycle import _serialized_mtp_operation,OMLXMTPGenerationSession,MTPLifecycleError
from ds41f_mlx.runtime.recipe_semantic_guard import RecipeSemanticGuard
import pytest


def chain_fixture(t,model,gb,drafts,bonus,*,align=0,clamp=None):
    safe=t.encode('Hi')[0]
    guard=attach(t,gb,['Hello']);guard.observe_canonical_emit(safe,None)
    gb.tokens[0].append(safe);gb._num_tokens[0]=1
    model._omlx_mtp_commit_align=align
    if clamp is not None:model.mtp_clamp_accept=clamp
    state=mtp._MtpState(uid=1);state.chain=True;state.depth=len(drafts);state.mtp_cache=model.rings;state.hist_offset=10
    state.next_main=mx.array([safe],dtype=mx.uint32);state.drafts=mx.array(drafts,dtype=mx.uint32)
    gb._omlx_mtp_state=state
    k=len(drafts)
    for c in gb.prompt_cache:c.n=10+k+1
    mtp._run_verify_cycle_chain(gb,state,verify_result=(mx.zeros((1,k+1,131072)),mx.zeros((1,k+1,4)),None),greedy_result=[k]+[bonus]*(k+1)+drafts)
    return guard,state


def test_alignment_never_materializes_predicted_terminal(rig):
    t,model,gb=rig
    guard,state=chain_fixture(t,model,gb,[t.encode('Hi')[0],t.encode('Hello')[0]],t.encode(' tail')[0],align=13)
    assert model.forwards==model.proposals==0
    assert not state.boundary_emit_pending
    assert horizon(gb).pending is not None
    with pytest.raises(SemanticGuardError):mtp._materialize_mtp_boundary_emit(gb,state)
    assert model.forwards==0
    guard.processor.close()


def test_shorter_model_clamp_repreviews_without_stale_terminal(rig):
    t,model,gb=rig
    calls=[]
    def clamp(cache,m,k):calls.append(m);return 0
    guard,state=chain_fixture(t,model,gb,[t.encode('Hi')[0],t.encode('Hello')[0]],t.encode(' tail')[0],clamp=clamp)
    assert calls==[1] and horizon(gb).pending is None
    assert len(state.queue)==1 and state.queue[0][0]==t.encode('Hi')[0]
    assert model.proposals==1 and guard.processor.semantic_terminal is None
    guard.processor.close()


def test_alignment_successor_previews_safe_queue_and_binds_exact_ordinal(rig):
    t,model,gb=rig
    safe,hello=t.encode('Hi')[0],t.encode('Hello')[0]
    guard=attach(t,gb,['Hello'])
    state=mtp._MtpState(uid=1);state.chain=True;state.mtp_cache=model.rings;state.hist_offset=10
    state.queue.append((safe,mx.zeros(131072),'bonus'));gb._omlx_mtp_state=state;gb.sampled_successor=hello
    before=guard.processor.semantic_snapshot()
    mtp._materialize_mtp_boundary_emit(gb,state)
    h=horizon(gb)
    assert guard.processor.semantic_snapshot()==before
    assert h.pending.ordinal==1 and h.pending.queued_prefix==1
    assert (model.forwards,model.proposals)==(1,0)
    assert [c.offset for c in state.mtp_cache]==[11]*3
    assert emit(gb,state).finish_reason is None
    assert emit(gb,state).finish_reason=='stop'
    guard.processor.close()


def test_terminal_in_committed_queue_fails_before_target_forward(rig):
    t,model,gb=rig
    guard=attach(t,gb,['Hello'])
    state=mtp._MtpState(uid=1);state.queue.append((t.encode('Hello')[0],None,'bonus'))
    with pytest.raises(SemanticGuardError):mtp._materialize_mtp_boundary_emit(gb,state)
    assert model.forwards==0 and guard.processor.semantic_terminal is None
    guard.processor.close()


def test_zero_depth_candidate_terminal_uses_append_not_proposal(rig):
    t,model,gb=rig
    hello=t.encode('Hello')[0]
    gb.sampled_successor=hello
    guard,state=chain_fixture(t,model,gb,[],hello)
    assert horizon(gb).pending is not None and model.proposals==0
    assert state.stats.zero_cycles==1 and state.queue[-1][0]==hello
    guard.processor.close()


def test_wrong_prediction_span_fails_canonical_confirmation(rig):
    t,model,gb=rig
    hello=t.encode('Hello')[0];guard=attach(t,gb,['Hello']);h=horizon(gb)
    owned=h.preview(gb,[hello],region='init-main')
    h.bind(replace(owned,prediction=Prediction('STOP_SEQUENCE',0,0,hello,('STOP_SEQUENCE',99,100))))
    with pytest.raises(SemanticGuardError):h.observe(gb,hello)
    assert guard.processor.semantic_terminal is not None
    assert gb.tokens[0]==[42]*10
    with pytest.raises(SemanticGuardError):h.preview(gb,[hello],region='chain')
    guard.processor.close()


def test_backend_control_eos_is_not_arbitrary_stop_id_matching(rig):
    t,model,gb=rig
    eos=t.encode('<｜end▁of▁sentence｜>')[0]
    opts=d.ParsingOptions()
    p=d.StreamProcessor(d.ChatCompletionChunkGenerator('x','v41',True,False),opts,t)
    guard=RecipeSemanticGuard(p,control_token_ids=[eos]);before=p.semantic_snapshot()
    assert guard.observe_canonical_emit(eos,None) is None
    assert p.semantic_snapshot()==before
    guard.finish_backend();assert p.finished and p.semantic_terminal is None
    p.close()
    opts.stop_sequences=['<｜end▁of▁sentence｜>']
    p=d.StreamProcessor(d.ChatCompletionChunkGenerator('x','v41',True,False),opts,t)
    guard=RecipeSemanticGuard(p,control_token_ids=[eos])
    assert guard.preview([eos]) is None
    with pytest.raises(SemanticGuardError):guard.observe_canonical_emit(eos,('STOP_SEQUENCE',0,27))
    assert p.semantic_terminal is None
    p.close()
    # Direct recipe callers may intentionally supply the literal, but the
    # official backend's suppressed control policy must never do so.
    p=d.StreamProcessor(d.ChatCompletionChunkGenerator('x','v41',True,False),opts,t)
    guard=RecipeSemanticGuard(p)
    prediction=guard.preview([eos]);assert prediction is not None
    assert guard.observe_canonical_emit(eos,prediction.identity)=='stop'
    assert p.semantic_terminal.kind=='STOP_SEQUENCE'
    p.close()


def test_session_serialization_and_internal_fault_are_fail_closed():
    entered,release,cancelled=Event(),Event(),Event()
    class Session:
        _operation_lock=RLock();_operation_failed=False
        @_serialized_mtp_operation
        def next_token(self):entered.set();assert release.wait(5)
        @_serialized_mtp_operation
        def quiesce(self):cancelled.set()
        @_serialized_mtp_operation
        def fault(self):raise TimeoutError('internal init')
    s=Session();worker=Thread(target=s.next_token);worker.start();assert entered.wait(5)
    cancel=Thread(target=s.quiesce);cancel.start();assert not cancelled.wait(.05)
    release.set();worker.join(5);cancel.join(5);assert cancelled.is_set()
    with pytest.raises(TimeoutError):s.fault()
    with pytest.raises(MTPLifecycleError):s.quiesce()
    # Actual session methods use this exact decorator. Init/chain helpers have
    # no await or response yield that could expose a halfway state to a caller.
    for name in ['start','next_token','quiesce','close']:
        assert hasattr(getattr(OMLXMTPGenerationSession,name),'__wrapped__')
    for method in [mtp._post_init_mtp,mtp._run_verify_cycle_chain]:
        tree=ast.parse(textwrap.dedent(inspect.getsource(method)))
        assert not any(isinstance(n,(ast.Await,ast.Yield,ast.YieldFrom)) for n in ast.walk(tree))


def test_missing_priming_and_foreign_uid_fail_before_init_forward(rig):
    t,model,gb=rig
    guard=attach(t,gb,['Hello']);hello=t.encode('Hello')[0]
    gb._next_tokens=mx.array([hello],dtype=mx.uint32);gb.sampled_successor=hello
    before=guard.processor.semantic_snapshot()
    valid=model.mtp_validate_committed_context
    def missing(cache):raise RuntimeError('committed context unavailable')
    model.mtp_validate_committed_context=missing
    with pytest.raises(RuntimeError):mtp._post_init_mtp(gb)
    assert model.forwards==model.proposals==0
    model.mtp_validate_committed_context=valid
    h=horizon(gb);h.bind(h.preview(gb,[hello],region='init-main'))
    gb.uids=[2]
    with pytest.raises(SemanticGuardError):h.observe(gb,hello)
    assert guard.processor.semantic_snapshot()==before
    guard.processor.close()


def test_dsml_semantic_finish_beats_simultaneous_length(rig):
    t,model,gb=rig
    call=('<｜DSML｜ calls>\n<｜DSML｜ invoke name="lookup">\n'
          '<｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>\n'
          '</｜DSML｜ invoke>\n</｜DSML｜ calls>')
    ids=t.encode(call)
    p=d.StreamProcessor(d.ChatCompletionChunkGenerator('length','v41',True,False),d.ParsingOptions(),t)
    events=[]
    for token in ids[:-1]:events.extend(p.push(d.InferenceChunk.token(token)))
    assert p.semantic_terminal is None
    guard=RecipeSemanticGuard(p);gb._omlx_semantic_guard=guard
    gb._next_tokens=mx.array([ids[-1]],dtype=mx.uint32);gb.max_tokens=[1]
    mtp._post_init_mtp(gb)
    response=emit(gb,gb._omlx_mtp_state)
    assert response.finish_reason=='stop' and model.forwards==model.proposals==0
    body=d.ChatCompletionResponse('length','v41',0,0,0)
    for event in events+guard.events:body.append(event)
    import json
    value=json.loads(body.to_json())
    assert value['choices'][0]['finish_reason']=='tool_calls'
    p.close()


def test_old_engine_cannot_silently_ignore_attached_guard(rig,monkeypatch):
    from types import SimpleNamespace
    import ds41f_mlx.runtime.mtp_lifecycle as life
    from ds41f_mlx.runtime.omlx_generation import OMLXDecodeConfig
    t,model,gb=rig
    original=life.importlib.import_module
    def modules(name,package=None):
        if name=='omlx.patches.mlx_lm_mtp.batch_generator':return SimpleNamespace(apply=lambda:True)
        return original(name,package)
    monkeypatch.setattr(life.importlib,'import_module',modules)
    with pytest.raises(MTPLifecycleError) as error:
        OMLXMTPGenerationSession(model,gb.prompt_cache,[42]*10,None,config=OMLXDecodeConfig(speculation_enabled=True,preserve_mtp=True),semantic_guard=object())
    assert 'semantic-horizon candidate engine' in str(error.value.__cause__)
    assert model.forwards==model.proposals==0


def test_delivery_ack_is_prefix_ordinal_owned_and_metadata_only():
    from ds41f_mlx.runtime.mtp_lifecycle import CanonicalTransportHistory
    s=object.__new__(OMLXMTPGenerationSession)
    s._operation_lock=RLock();s._operation_failed=False
    s.history=CanonicalTransportHistory((1,2));s.history.commit_undelivered([3,3])
    s.confirm_delivery([3],start_ordinal=0)
    assert s.history.canonical_frontier==4 and s.history.delivered_frontier==3
    s.confirm_delivery([3],start_ordinal=1)
    assert s.history.recovery_suffix_tokens==[]
    with pytest.raises(MTPLifecycleError):s.confirm_delivery([3],start_ordinal=1)
    assert s._operation_failed


def test_suppressed_eos_spelling_never_becomes_user_stop_or_alignment_forward(rig):
    from types import SimpleNamespace
    t,model,gb=rig
    eos=t.encode('<｜end▁of▁sentence｜>')[0];safe=t.encode('Hi')[0]
    guard=attach(t,gb,['sentence']);guard.control_token_ids=frozenset([eos])
    assert guard.processor.preview_tokens([eos]).terminal_kind=='STOP_SEQUENCE'
    assert guard.preview([eos]) is None
    with pytest.raises(SemanticGuardError):guard.preview([eos,safe])
    guard.observe_canonical_emit(safe,None);gb.tokens[0].append(safe);gb._num_tokens=[1]
    model._omlx_mtp_commit_align=13
    gb._matchers=[SimpleNamespace(advance=lambda token:token==eos)]
    state=mtp._MtpState(uid=1);state.chain=True;state.depth=1;state.mtp_cache=model.rings;state.hist_offset=10
    state.next_main=mx.array([safe],dtype=mx.uint32);state.drafts=mx.array([safe],dtype=mx.uint32);gb._omlx_mtp_state=state
    for cache in gb.prompt_cache:cache.n=12
    mtp._run_verify_cycle_chain(gb,state,verify_result=(mx.zeros((1,2,131072)),mx.zeros((1,2,4)),None),greedy_result=[1,eos,eos,safe])
    assert not state.boundary_emit_pending and model.forwards==0
    assert horizon(gb).pending is None
    assert emit(gb,state).finish_reason is None
    before=guard.processor.semantic_snapshot()
    assert emit(gb,state).finish_reason=='stop'
    assert guard.processor.semantic_snapshot()==before
    guard.finish_backend();guard.processor.close()
