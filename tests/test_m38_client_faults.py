"""M38 synthetic exact-boundary faults. No checkpoint/model authority claimed."""
from copy import deepcopy
import json
import pytest
from ds41f_mlx.internal_local_client import ClientStateError, InternalLocalClient
from tests.test_m37_local_client import Runtime, submit, certify_tool, outcome, tool, Conn, Response
from ds41f_mlx.internal_local_client import OwnedStream
from ds41f_mlx.web_client import RuntimeHTTPError


class ResetRuntime(Runtime):
    def __init__(self):
        super().__init__(); self.creates = []; self.deletes = []
    def create_session(self, sid):
        self.creates.append(sid); self.next=1; self.slot=None; self.out=None
        return super().create_session(sid)
    def close_session(self, sid):
        self.deletes.append(sid); return super().close_session(sid)


@pytest.mark.parametrize('action', ['delete', 'create'])
@pytest.mark.parametrize('reached', [False, True])
def test_lifecycle_ambiguity_is_sticky(action, reached):
    r=ResetRuntime(); c=InternalLocalClient(r); c.create('old'); submit(c); c.reconcile()
    if action=='create': c.retire()
    original = r.close_session if action=='delete' else r.create_session
    def loss(sid):
        if reached: original(sid)
        raise OSError('response lost; admission unknown')
    if action=='delete': r.close_session=loss
    else: r.create_session=loss
    with pytest.raises(OSError): c.retire() if action=='delete' else c.create('new')
    if action=='delete': r.close_session=original
    else: r.create_session=original
    before=(list(r.creates), list(r.deletes), list(r.posts))
    for fn in [c.retire, lambda:c.create('another'), lambda:submit(c), c.reconcile]:
        with pytest.raises(ClientStateError): fn()
    assert c.state=='stopped' and c.lifecycle_uncertain['action']==action
    assert before==(r.creates,r.deletes,r.posts)


@pytest.mark.parametrize('result', ['raise', 'oversized', 'invalid'])
def test_reserved_effect_cannot_be_bypassed_by_observation_or_retirement(result):
    r=ResetRuntime(); c=InternalLocalClient(r); c.create('old'); certify_tool(r,c); effects=[]
    def effect(call):
        effects.append(call)
        if result=='raise': raise OSError('effect may have happened')
        return 'x'*65537 if result=='oversized' else None
    with pytest.raises((OSError,ClientStateError)): c.execute_tools(effect)
    ledger=c.ledger; posts=len(r.posts)
    for _ in range(4):
        c.reconcile(); assert c.state=='tool_ambiguous'
        for fn in [lambda:c.execute_tools(effect), lambda:submit(c), c.retire,
                   lambda:c.submit_tool_results(options={}), lambda:c.create('fresh')]:
            with pytest.raises(ClientStateError): fn()
        assert c.ledger==ledger
    assert len(effects)==1 and len(r.deletes)==0 and len(r.creates)==1
    assert len(r.posts)==posts+4


def test_ambiguous_effect_direct_retire_cannot_construct_fresh_history():
    r=ResetRuntime();c=InternalLocalClient(r);c.create('s');certify_tool(r,c)
    def effect(_):raise OSError('external effect may have occurred')
    with pytest.raises(OSError):c.execute_tools(effect)
    with pytest.raises(ClientStateError,match='application intervention'):c.retire()
    assert not r.deletes and len(c.ledger)==1 and c.state=='tool_ambiguous'


def test_actual_128_entry_capacity_multi_session_no_eviction_or_alias():
    r=ResetRuntime(); c=InternalLocalClient(r); c.create('s0'); effects=[]
    saved={}; samples=[]
    for i in range(128):
        if i and i%32==0:
            c.retire(); c.create(f's{i//32}')
            assert c.next_sequence==1
        submit(c)
        # Force a real certificate-shaped completed outcome at the exact ledger boundary.
        r.out=outcome(json.loads(c.identity.body), seq=c.identity.sequence, calls=[tool()])
        c.reconcile(); prior=c.messages
        c.reconcile(); assert c.messages==prior
        stored=c.execute_tools(lambda call: effects.append(deepcopy(call)) or f'result-{i}')
        c.reconcile(); assert c.execute_tools(lambda _:pytest.fail('duplicate'))==stored
        assert len(c.ledger)==i+1 and len(effects)==i+1
        for k,v in saved.items(): assert c.ledger[k]==v
        saved=c.ledger
        c.submit_tool_results(options=dict(stream=False)); c.reconcile()
        assert c.next_sequence==c.identity.sequence+1
        assert len(c.timings)<=128
        if (i+1)%32==0:
            samples.append(dict(effects=i+1,ledger_entries=len(c.ledger),timings=len(c.timings),
                client_bytes=c.retained_state_bytes(),request_bytes=len(c.identity.body),
                outcome_bytes=len(json.dumps(c.outcome).encode()),session_id=c.session_id))
    submit(c); r.out=outcome(json.loads(c.identity.body),seq=c.identity.sequence,calls=[tool()]); c.reconcile()
    with pytest.raises(ClientStateError,match='ledger full'):
        c.execute_tools(lambda _:pytest.fail('overflow effect'))
    assert c.state=='stopped' and c.ledger==saved and len(effects)==128
    assert len(c.timings)==128
    assert len({k[0] for k in c.ledger})==4
    assert [v['result']['content'] for v in c.ledger.values()]==[f'result-{i}' for i in range(128)]
    import os
    if os.environ.get('DS41F_M38_EVIDENCE'):
        from pathlib import Path
        Path(os.environ['DS41F_M38_EVIDENCE']).write_text(json.dumps(dict(
            scope='synthetic certificate/runtime; actual InternalLocalClient',status='PASS',
            samples=samples,effects=len(effects),capacity=c.ledger_limit,state=c.state,
            no_eviction=True,overflow_effects=0,results_unchanged=True,cross_session_alias=False,
            ledger=[dict(key=list(k),**v) for k,v in c.ledger.items()]),indent=2)+'\n')


def test_maximum_stored_result_is_reused_without_second_effect():
    r=ResetRuntime();c=InternalLocalClient(r);c.create('s');certify_tool(r,c)
    value='x'*65536;stored=c.execute_tools(lambda _:value)
    c.reconcile();assert c.execute_tools(lambda _:pytest.fail('duplicate'))==stored
    assert next(iter(c.ledger.values()))['result']['content']==value


@pytest.mark.parametrize('fault',['body_limit','owned_messages','not_serializable'])
def test_local_request_faults_do_not_mutate_server(fault):
    r=ResetRuntime();c=InternalLocalClient(r);c.create('s')
    additions=[dict(role='user',content='x'*1048576 if fault=='body_limit' else 'hi')]
    options={'messages':[]} if fault=='owned_messages' else {'bad':object()} if fault=='not_serializable' else {}
    with pytest.raises((ClientStateError,TypeError)):c.submit(additions,options=options)
    assert not r.posts and c.identity is None and c.next_sequence==1 and c.messages==[]


def test_absence_lookup_then_active_retry_race_keeps_identity():
    r=ResetRuntime();c=InternalLocalClient(r);c.create('s');r.fail=True
    with pytest.raises(OSError):submit(c)
    identity=c.identity
    def active(sid,raw,seq):
        r.posts.append((sid,raw,seq))
        raise RuntimeHTTPError(409,'active request; observe session outcome')
    r.internal_fenced_request=active
    with pytest.raises(RuntimeHTTPError):c.reconcile()
    assert c.state=='ambiguous' and c.identity==identity and c.next_sequence==1
    assert all(p[1:]==(identity.body,1) for p in r.posts)


def test_dropped_owned_handle_closes_response_without_history():
    import gc
    r=ResetRuntime();c=InternalLocalClient(r);c.create('s');submit(c)
    before=c.messages;response=Response(b'data: {"choices": [{"delta": {"content": "display"}}]}\n\n')
    stream=OwnedStream(c,Conn(),response)
    del stream;gc.collect()
    assert response.closed and c.state=='ambiguous' and c.messages==before
    c.reconcile();assert c.state=='ready'


@pytest.mark.parametrize('additions', [[None], ['user'], [{'role':'tool','content':'invented'}]])
def test_invalid_local_additions_reject_before_mutation(additions):
    r=ResetRuntime(); c=InternalLocalClient(r); c.create('s')
    with pytest.raises(ClientStateError): c.submit(additions,options={})
    assert not r.posts and c.identity is None and c.next_sequence==1
