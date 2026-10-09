"""Bounded fence lifecycle regression; synthetic, no checkpoint claims."""
import asyncio
from copy import deepcopy
import json
import pytest
from ds41f_mlx.serving.internal_mtp import QualificationSession
from ds41f_mlx.serving.request_fence import observe_retry, reserve, finish
from ds41f_mlx.serving.recovery_certificate import LocalToolLedger
from test_m35_transport_lease import FixtureBackend, request


def test_fence_not_admitted_active_recoverable_expired_atomic():
    rec=QualificationSession('one');body=b'{}'
    assert observe_retry(rec,1,body) is None
    before=deepcopy(rec.to_json())
    for seq in [0,2,99]:
        with pytest.raises(ValueError):observe_retry(rec,seq,body)
        assert rec.to_json()==before
    reserve(rec,1,body)
    with pytest.raises(RuntimeError):observe_retry(rec,1,body)
    before=deepcopy(rec.to_json())
    with pytest.raises(ValueError):observe_retry(rec,1,b'{ }')
    assert rec.to_json()==before
    finish(rec,unstarted=True)
    assert observe_retry(rec,1,body) is None and rec.consumed_sequence==0
    reserve(rec,1,body);rec.fence['started']=True
    rec.certificate={'representable':True};rec.last_turn={'response':{'choices':[]}}
    finish(rec)
    before=deepcopy(rec.to_json())
    for _ in range(3):assert observe_retry(rec,1,body)['outcome_state']=='recoverable'
    assert rec.to_json()==before
    reserve(rec,2,body)
    with pytest.raises(ValueError):observe_retry(rec,1,body)
    assert rec.fence['sequence']==2
    # Worker-certified consumption is observable while its original socket
    # still owns the busy delivery lease. Transport cannot delay permission.
    certified=dict(session_id='one',sequence=2,body_sha256=rec.fence['body_sha256'],
                   outcome_state='recoverable',certificate={'representable':True},
                   response={'choices':[]},canonical_events=['immutable'])
    rec.last_turn={'certified_outcome':json.dumps(certified)}
    rec.fence['state']='recoverable';rec.consumed_sequence=2;rec.busy=True
    assert rec.to_json()['outcome_state']=='recoverable'
    assert rec.to_json()['next_sequence']==3
    assert observe_retry(rec,2,body)==certified
    rec.certificate['representable']=False
    assert observe_retry(rec,2,body)==certified


@pytest.mark.parametrize('unrecoverable,poisoned,state',[(True,True,'unrecoverable'),(False,True,'poisoned')])
def test_failure_states_are_not_cache_coherence(unrecoverable,poisoned,state):
    rec=QualificationSession('one',unrecoverable=unrecoverable,poisoned=poisoned)
    reserve(rec,1,b'{}');rec.fence['started']=True;finish(rec)
    assert rec.fence['state']==rec.to_json()['outcome_state']==state
    rec.busy=True;assert rec.to_json()['outcome_state']=='active'


def test_header_failure_is_retryable_with_same_sequence():
    async def run():
        b=FixtureBackend();rec=await b.create_stateful_session(session_id='one')
        body=json.dumps({'messages':[]}).encode()
        r=await b.qualification_response('one',request(),tokenizer=None,body=body,sequence=1)
        async def receive():await asyncio.sleep(10)
        async def send(_):raise OSError('header failed')
        with pytest.raises(BaseException):await r({'type':'http','asgi':{'spec_version':'2.4'}},receive,send)
        assert rec.fence['state']=='not_admitted' and rec.consumed_sequence==0 and not rec.busy
        # Retry reserves same identity, not a new assistant outcome.
        r=await b.qualification_response('one',request(),tokenizer=None,body=body,sequence=1)
        with pytest.raises(BaseException):await r({'type':'http','asgi':{'spec_version':'2.4'}},receive,send)
        b.close()
    asyncio.run(run())


def test_new_sequence_cannot_bypass_exact_prefix_to_generate_committed_turn_again():
    async def run():
        b=FixtureBackend();rec=await b.create_stateful_session(session_id='one')
        body=b'{"messages": []}'
        reserve(rec,1,body);rec.fence['started']=True
        rec.canonical=[1,2,3];rec.certificate={'representable':True};rec.last_turn={'response':{'choices':[]}}
        finish(rec);before=deepcopy(rec.to_json())
        with pytest.raises(ValueError,match='exactly extend'):
            await b.qualification_response('one',request(),tokenizer=None,body=body,sequence=2)
        assert rec.to_json()==before and not b._lock.locked() and not b.session_traces
        b.close()
    asyncio.run(run())


def test_started_internal_failure_consumes_identity_and_is_observable_not_retryable():
    async def run():
        b=FixtureBackend();rec=await b.create_stateful_session(session_id='one')
        def fail(*_):raise RuntimeError('protected failure')
        b._start=fail;body=b'{"messages": []}'
        r=await b.qualification_response('one',request(),tokenizer=None,body=body,sequence=1)
        with pytest.raises(RuntimeError,match='protected failure'):await anext(r.body_iterator)
        assert not rec.busy and rec.consumed_sequence==1 and rec.fence['state']=='poisoned'
        before=deepcopy(rec.to_json())
        retry=await b.qualification_response('one',request(),tokenizer=None,body=body,sequence=1)
        assert json.loads(retry.body)['outcome_state']=='poisoned' and rec.to_json()==before
        b.close()
    asyncio.run(run())


def test_delete_failure_is_internal_poison_not_protocol_limitation():
    async def run():
        b=FixtureBackend();rec=await b.create_stateful_session(session_id='one')
        rec.unrecoverable=rec.poisoned=True
        def fail(_):raise RuntimeError('retirement failed')
        b._retire=fail
        with pytest.raises(RuntimeError):await b.close_stateful_session('one')
        assert rec.to_json()['outcome_state']=='poisoned' and not rec.busy
        b.close()
    asyncio.run(run())


def test_tools_reserve_before_effect_and_never_retry_ambiguous_effect():
    ledger=LocalToolLedger()
    out={'sequence':1,'outcome_state':'recoverable','certificate':{'representable':True,'executable_tools':True},'response':{'choices':[{'message':{'tool_calls':[{'id':'call','function':{'name':'x','arguments':'{}'}}]}}]}}
    calls=[]
    with pytest.raises(ValueError):ledger.execute('s',2,out,lambda _:calls.append(1))
    out['outcome_state']='active'
    with pytest.raises(ValueError):ledger.execute('s',1,out,lambda _:calls.append(1))
    out['outcome_state']='recoverable'
    def fail(call):calls.append(call);raise OSError('ambiguous side effect')
    with pytest.raises(OSError):ledger.execute('s',1,out,fail)
    with pytest.raises(RuntimeError):ledger.execute('s',1,out,fail)
    assert len(calls)==1
    out['certificate']['representable']=False
    with pytest.raises(ValueError):ledger.execute('s',1,out,fail)
    out['certificate']['representable']=True
    owned=LocalToolLedger(); effects=[]
    result=owned.execute('s',1,out,lambda call: effects.append(call) or 'done')
    assert owned.execute('s',1,deepcopy(out),lambda _: pytest.fail('duplicate effect')) == result
    conflicting=deepcopy(out)
    conflicting['response']['choices'][0]['message']['tool_calls'][0]['function']['arguments']='{"changed":true}'
    with pytest.raises(ValueError,match='conflicting'):
        owned.execute('s',1,conflicting,lambda _: pytest.fail('conflicting effect'))
    assert len(effects)==1
    from ds41f_mlx.serving.internal_mtp import InternalMTPQualificationBackend as B
    rec=QualificationSession('s',reconstruction_body={'messages':[]})
    certified=deepcopy(out); certified['session_id']='s'
    msg=certified['response']['choices'][0]['message']
    rec.last_turn={'certified_outcome':json.dumps(certified)}
    body=json.dumps({'messages':[msg]+result}).encode()
    B._validate_result_reentry(rec,body,2)
    B._validate_result_reentry(rec,body,2)
    for seq, changed in [(3,body),(2,json.dumps({'messages':[msg,dict(role='tool',tool_call_id='foreign',content='done')]}).encode()),
                         (2,json.dumps({'messages':[msg]+result+result}).encode())]:
        with pytest.raises(ValueError):B._validate_result_reentry(rec,changed,seq)
    certified['certificate']['executable_tools']=False
    rec.last_turn={'certified_outcome':json.dumps(certified)}
    with pytest.raises(ValueError):B._validate_result_reentry(rec,body,2)
