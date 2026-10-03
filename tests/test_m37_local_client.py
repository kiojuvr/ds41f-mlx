"""Synthetic client contract fault tests; real checkpoint evidence is separate."""
from copy import deepcopy
import hashlib
import io
import json
import pytest

from ds41f_mlx.internal_local_client import InternalLocalClient, ClientStateError, OwnedStream
from ds41f_mlx.web_client import RuntimeHTTPError


def outcome(body, seq=1, calls=None):
    msg=dict(role='assistant',content='hello' if calls is None else None)
    if calls:msg['tool_calls']=calls
    witness=deepcopy(body);witness['messages'].append(msg);witness['messages'].append(dict(role='user',content='Continue.'))
    return dict(sequence=seq,outcome_state='recoverable',certificate=dict(version=1,representable=True,
        exact_prefix=True,semantic_complete=True,first_mismatch=None,frontier=10,encoded_length=20,
        canonical_sha256='a'*64,executable_tools=bool(calls),witness=witness),
        response=dict(choices=[dict(message=msg,finish_reason='tool_calls' if calls else 'stop')]))


class Conn:
    sock=None
    def close(self):pass


class Response(io.BytesIO):
    fp=None
    def getheader(self,*a):return 'application/json'


class Runtime:
    def __init__(self):
        self.posts=[];self.next=1;self.slot=None;self.out=None;self.fail=False
    def create_session(self,sid):return dict(id=sid or 'test',outcome_state='not_admitted',next_sequence=1)
    def close_session(self,sid):return dict(state='closed')
    def get_session(self,sid):
        return dict(next_sequence=self.next,request_fence=self.slot,outcome_state='not_admitted' if self.slot is None else self.slot['state'])
    def internal_fenced_request(self,sid,raw,seq):
        self.posts.append((sid,raw,seq))
        if self.fail:
            self.fail=False;raise OSError('lost before admission')
        if self.slot is None or seq==self.next:
            self.slot=dict(sequence=seq,body_sha256=hashlib.sha256(raw).hexdigest(),state='recoverable')
            self.next=seq+1;self.out=outcome(json.loads(raw),seq)
            value=self.out['response'] # ordinary non-stream response, no certificate
        else:value=self.out
        return Conn(),Response(json.dumps(value).encode())


def setup():
    r=Runtime();c=InternalLocalClient(r);c.create();return r,c


def submit(c):return c.submit([dict(role='user',content='hi')],options=dict(stream=False))


def test_lost_unstarted_retry_frozen_bytes_and_sequence():
    r,c=setup();r.fail=True
    with pytest.raises(OSError):submit(c)
    identity=c.identity
    assert c.state=='ambiguous' and c.next_sequence==1
    c.reconcile() # absent GET is not ACK; same bytes resubmitted
    c.reconcile()
    assert c.state=='ready' and c.next_sequence==2
    assert all(p[1:]==(identity.body,1) for p in r.posts)
    assert c.messages==json.loads(identity.body)['messages']+[r.out['response']['choices'][0]['message']]


def test_active_never_advances_and_new_work_stops():
    r,c=setup();submit(c);r.slot['state']='active';r.next=1
    before=len(r.posts)
    assert c.reconcile()['outcome_state']=='active'
    assert c.next_sequence==1 and len(r.posts)==before
    with pytest.raises(ClientStateError):submit(c)
    assert c.state=='stopped'


@pytest.mark.parametrize('fault',['sequence','body','next','state'])
def test_disagreement_stops_without_substitute(fault):
    r,c=setup();submit(c)
    if fault=='sequence':r.slot['sequence']=0
    if fault=='body':r.slot['body_sha256']='b'*64
    if fault=='next':r.next=9
    if fault=='state':r.slot['state']='nonsense'
    before=len(r.posts)
    with pytest.raises(ClientStateError):c.reconcile()
    assert c.state=='stopped' and len(r.posts)==before


@pytest.mark.parametrize('fault',['certificate','identity','witness','tools','response','changed','hash','version','flag'])
def test_invalid_or_changed_outcome_stops(fault):
    r,c=setup();submit(c)
    if fault=='certificate':r.out['certificate']['representable']=False
    if fault=='hash':r.out['certificate']['canonical_sha256']='z'*64
    if fault=='version':r.out['certificate']['version']=True
    if fault=='flag':r.out['certificate']['executable_tools']=0
    if fault=='identity':r.out['sequence']=10
    if fault=='witness':r.out['certificate']['witness']['messages'][0]['content']='not this request'
    if fault=='tools':r.out['certificate']['executable_tools']=True
    if fault=='response':r.out['response']['choices']=[]
    if fault=='changed':
        c.reconcile();r.out['response']['choices'][0]['message']['content']='changed'
    with pytest.raises(ClientStateError):c.reconcile()
    assert c.state=='stopped'


def tool():return dict(id='call_1',type='function',function=dict(name='weather',arguments='{"city":"Paris"}'))


def certify_tool(r,c):
    submit(c);r.out=outcome(json.loads(c.identity.body),calls=[tool()]);c.reconcile()


def test_tool_effect_reserved_completed_reobserved_result_submitted():
    r,c=setup();certify_tool(r,c);effects=[]
    def execute(call):effects.append(call);return 'sunny'
    stored=c.execute_tools(execute)
    for _ in range(3):
        c.reconcile();assert c.execute_tools(execute)==stored
    assert len(effects)==1 and len(c.messages)==3
    c.submit_tool_results(options=dict(stream=False))
    assert json.loads(c.identity.body)['messages'][-1]==stored[0]
    assert c.identity.sequence==2


def test_ambiguous_effect_no_retry_even_after_reobservation():
    r,c=setup();certify_tool(r,c);effects=[]
    def fail(call):effects.append(call);raise OSError('effect result lost')
    with pytest.raises(OSError):c.execute_tools(fail)
    assert c.state=='tool_ambiguous' and list(c.ledger.values())[0]['status']=='reserved'
    c.reconcile()
    with pytest.raises(ClientStateError):c.execute_tools(fail)
    assert len(effects)==1


def test_ledger_identity_mismatch_and_bound():
    r,c=setup();certify_tool(r,c)
    c.execute_tools(lambda _: 'sunny')
    key=next(iter(c._ledger));c._ledger[key]['call']['function']['name']='changed'
    with pytest.raises(ClientStateError):c.execute_tools(lambda _:pytest.fail('duplicate'))
    r,c=setup();c.ledger_limit=0;certify_tool(r,c)
    with pytest.raises(ClientStateError):c.execute_tools(lambda _:pytest.fail('overflow'))


@pytest.mark.parametrize('state',['unrecoverable','poisoned'])
def test_negative_poison_never_insert_native_assistant_and_delete_fresh(state):
    r,c=setup();submit(c);identity=c.identity
    r.slot['state']=state;r.out.update(outcome_state=state,certificate=dict(representable=False))
    c.reconcile();assert c.state==state and c.messages==json.loads(identity.body)['messages']
    with pytest.raises(ClientStateError):c.execute_tools(lambda _:pytest.fail('negative effect'))
    c.retire();assert c.state=='retired'
    with pytest.raises(ClientStateError):c.observe_identity(identity)
    preserved=c.messages;c.create('fresh');assert c.messages==preserved and c.next_sequence==1


def test_expired_older_identity_never_regenerates():
    r,c=setup();submit(c);old=c.identity;c.reconcile();submit(c);c.reconcile()
    n=len(r.posts)
    assert c.observe_identity(old)['outcome_state']=='expired'
    assert len(r.posts)==n and c.state=='ready'


def test_current_identity_expired_requires_explicit_action():
    r,c=setup();submit(c);r.slot['sequence']=2;r.next=3
    n=len(r.posts)
    assert c.reconcile()['outcome_state']=='expired' and c.state=='expired'
    assert len(r.posts)==n
    with pytest.raises(ClientStateError):submit(c)


@pytest.mark.parametrize('action',['delete','create'])
def test_delete_and_create_failure_stop(action):
    r,c=setup();submit(c);c.reconcile()
    if action=='delete':
        r.close_session=lambda _:dict(state='idle')
        with pytest.raises(ClientStateError):c.retire()
    else:
        c.retire();r.create_session=lambda _:dict(id='bad',next_sequence=7)
        with pytest.raises(ClientStateError):c.create()
    assert c.state=='stopped'


def test_actual_stream_handle_interruption_discards_incomplete_fragment():
    r,c=setup();submit(c)
    response=Response(b'data: {"choices": [{"delta": {"content": "\xc3')
    stream=OwnedStream(c,Conn(),response)
    with pytest.raises((UnicodeDecodeError,json.JSONDecodeError)):list(stream)
    assert response.closed and c.state=='ambiguous'
    c.reconcile();assert c.messages[-1]['content']=='hello'


def test_http_body_fence_rejection_is_client_state_error_not_new_work():
    r,c=setup();submit(c);identity=c.identity
    def reject(*args):
        r.posts.append(args)
        raise RuntimeHTTPError(400, 'request sequence body mismatch')
    r.internal_fenced_request=reject
    with pytest.raises(ClientStateError,match='state error'):c.reconcile()
    assert c.state=='stopped' and c.identity==identity and c.next_sequence==1


def test_malformed_transport_outcome_is_not_a_retry_heuristic():
    r,c=setup()
    r.internal_fenced_request=lambda *args:(Conn(),Response(b'{invalid JSON'))
    with pytest.raises(ClientStateError,match='malformed HTTP'):submit(c)
    assert c.state=='stopped' and c.next_sequence==1
