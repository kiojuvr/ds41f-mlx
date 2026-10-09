"""Fresh capability/admission/projection faults; native execution labelled separately."""
import asyncio
from copy import deepcopy
import hashlib
import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from ds41f_mlx.config import RuntimeConfig
from ds41f_mlx.mtp_profile import PROFILE, LIMITS, WEATHER, strict_json, validate_chat, sequence, certificate_projection
from ds41f_mlx.serving.mtp_public import LocalMTPBackend, LocalBoundary
from ds41f_mlx.serving.server import create_app
from ds41f_mlx.mtp_identity import config

BASE = dict(model='deepseek-v4.1-flash',messages=[dict(role='user',content='Hello')],max_tokens=32)


@pytest.mark.parametrize('raw',[b'[]',b'null',b'{',b'{"a":1,"a":2}',b'{"a":{"x":1,"x":2}}',
    b'{"n":NaN}',b'{"n":Infinity}',b'{"n":1e999}',b'\xff',b'{"a":1} trailing'])
def test_strict_json(raw):
    with pytest.raises(ValueError):strict_json(raw)


@pytest.mark.parametrize('field,value', [('model','other'),('max_tokens',True),('max_tokens',0),
    ('max_tokens',769),('max_tokens',1.0),('temperature',True),('temperature',.5),('reasoning_effort','high'),
    ('stream',1),('stream',None),('tools',[]),('tools',[dict(WEATHER,extra=True)]),('tool_choice','auto'),
    ('stop',None),('stop',[]),('top_p',1),('top_k',1),('n',1),('seed',1),('logprobs',False),
    ('user','x'),('mtp',True),('dspark',True),('response_format',{'type':'json_object'}),
    ('stream_options',{'include_usage':True}),('max_completion_tokens',1),('artifact_path','/tmp/x'),
    ('messages',[]),('messages',[{'role':'user','content':[{'type':'text','text':'hi'}]}]),
    ('messages',[{'role':'user','content':'<|special|>'}]),
    ('messages',[{'role':'user','content':'hello','name':'x'}]),
    ('messages',[{'role':'tool','content':'x','tool_call_id':'bad'}])])
def test_allowlist(field,value):
    body=deepcopy(BASE);body[field]=value
    with pytest.raises(ValueError):validate_chat(json.dumps(body).encode())


@pytest.mark.parametrize('model',['deepseek-v4.1-flash','deepseek-v41-flash','deepseek-flash'])
@pytest.mark.parametrize('n',[1,768])
def test_positive_boundary(model,n):
    b=dict(BASE,model=model,max_tokens=n,temperature=0,reasoning_effort='none',stream=False)
    assert validate_chat(json.dumps(b).encode())==b


@pytest.mark.parametrize('value',[b'0',b'01',b'+1',b'-1',b' 1',b'1 ',b'18446744073709551616',b''])
def test_sequence_syntax(value):
    with pytest.raises(ValueError):sequence([(b'x-ds41f-request-sequence',value)])


def test_sequence_no_duplicate_and_ceiling():
    assert sequence([(b'x-ds41f-request-sequence',b'18446744073709551615')])==(1<<64)-1
    for headers in ([],[(b'x-ds41f-request-sequence',b'1')]*2):
        with pytest.raises(ValueError):sequence(headers)


def test_http_rejections_do_not_issue_or_mutate(monkeypatch):
    # Real recipe tokenizer/conversion; no model/native owner execution.
    import sys
    recipe=__import__('pathlib').Path(sys.prefix)/'share/ds41f-mtp/recipe'
    cfg=replace(config(),port=8000,recipe_path=recipe)
    b=LocalMTPBackend(runtime_config=cfg)
    b._retire=lambda _:None
    app=create_app(backend=b,runtime_config=cfg,profile=PROFILE)
    try:
        with TestClient(app,base_url='http://127.0.0.1:8000') as c:
            for raw in (b'[]',b'{"id":null}',b'{"id":""}',b'{"extra":1}',b'{"a":1,"a":2}'):
                assert c.post('/v1/sessions',content=raw,headers={'Content-Type':'application/json'}).status_code==400
                assert b._issued==0 and not b.sessions
            rec=c.post('/v1/sessions',json={}).json();sid=rec['id'];before=deepcopy(b.sessions[sid].to_json())
            p=f'/v1/sessions/{sid}/chat/completions'
            for route in ('/v1/chat/completions','/v1/responses','/v1/messages','/v1/sessions/restore',
                          f'/v1/sessions/{sid}/persist','/docs','/redoc','/openapi.json','/_ds41f/diagnostics'):
                assert c.post(route,content=b'not JSON').json()['error']['code']=='unsupported_capability'
            for headers in ({'Origin':'http://evil.invalid'},{'Host':'evil.invalid'},{'Content-Encoding':'gzip'},
                            {'Content-Type':'text/plain'},{'X-DS41F-Request-Sequence':'01'}):
                h={'Content-Type':'application/json','X-DS41F-Request-Sequence':'1'};h.update(headers)
                assert c.post(p,content=json.dumps(BASE),headers=h).status_code==400
                assert before==b.sessions[sid].to_json() and not b._lock.locked()
            assert c.get('/health').json()['profile']==PROFILE
            status=c.get('/v1/sessions/'+sid).json()
            assert not {'last_turn','witness','canonical','target_offsets'} & set(status)
            assert c.post('/v1/sessions',json={}).status_code==409
            assert b._issued==1
            assert c.delete('/v1/sessions/'+sid,headers={'Content-Type':'application/json'}).status_code==200
            assert not b.sessions
    finally:b.close()


@pytest.mark.parametrize('env,value',[('DS41F_HOST','0.0.0.0'),('DS41F_HOST','localhost'),
    ('DS41F_MAX_LIVE_SESSIONS','4'),('DS41F_ENABLE_DIAGNOSTIC_ENDPOINTS','1'),
    ('DS41F_OMLX_PATH','/tmp/any'),('DS41F_RECIPE_PATH','/tmp/any'),('PYTHONPATH','.'),
    ('OMLX_MTP_ROW_EXACT_VERIFY','0'),('OMLX_MTP_PRIME_WINDOW','128'),('DS41F_P8_TILE_NATIVE_CARRY','1'),
    ('WEB_CONCURRENCY','2'),('UVICORN_FD','3'),('DS41F_TRACE_HISTORY_LIMIT','2048'),('DYLD_LIBRARY_PATH','/tmp')])
def test_config_fail_closed(monkeypatch,env,value):
    monkeypatch.setenv(env,value)
    with pytest.raises(ValueError):config()


def test_bounded_chunked_body_timeout_slot_and_send_cleanup():
    async def run():
        async def app(scope,receive,send):
            while True:
                e=await receive()
                if not e.get('more_body'):break
            await send({'type':'http.response.start','status':200,'headers':[]})
            await send({'type':'http.response.body','body':b'{}'})
        boundary=LocalBoundary(app,authority='127.0.0.1:8000',body_timeout=.02,send_timeout=.02)
        scope=dict(type='http',method='POST',path='/v1/sessions',query_string=b'',
                   headers=[(b'host',b'127.0.0.1:8000'),(b'content-type',b'application/json')])
        output=[]
        async def send(e):output.append(e)
        events=iter([dict(type='http.request',body=b'x'*600000,more_body=True),
                     dict(type='http.request',body=b'x'*600000,more_body=False)])
        async def receive():return next(events)
        await boundary(scope,receive,send)
        assert output[0]['status']==413 and not boundary.preparing
        output.clear()
        async def slow():await asyncio.sleep(1)
        task=asyncio.create_task(boundary(scope,slow,send))
        await asyncio.sleep(.001)
        extra=[]
        async def another(e):extra.append(e)
        await boundary(scope,slow,another)
        assert extra[0]['status']==409
        await task
        assert output[0]['status']==408 and not boundary.preparing
        # A response send stall calls the actual owned-iterator finally path.
        closed=[]
        async def response(scope,receive,send):
            try:
                await send({'type':'http.response.start','status':200,'headers':[]})
                await send({'type':'http.response.body','body':b'x'})
            finally:closed.append(True)
        boundary=LocalBoundary(response,authority='127.0.0.1:8000',send_timeout=.02)
        async def stall(e):
            if e['type']=='http.response.body':await asyncio.sleep(1)
        with pytest.raises(TimeoutError):await boundary(scope,slow,stall)
        assert closed==[True] and not boundary.preparing
    asyncio.run(run())


def test_weather_schema_types_and_real_result_order():
    weather=deepcopy(WEATHER);weather['function']['strict']=1
    with pytest.raises(ValueError):validate_chat(json.dumps(dict(BASE,tools=[weather])).encode())
    weather=deepcopy(WEATHER);weather['function']['parameters']['additionalProperties']=0
    with pytest.raises(ValueError):validate_chat(json.dumps(dict(BASE,tools=[weather])).encode())
    calls=[dict(id='call_'+str(i),type='function',function=dict(name='lookup_weather',arguments='{"city":"Paris"}')) for i in range(2)]
    b=dict(BASE,tools=[WEATHER],messages=BASE['messages']+[dict(role='assistant',content=None,tool_calls=calls),
        dict(role='tool',tool_call_id='call_0',content='sunny'),dict(role='tool',tool_call_id='call_1',content='sunny')])
    assert validate_chat(json.dumps(b).encode())==b
    b['messages'][-1]['tool_call_id']='call_0'
    with pytest.raises(ValueError):validate_chat(json.dumps(b).encode())


def test_no_profile_fallback():
    with pytest.raises(ValueError, match='no fallback'):
        create_app(profile=PROFILE, runtime_config=replace(RuntimeConfig(),mtp='ON',max_live_sessions=1))
    with pytest.raises(ValueError):create_app(profile='unknown')


def test_public_projection_identity_faults():
    from ds41f_mlx.local_client import LocalMTPClient,ClientStateError
    from ds41f_mlx.internal_local_client import InternalLocalClient,RequestIdentity
    request=deepcopy(BASE)
    msg=dict(role='assistant',content='Hello')
    raw=json.dumps(request).encode()
    cert=dict(version=1,frontier=5,canonical_sha256='a'*64,representable=True,executable_tools=False,
              exact_prefix=True,semantic_complete=True,first_mismatch=None,encoded_length=10,
              profile_supported=True, witness={'secret':'must not be exposed'})
    response=dict(choices=[dict(message=msg,finish_reason='stop')])
    projection=certificate_projection(cert,request,response)
    assert 'witness' not in projection
    for fault in (None,'profile','body','binding','response','projection','canonical_hash','version'):
        c=LocalMTPClient.__new__(LocalMTPClient);InternalLocalClient.__init__(c,None)
        c._pending=RequestIdentity('test',1,raw)
        out=dict(sequence=1,outcome_state='recoverable',profile=PROFILE,body_sha256=hashlib.sha256(raw).hexdigest(),
                 certificate=deepcopy(projection),response=deepcopy(response))
        if fault=='profile':out['profile']='standard-off'
        if fault=='body':out['body_sha256']='b'*64
        if fault=='binding':out['certificate']['ordinary_binding_sha256']='b'*64
        if fault=='response':out['response']['choices'][0]['message']['content']='edited'
        if fault=='projection':out['certificate']['projection']='unknown'
        if fault=='canonical_hash':out['certificate']['canonical_sha256']='z'*64
        if fault=='version':out['certificate']['version']=True
        if fault:
            with pytest.raises(ClientStateError):c._accept(out)
            assert c.state=='stopped'
        else:
            c._accept(out);assert c.state=='ready' and c.messages==request['messages']+[msg]


def test_public_create_shape_and_handshake_fail_before_effects(monkeypatch):
    from ds41f_mlx import local_client as module
    sid='mtp_'+'a'*32+'_'+f'{1:032x}'
    valid=dict(id=sid,profile=PROFILE,state='empty',outcome_state='not_admitted',next_sequence=1,
               canonical_frontier=0,request_count=0,request_fence=None,certificate=None)
    class Runtime:
        def __init__(self,_):self.operations=[]
        def health(self):return dict(profile=PROFILE,limits=LIMITS,dependency_identity='a'*64)
        def create_session(self,_):self.operations.append('create');return deepcopy(valid)
        def close_session(self,_):self.operations.append('delete');return dict(state='closed')
    monkeypatch.setattr(module,'RuntimeClient',Runtime)
    for field,value in (('id','test'),('id','mtp_'+'a'*32+'_'+'0'*32),('next_sequence',True),
                        ('canonical_frontier',1),('profile','standard-off'),('state','idle')):
        old=valid[field];valid[field]=value
        c=module.LocalMTPClient()
        with pytest.raises(module.ClientStateError):c.create()
        assert c.state=='stopped' and c.lifecycle_uncertain['action']=='create'
        with pytest.raises(module.ClientStateError):c.retire()
        assert c.runtime.operations==['create']
        valid[field]=old
    c=module.LocalMTPClient();c.create();assert c.session_id==sid and c.dependency_identity=='a'*64
    for h in (dict(profile='standard-off',limits=LIMITS,dependency_identity='a'*64),
              dict(profile=PROFILE,limits=LIMITS,dependency_identity=None),
              dict(profile=PROFILE,limits=dict(LIMITS,live_sessions=True),dependency_identity='a'*64)):
        monkeypatch.setattr(Runtime,'health',lambda _,h=h:h)
        with pytest.raises(module.ClientStateError):module.LocalMTPClient()


def test_mtp_fence_conflict_codes_are_pre_mutation(monkeypatch):
    c=TestClient(create_app(profile=PROFILE,runtime_config=config(),backend=LocalMTPBackend(runtime_config=config())),base_url='http://127.0.0.1:8000')
    sid=c.post('/v1/sessions',json={}).json()['id']
    before=c.get('/v1/sessions/'+sid).json()
    gap=c.post('/v1/sessions/'+sid+'/chat/completions',json=BASE,headers={'X-DS41F-Request-Sequence':'2'})
    assert gap.status_code==409 and gap.json()['error']['code']=='request_sequence_gap'
    assert c.get('/v1/sessions/'+sid).json()==before
    c.app.state.backend.close()


def test_living_frozen_retry_when_previous_slot_is_still_latest():
    from ds41f_mlx.internal_local_client import InternalLocalClient, RequestIdentity
    from types import SimpleNamespace
    class Runtime:
        def get_session(self,_):return dict(outcome_state='recoverable',next_sequence=2,
            request_fence=dict(sequence=1,state='recoverable',body_sha256='old_body'))
    c=InternalLocalClient(Runtime());c.session_id='s';c.next_sequence=2;c.state='ambiguous'
    c._pending=RequestIdentity('s',2,b'current_frozen')
    c._send=lambda: c._pending
    assert c.reconcile() is c._pending and c.next_sequence==2 and c.state=='ambiguous'
    # This observation never advances or substitutes a third identity.


def test_legacy_python_tool_client_refuses_mtp_before_inference():
    from ds41f_mlx.web_client import StatefulToolChatClient
    from ds41f_mlx.web_tools import ToolRegistry, ToolError
    class Runtime:
        def health(self):return {'profile':PROFILE}
        def chat(self,*_):raise AssertionError('unqualified mutation')
    with pytest.raises(ToolError):
        StatefulToolChatClient(Runtime(),ToolRegistry()).run_turn(
            session_id='s',transcript=[],user_message='hi')


def test_unsupported_completed_output_is_negative_profile_not_repair(monkeypatch):
    from types import SimpleNamespace
    from ds41f_mlx.serving.internal_mtp import InternalMTPQualificationBackend
    b=LocalMTPBackend(runtime_config=config())
    try:
        monkeypatch.setattr(InternalMTPQualificationBackend,'_settle',lambda *_:None)
        for message in (dict(role='assistant',content='raw <|source|>'),
                        dict(role='assistant',content=None),
                        dict(role='assistant',content=None,tool_calls=[dict(id='c',type='function',
                            function=dict(name='unqualified',arguments='{"city":"Paris"}'))])):
            rec=SimpleNamespace(certificate={'representable':True,'executable_tools':True},poisoned=False,unrecoverable=False)
            trace={'response':{'choices':[{'message':message}]}}
            b._finalize_application_certificate(rec,trace)
            assert rec.unrecoverable and rec.poisoned and rec.certificate['representable'] is False
            assert rec.certificate['profile_supported'] is False
    finally:b.close()


def test_normal_local_checkpoint_byte_cache_invalidates_changed_file(tmp_path, monkeypatch):
    # New normal-local boundary: metadata alone may not certify weight bytes.
    from ds41f_mlx import mtp_identity as identity
    monkeypatch.setattr(identity, 'RECORD', tmp_path/'identity.json')
    shard = tmp_path/'model.safetensors'
    shard.write_bytes(b'qualified')
    assets = {'shards':[{'name':shard.name,
                        'lfs_sha256':hashlib.sha256(b'qualified').hexdigest()}]}
    identity.verify_checkpoint_bytes(tmp_path, assets)
    receipt = (tmp_path/'checkpoint-bytes.json').read_bytes()
    identity.verify_checkpoint_bytes(tmp_path, assets)
    assert (tmp_path/'checkpoint-bytes.json').read_bytes() == receipt
    shard.write_bytes(b'corrupted')  # equal size, changed file identity
    with pytest.raises(ValueError, match='checkpoint bytes'):
        identity.verify_checkpoint_bytes(tmp_path, assets)
    assert (tmp_path/'checkpoint-bytes.json').read_bytes() == receipt


def test_normal_local_seal_rejects_unqualified_native_wheel(tmp_path):
    from ds41f_mlx import mtp_identity as identity
    wheel = tmp_path/'unqualified.whl'
    wheel.write_bytes(b'not the qualified M52R artifact')
    with pytest.raises(ValueError, match='repository-qualified M52R wheel'):
        identity.main(['seal', '--wheel', str(wheel)])
