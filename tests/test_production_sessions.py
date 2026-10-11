"""Production session protocol is logical identity over the SAME Scheduler."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest

from ds41f_mlx.serving.production_sessions import ProductionSessions


def prepared(ids=(1,2,3,4)):
    return NS(protocol='chat_completions', image_sources=[], token_ids=list(ids))


def test_requested_ids_are_nonreusable_and_session_has_no_native_owner():
    sessions = ProductionSessions(NS(fatal_error=None))
    rec = sessions.create('known-id')
    assert not any(hasattr(rec, name) for name in ('cache', 'rings', 'owner', 'guard', 'processor'))
    assert rec.to_json()['state'] == 'empty'
    with pytest.raises(RuntimeError, match='maximum live'):
        sessions.create('other')
    del sessions.records[rec.session_id]
    with pytest.raises(RuntimeError, match='already issued'):
        sessions.create('known-id')
    with pytest.raises(KeyError):
        sessions.get('known-id')


def test_new_session_cannot_rebuild_foreign_canonical_assistant_history():
    sessions = ProductionSessions(NS(fatal_error=None)); rec = sessions.create()
    with pytest.raises(ValueError, match='no session replay'):
        sessions.validate(rec.session_id, prepared(), {'messages': [
            {'role':'user','content':'old question'}, {'role':'assistant','content':'old response'},
            {'role':'user','content':'continue'}]})
    assert not rec.busy and rec.request_count == 0


def test_failed_cleanup_keeps_admitted_identity_fenced(monkeypatch):
    pytest.importorskip('omlx')
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    from ds41f_mlx.mtp_identity import config
    async def check():
        backend = ProductionMTPBackend(runtime_config=config(profile='mtp-serving-v1'))
        rec = await backend.create_stateful_session()
        backend.scheduler = NS(requests={})
        async def failed(prepared, **kwargs):
            rid = kwargs['request_id']
            backend.scheduler.requests[rid] = True
            raise RuntimeError('uncertain admitted cleanup')
        monkeypatch.setattr(backend, 'ordinary_response', failed)
        req = prepared(); req.stream = True
        try:
            with pytest.raises(RuntimeError, match='uncertain admitted'):
                await backend.production_session_response(rec.session_id, req, tokenizer=None,
                    body=json.dumps({'messages':[{'role':'user','content':'Hi'}]}).encode(),
                    expected_count=0, application_id=str(uuid4()))
            assert rec.busy and rec.publication['state'] == 'unrecoverable' and rec.request_count == 0
            with pytest.raises(RuntimeError, match='active request'):
                await backend.close_stateful_session(rec.session_id)
        finally:
            # Synthetic owner has no native payload. Never production recovery.
            backend.sessions.clear(); backend.close()
    asyncio.run(check())


def test_counts_correlation_and_busy_are_not_delivery_acknowledgements():
    sessions = ProductionSessions(NS(fatal_error=None)); rec = sessions.create()
    body = {'messages': [{'role': 'user', 'content': 'Hi'}]}
    appid = str(uuid4())
    sessions.reserve(rec.session_id, prepared(), body, 0, appid, 'r1')
    assert rec.request_count == 0 and rec.to_json()['canonical_frontier'] == 0
    assert rec.to_json()['state'] == 'busy'
    with pytest.raises(RuntimeError, match='active request'):
        sessions.reserve(rec.session_id, prepared(), body, 0, str(uuid4()), 'r2')
    sessions.unstarted(rec, 'foreign', 'ignored')
    assert rec.busy
    sessions.unstarted(rec, 'r1', 'cancel before enqueue')
    assert not rec.busy and rec.request_count == 0
    with pytest.raises(RuntimeError, match='already reserved'):
        sessions.reserve(rec.session_id, prepared(), body, 0, appid, 'retry')
    with pytest.raises(RuntimeError, match='stale'):
        sessions.reserve(rec.session_id, prepared(), body, 1, str(uuid4()), 'stale')
    with pytest.raises(ValueError, match='canonical UUID'):
        sessions.reserve(rec.session_id, prepared(), body, 0, 'not-a-UUID', 'r')


def publication(sessions, rec, message=None, representable=True, retained=True, error=None):
    message = message or {'role': 'assistant', 'content': 'Hello'}
    body = {'messages': [{'role': 'user', 'content': 'Hi'}]}
    trace = dict(quiescence={}, response={'choices': [{'message': message, 'finish_reason': 'tool_calls' if message.get('tool_calls') else 'stop'}]},
                 cancelled=True, canonical_frontier=4, target_offsets=[4]*40, dspark_offsets=[4]*3,
                 prompt_replay=0, full_cache_repack=0, cache_published=True, cached_tokens=0, generated=1)
    request = NS(logical_session=rec, trace=trace, session_pair_retained=retained, request_id='r1', application_id=str(uuid4()),
                 execution=NS(canonical=[1,2,3,4], certificate={'representable': representable, 'executable_tools': bool(message.get('tool_calls')), 'witness': {}}),
                 prepared=NS(capacity={}), session_body=body)
    sessions.publish(request, error)
    return request


def test_atomic_canonical_outcome_history_and_exact_token_fencing():
    sessions = ProductionSessions(NS(fatal_error=None)); rec = sessions.create()
    publication(sessions, rec)
    data = rec.to_json()
    assert data['state'] == 'idle' and data['request_count'] == 1 and data['canonical_frontier'] == 4
    assert data['last_turn']['cancelled'] and data['last_turn']['response_json']['choices'][0]['message']['content'] == 'Hello'
    assert 'witness' not in data['last_turn']['reconstruction']
    body = {'messages': deepcopy(rec.history)+[{'role': 'user', 'content': 'next'}]}
    sessions.validate(rec.session_id, prepared([1,2,3,4,5,6]), body, 1)
    body['messages'][0]['content'] = 'different session history'
    with pytest.raises(ValueError, match='history differs'):
        sessions.validate(rec.session_id, prepared([1,2,3,4,5,6]), body)
    body['messages'] = deepcopy(rec.history)
    with pytest.raises(ValueError, match='canonical token frontier'):
        sessions.validate(rec.session_id, prepared([1,2,3,9,5,6]), body)
    data['last_turn']['response_json']['choices'][0]['message']['content'] = 'tamper'
    assert rec.history[-1]['content'] == 'Hello'


@pytest.mark.parametrize('representable,retained,error', [(False,True,None),(True,False,None),(True,True,RuntimeError('fatal'))])
def test_unrepresentable_or_unpinned_or_failed_outcome_fails_closed(representable, retained, error):
    sessions = ProductionSessions(NS(fatal_error=None)); rec = sessions.create()
    publication(sessions, rec, representable=representable, retained=retained, error=error)
    assert rec.to_json()['state'] == 'unrecoverable'
    assert rec.request_count == (0 if error else 1)
    with pytest.raises(RuntimeError, match='unrecoverable'):
        sessions.validate(rec.session_id, prepared(), {'messages': []})


def test_tool_results_bound_to_certified_session_call_without_new_content_grammar():
    sessions = ProductionSessions(NS(fatal_error=None)); rec = sessions.create()
    call = {'id': 'c1', 'type': 'function', 'function': {'name': 'read', 'arguments': '{}'}}
    publication(sessions, rec, message={'role': 'assistant', 'content': None, 'tool_calls': [call]})
    body = {'messages': deepcopy(rec.history)+[{'role': 'tool', 'tool_call_id': 'c1', 'content': 'result'}]}
    sessions.validate(rec.session_id, prepared([1,2,3,4,5,6]), body)
    body['messages'][-1]['tool_call_id'] = 'other'
    with pytest.raises(ValueError, match='certified call batch'):
        sessions.validate(rec.session_id, prepared([1,2,3,4,5,6]), body)
    body['messages'][-1]['tool_call_id'] = 'c1'
    body['messages'].append(body['messages'][-1])
    with pytest.raises(ValueError, match='duplicate tool result'):
        sessions.validate(rec.session_id, prepared([1,2,3,4,5,6]), body)


def test_production_dispatch_never_qualification_and_budget_is_observation(monkeypatch):
    pytest.importorskip('omlx')
    from fastapi.testclient import TestClient
    from fastapi.responses import StreamingResponse
    from ds41f_mlx.mtp_identity import config
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    from ds41f_mlx.serving.server import create_app
    backend = ProductionMTPBackend(runtime_config=config(profile='mtp-serving-v1'))
    backend.dependency_identity = 'admitted-fixture'
    async def forbidden(*args, **kwargs):
        raise AssertionError('qualification/standard-OFF owner exposed')
    monkeypatch.setattr(backend, 'qualification_response', forbidden)
    monkeypatch.setattr(backend, 'open_stateful_stream', forbidden)
    seen = []
    async def broker(prepared, *, tokenizer, http_request, request_id, logical_session, body, application_id):
        seen.append((prepared, logical_session, body, application_id))
        assert logical_session.busy and logical_session.request_count == 0
        backend.production_sessions.unstarted(logical_session, request_id, 'fixture has no native execution')
        async def stream():
            yield 'data: [DONE]\n\n'
        return StreamingResponse(stream(), headers={'X-DS41F-Request-ID':request_id}, media_type='text/event-stream')
    monkeypatch.setattr(backend, 'ordinary_response', broker)
    try:
        with TestClient(create_app(backend=backend, runtime_config=backend.runtime_config, profile='mtp-serving-v1')) as client:
            assert client.get('/health').json()['capabilities']['stateful_chat'] == 'production-sessions-v1'
            rec = client.post('/v1/sessions', json={'id':'app-session'}).json()
            sid = rec['id']; path = '/v1/sessions/'+sid
            body = dict(model='deepseek-v4.1-flash', messages=[{'role':'user','content':'Hi'}], stream=True,
                        temperature=0, top_p=1, reasoning_effort='none', max_tokens='auto')
            budget = client.post(path+'/budget', json=body)
            assert budget.status_code == 200 and budget.json()['max_tokens'] == 393216
            assert client.get(path).json()['request_count'] == 0 and not seen
            assert client.post(path+'/chat/completions', json=body).status_code == 400
            headers = {'X-DS41F-Expected-Request-Count':'0','X-DS41F-Application-Request-ID':str(uuid4())}
            response = client.post(path+'/chat/completions', json=body, headers=headers)
            assert response.status_code == 200 and response.headers['x-ds41f-request-id']
            assert len(seen) == 1
            assert client.post(path+'/chat/completions', json=body, headers=headers).status_code == 409
            assert len(seen) == 1
            for unsupported in (path+'/persist', '/v1/sessions/restore'):
                assert client.post(unsupported, json={}).status_code == 400
            assert client.post(path+'/cancel', content=b'{"request_id":"a","request_id":"b"}',
                               headers={'Content-Type':'application/json'}).status_code == 400
            closed = client.delete(path, headers={'Content-Type':'application/json'})
            assert closed.status_code == 200 and closed.json()['active_request_id'] is None
            assert client.get(path).status_code == 404
            assert client.post('/v1/sessions', json={'id':sid}).status_code == 409
    finally:
        backend.close()
