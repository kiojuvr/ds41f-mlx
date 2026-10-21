"""Controlled native tokens, ordinary admission and shared serving ownership."""
import json
from pathlib import Path
import sys

import deepseek_recipe as d
import pytest

from ds41f_mlx.runtime.recipe_semantic_guard import MessagesSemanticGuard
from ds41f_mlx.serving.messages_admission import convert_messages
from test_recipe02_semantic_guard import encode, feed, PREFIX, ARG_END, TAIL

TOKENIZER = Path(sys.prefix)/'share/ds41f-mtp/recipe/static/tokenizers/v41/tokenizer.json'


def payload(**kwargs):
    return dict(model='deepseek-flash', messages=[dict(role='user', content='hi')],
                max_tokens=256, thinking={'type': 'disabled'}, **kwargs)


def convert(body):
    return convert_messages(json.dumps(body).encode())[0]


def owner(thinking=False):
    body = payload(tools=[{'name': n, 'input_schema': {'type': 'object'}} for n in ('f', 'g')])
    body['thinking']['type'] = 'enabled' if thinking else 'disabled'
    request = convert(body)
    tok = d.Tokenizer.from_file(str(TOKENIZER))
    g = d.MessagesRequest.chunk_generator(request, 'response', 'model').with_signature('exact-signature')
    session = d.MessagesSemanticSession(g, request.parsing_options, tok)
    guard = MessagesSemanticGuard(session, frontier=17)
    guard._push(d.InferenceChunk.ready(prompt_usage=d.PromptUsage(prompt_tokens=17)))
    return tok, guard, body


@pytest.mark.parametrize('thinking', [False, True])
def test_exact_native_events_and_continuation(thinking):
    tok, guard, body = owner(thinking)
    text = ('inspect</think>' if thinking else '') + 'intro ' + PREFIX + 'Tokyo🤭' + ARG_END
    feed(guard, encode(tok, text))
    assert len(guard.call_certificates) == 1 and not guard.tool_complete
    feed(guard, encode(tok, 'invoke><｜DSML｜invoke name="g"></｜DSML｜invoke>'))
    assert len(guard.call_certificates) == 2 and not guard.finished
    feed(guard, encode(tok, '</｜DSML｜tool_calls'))
    assert not guard.finished
    ids = encode(tok, '>after')
    prediction = guard.preview(ids)
    expected = prediction.identity[1].outcome
    assert feed(guard, ids) == 1
    assert guard.finished and guard.tool_complete
    assert guard.response.to_json() == expected
    response = json.loads(expected)
    accumulated = d.MessagesResponse('guard', 'guard', 0, 0, 0)
    for e in guard.events:
        before = e.to_json()
        accumulated.accumulate_copy(e)
        assert e.to_json() == before
    assert accumulated.to_json() == expected
    calls = [b for b in response['content'] if b['type'] == 'tool_use']
    assert calls == [json.loads(c.call) for c in guard.call_certificates]
    assert calls[0]['input'] == {'x': 'Tokyo🤭'}
    events = [json.loads(e.to_json()) for e in guard.events]
    assert events[0]['type'] == 'message_start' and events[-1]['type'] == 'message_stop'
    assert all(d.MessagesResponse.sse_event_name(e) == json.loads(e.to_json())['type'] for e in guard.events)
    if thinking:
        assert response['content'][0]['signature'] == 'exact-signature'
    body['messages'] += [dict(role='assistant', content=response['content']), dict(role='user', content=[
        dict(type='tool_result', tool_use_id=c['id'], content='ok') for c in calls])]
    continued = convert(body)
    assistant = next(m for m in continued.conversation.messages if m.role == 'assistant')
    assert [c.id for c in assistant.tool_calls] == [c['id'] for c in calls]
    encoding = d.DeepseekV41Encoding().with_tokenizer(tok)
    assert len(encoding.encode(continued.conversation)) > 17
    guard.close()


@pytest.mark.parametrize('fault', ['stale', 'divergent', 'foreign', 'response'])
def test_shared_proofs_fail_closed(fault):
    tok, guard, _ = owner()
    ids = encode(tok, PREFIX + 'ok' + ARG_END + TAIL)
    pred = guard.preview(ids)
    assert guard.preview(ids).identity[1].outcome == pred.identity[1].outcome
    if fault == 'stale':
        guard.processor.push(d.InferenceChunk.tokens([ids[0]]))
        action = lambda: guard.observe_canonical_emit(ids[0], None)
    elif fault == 'divergent':
        action = lambda: guard.observe_canonical_emit(tok.encode('different')[0], None)
    elif fault == 'response':
        guard.response = d.MessagesResponse('foreign', 'model', 0, 0, 0)
        action = lambda: guard.observe_canonical_emit(ids[0], None)
    else:
        _, other, _ = owner()
        foreign = other.preview(ids)
        for t in ids[:pred.index]:
            guard.observe_canonical_emit(t, None)
        action = lambda: guard.observe_canonical_emit(ids[pred.index], foreign.identity)
    with pytest.raises(RuntimeError):
        action()
    assert guard.failed and guard.processor.status[3] and not guard.tool_complete
    assert not guard.call_certificates


def test_messages_user_stop_and_backend_control():
    tok, guard, body = owner()
    # Native stop-sequence parsing remains request-owned, not a wire scanner.
    guard.close()
    body['stop_sequences'] = ['HALT']
    converted = convert(body)
    generator = d.MessagesRequest.chunk_generator(converted, 'response', 'model')
    guard = MessagesSemanticGuard(d.MessagesSemanticSession(generator, converted.parsing_options, tok))
    guard._push(d.InferenceChunk.ready(prompt_usage=d.PromptUsage(prompt_tokens=17)))
    feed(guard, tok.encode('answer HA'))
    ids = tok.encode('LT after')
    pred = guard.preview(ids)
    assert pred.kind == 'RECIPE_STOP'
    assert feed(guard, ids) == pred.index + 1
    response = json.loads(guard.response.to_json())
    assert response['stop_reason'] == 'stop_sequence'
    assert response['stop_sequence'] == 'HALT'
    assert response['content'][0]['text'] == 'answer '
    guard.close()
    tok, guard, _ = owner()
    eos = tok.encode(d.SpecialToken.Eos.default_str())[0]
    guard.control_token_ids = frozenset([eos])
    ids = tok.encode('hello') + [eos]
    feed(guard, ids)
    assert guard.emitted == len(ids) and guard.processor.status[1] == len(ids)-1
    guard.finish_backend('stop')
    assert json.loads(guard.response.to_json())['stop_reason'] == 'end_turn'
    guard.close()


def test_rejected_candidates_partial_accept_and_cancellation():
    tok, guard, _ = owner()
    before = guard.processor.status, guard.response.to_json()
    guard.preview(encode(tok, PREFIX + 'rejected' + ARG_END + TAIL))
    assert (guard.processor.status, guard.response.to_json()) == before
    feed(guard, encode(tok, PREFIX))
    feed(guard, encode(tok, 'accepted' + ARG_END))
    guard.close()
    assert guard.failed and not guard.call_certificates and not guard.tool_complete


@pytest.mark.parametrize('end,reason', [('stop', 'tool_use'), ('length', 'max_tokens'), ('eof', 'end_turn')])
def test_incomplete_block_never_certifies_tools(end, reason):
    tok, guard, _ = owner()
    feed(guard, encode(tok, PREFIX + 'ok' + ARG_END))
    if end == 'eof':
        guard.finish_eof()
    else:
        guard.finish_backend(end)
    assert guard.finished and not guard.tool_complete
    assert json.loads(guard.response.to_json())['stop_reason'] == reason
    guard.close()


@pytest.mark.parametrize('extension', [
    {'unknown': True}, {'output_config': {'format': {'type': 'json_schema'}}},
    {'thinking': {'type': 'enabled', 'budget_tokens': 10}},
    {'tool_choice': {'type': 'any'}},
    {'tool_choice': {'type': 'auto', 'disable_parallel_tool_use': True}},
    {'tools': [{'type': 'web_search_20250305', 'name': 'web_search'}]},
    {'messages': [{'role': 'user', 'content': [{'type': 'document', 'source': {'media_type': 'text/plain'}}]}]},
    {'messages': [{'role': 'user', 'content': [{'type': 'image', 'source': {'type': 'file', 'file_id': 'f'}}]}]},
    {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'hi', 'cache_control': {}}]}]},
])
def test_unsupported_wire_rejected(extension):
    with pytest.raises((ValueError, d.ConversionError)):
        convert(payload() | extension)


def test_in_history_system_and_tool_addition():
    body = payload(tools=[{'name': 'f', 'input_schema': {'type': 'object'}, 'defer_loading': True},
                          {'name': 'g', 'input_schema': {'type': 'object'}}])
    body['messages'] += [{'role': 'system', 'content': [
        {'type': 'text', 'text': 'updated instructions'},
        {'type': 'tool_addition', 'tool': {'type': 'tool_reference', 'name': 'f'}}]}]
    converted = convert(body)
    tok = d.Tokenizer.from_file(str(TOKENIZER))
    encoding = d.DeepseekV41Encoding().with_tokenizer(tok)
    assert encoding.encode(converted.conversation)
    with pytest.raises((ValueError, d.ConversionError)):
        convert(body | {'model': 'unadmitted'})


def test_production_sessions_reject_messages_before_reservation():
    import asyncio
    from types import SimpleNamespace as NS
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    backend = object.__new__(ProductionMTPBackend)
    async def run():
        with pytest.raises(ValueError, match='Chat Completions only'):
            await backend.production_session_response('not-owned', NS(protocol='messages', stream=True),
                tokenizer=None, body=b'{}', expected_count=0, application_id=None)
    asyncio.run(run())


def test_official_prompt_encoding_and_expanded_image_capacity(tmp_path):
    import base64
    import io
    from PIL import Image
    from ds41f_mlx.serving.server import prepare_request, RequestError
    tok = d.Tokenizer.from_file(str(TOKENIZER))
    checkpoint = tmp_path
    # Use the existing checkpoint geometry; no weights or executable cache.
    from ds41f_mlx.config import load_runtime_config
    config = json.loads((load_runtime_config().checkpoint_path/'config.json').read_text())
    config.get('text_config', config)['max_position_embeddings'] = 8192
    (checkpoint/'config.json').write_text(json.dumps(config))
    body = payload(system='system instructions')
    prepared = prepare_request('messages', json.dumps(body).encode(), tokenizer=tok,
                               checkpoint=checkpoint, ordinary=True)
    converted = convert(body)
    encoding = d.DeepseekV41Encoding().with_tokenizer(tok)
    assert prepared.token_ids == list(encoding.encode(converted.conversation))
    image = io.BytesIO()
    Image.new('RGB', (32, 32)).save(image, format='PNG')
    body['messages'][0]['content'] = [dict(type='image', source=dict(type='base64',
        media_type='image/png', data=base64.b64encode(image.getvalue()).decode())), dict(type='text', text='describe')]
    prepared = prepare_request('messages', json.dumps(body).encode(), tokenizer=tok,
                               checkpoint=checkpoint, ordinary=True)
    assert prepared.multimodal is not None and len(prepared.token_ids) > 100
    config.get('text_config', config)['max_position_embeddings'] = len(prepared.token_ids) + 255
    overflow = tmp_path/'overflow'
    overflow.mkdir()
    (overflow/'config.json').write_text(json.dumps(config))
    with pytest.raises(RequestError, match='requested output exceeds'):
        prepare_request('messages', json.dumps(body).encode(), tokenizer=tok,
                        checkpoint=overflow, ordinary=True)
    prepared.multimodal.release_sources()


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('thinking', [False, True])
def test_ordinary_http_native_transport(monkeypatch, stream, thinking):
    # Real ingress/preparation/ordinary_response/pump/cleanup, controlled native
    # output only. No claim of checkpoint or numerical Scheduler execution.
    import os
    from types import SimpleNamespace as NS
    from fastapi.testclient import TestClient
    from ds41f_mlx.mtp_identity import config
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    from ds41f_mlx.serving.server import create_app
    previous = dict(os.environ)
    backend = ProductionMTPBackend(runtime_config=config(profile='mtp-serving-v1'))
    backend.dependency_identity = 'controlled'
    scheduler = NS(requests={}, close_serving=lambda: None)
    backend.scheduler = scheduler
    expected = {}
    def enqueue(prepared, tokenizer, rid):
        assert prepared.protocol == 'messages'
        converted = prepared.conversation_request
        session = d.MessagesSemanticSession(d.MessagesRequest.chunk_generator(converted, rid, 'model'),
                                           converted.parsing_options, tokenizer)
        guard = MessagesSemanticGuard(session)
        guard._push(d.InferenceChunk.ready(prompt_usage=d.PromptUsage(prompt_tokens=len(prepared.token_ids))))
        feed(guard, encode(tokenizer, ('inspect</think>' if thinking else '') + 'hello'))
        guard.finish_backend('stop')
        expected['events'] = [e.to_json() for e in guard.events]
        expected['body'] = json.loads(guard.response.to_json())
        scheduler.requests[rid] = True
        guard.close()
    def step():
        rid = next(iter(scheduler.requests))
        scheduler.requests.clear()
        return NS(outputs=[NS(request_id=rid, finished=True, error=None,
            recipe_events=expected['events'], recipe_response=expected['body'])])
    scheduler.step = step
    monkeypatch.setattr(backend, '_enqueue', enqueue)
    try:
        app = create_app(profile='mtp-serving-v1', backend=backend, runtime_config=backend.runtime_config)
        with TestClient(app, base_url='http://127.0.0.1:8000') as client:
            body = payload(stream=stream)
            body['thinking']['type'] = 'enabled' if thinking else 'disabled'
            result = client.post('/v1/messages', json=body)
            assert result.status_code == 200, result.text
            if stream:
                from ds41f_mlx.serving.server import sse_frame
                assert result.text == ''.join(sse_frame(json.loads(e)['type'], e) for e in expected['events'])
            else:
                assert result.json() == expected['body']
            assert not backend._deliveries and not backend._settled
            assert client.post('/v1/messages', json=body | {'extension': True}).status_code == 400
            assert client.post('/v1/sessions/x/messages', json=body).status_code == 400
    finally:
        backend.scheduler = None
        backend.close()
        os.environ.clear()
        os.environ.update(previous)
