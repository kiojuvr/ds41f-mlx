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


@pytest.mark.parametrize('inventory', [None, False, 42, 'opaque', [],
    {'packages': [{'name': 'example', 'version': '1'}], 'unknown': {'model': 'ignored'}}])
def test_plugin_inventory_is_excluded_before_recipe(monkeypatch, inventory):
    native_request = d.MessagesRequest
    admitted = []
    def request(raw):
        admitted.append(raw)
        return native_request(raw)
    monkeypatch.setattr(d, 'MessagesRequest', request)
    baseline = convert(payload())
    decorated = convert(payload(dsh_plugin_packages=inventory))
    assert len(admitted) == 2 and admitted[0] == admitted[1]
    assert 'dsh_plugin_packages' not in json.loads(admitted[1])
    encoding = d.DeepseekV41Encoding()
    assert encoding.render_conversation(baseline.conversation).prompt == encoding.render_conversation(decorated.conversation).prompt


@pytest.mark.parametrize('metadata,reason', [
    ('{"x": 1, "x": 2}', 'duplicate JSON key'),
    ('NaN', 'nonfinite JSON number'),
    ('1e999', 'nonfinite JSON number'),
])
def test_plugin_inventory_still_obeys_json_admission(metadata, reason):
    raw = json.dumps(payload())[:-1] + ', "dsh_plugin_packages": ' + metadata + '}'
    with pytest.raises(ValueError, match=reason):
        convert_messages(raw.encode())


def test_plugin_inventory_still_counts_toward_body_limit():
    from ds41f_mlx.serving.capacity import ORDINARY_BODY_BYTES
    raw = json.dumps(payload(dsh_plugin_packages='x' * ORDINARY_BODY_BYTES)).encode()
    with pytest.raises(ValueError, match='body limit exceeded'):
        convert_messages(raw)


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
    {'unknown': True}, {'dsh_session_log': {}},
    {'dsh_session_log': {}, 'dsh_plugin_packages': []},
    {'unknown': True, 'dsh_plugin_packages': []},
    {'output_config': {'format': {'type': 'json_schema'}}},
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


def inline_image(color='red'):
    import base64
    from io import BytesIO
    from PIL import Image
    stream = BytesIO()
    Image.new('RGB', (32, 32), color).save(stream, format='PNG')
    return dict(type='image', source=dict(type='base64', media_type='image/png',
        data=base64.b64encode(stream.getvalue()).decode()))


def image_results_body(slots, *, trailing=False, historical=False, identical=False):
    ids = ['Read:Exact-A', 'Read:Exact-B', 'Read:Exact-C'][:len(slots)]
    body = payload(tools=[dict(name='read_image', input_schema=dict(type='object'))])
    body['messages'] = [dict(role='user', content=(
        [inline_image(), dict(type='text', text='inspect')] if historical else 'inspect')),
        dict(role='assistant', content=[dict(type='tool_use', id=i, name='read_image', input={}) for i in ids]),
        dict(role='user', content=[dict(type='tool_result', tool_use_id=i,
            content=[inline_image('red' if identical else color)] if color else 'done')
            for i, color in zip(ids, slots)])]
    if trailing:
        body['messages'][-1]['content'].append(dict(type='text', text='compare now'))
    return body, ids


def enqueue_images(body):
    from types import SimpleNamespace as NS
    from ds41f_mlx.serving.server import prepare_request
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    tok = d.Tokenizer.from_file(str(TOKENIZER))
    prepared = prepare_request('messages', json.dumps(body).encode(), tokenizer=tok, ordinary=True)
    queued = []
    backend = NS(fatal_error=None, load=lambda: None,
                 scheduler=NS(add_request=queued.append), request_max_tokens=lambda p: 256)
    ProductionMTPBackend._enqueue(backend, prepared, tok, 'image')
    return prepared, queued[0]


@pytest.mark.parametrize('slots,trailing,historical,identical', [
    (['red'], False, False, False),
    (['red', None, None], False, False, False),
    ([None, 'red', None], False, False, False),
    ([None, None, 'red'], False, False, False),
    (['red', None], False, False, False),
    (['red'], True, False, False),
    (['blue', None], False, True, False),
    (['red', None], True, True, True),
    (['blue', 'green', 'red'], True, True, False),
])
def test_wire_image_origin_survives_recipe_expansion(slots, trailing, historical, identical):
    from hashlib import sha256
    from ds41f_mlx.runtime.multimodal import image_bytes
    body, ids = image_results_body(slots, trailing=trailing, historical=historical, identical=identical)
    # Return order need not be tool-call order. Exact opaque IDs survive.
    body['messages'][1]['content'].reverse()
    prepared, request = enqueue_images(body)
    converted = prepared.conversation_request.conversation.messages
    assert [m.tool_call_id for m in converted if m.role == 'tool'] == ids
    assert [c.id for m in converted if m.role == 'assistant' for c in m.tool_calls] == ids[::-1]
    sources = [s for m in converted for s in m.image_sources]
    old_paths = ['messages.0.content[0]'] if historical else []
    new_paths = [f'messages.2.content.{i}.tool_result' for i, color in enumerate(slots) if color]
    assert [s.path for s in sources] == old_paths + new_paths
    # Recipe renders results in tool-call order, not returned-result order.
    # Multimodal identities and origin metadata must follow RENDERED sources.
    assert [s.path for s in prepared.image_sources] == old_paths + new_paths[::-1]
    assert [sha256(image_bytes(s)).hexdigest() for s in prepared.image_sources] == [
        s['sha256'] for s in request.image_identities]
    assert prepared.image_is_historical == (True,) * int(historical) + (False,) * sum(bool(c) for c in slots)
    expected = request.image_identities[0]['start'] + request.image_identities[0]['length'] if historical else 0
    assert request.required_image_frontier == expected
    assert all(s.start + s.length <= len(prepared.token_ids)-2 for s in prepared.multimodal.spans)
    prepared.multimodal.release_sources()


def test_multiple_occurrences_in_one_result_keep_rendered_order():
    from hashlib import sha256
    from ds41f_mlx.runtime.multimodal import image_bytes
    body, _ = image_results_body(['red', None], historical=True, identical=True)
    body['messages'][-1]['content'][0]['content'] += [inline_image('blue'), inline_image()]
    prepared, request = enqueue_images(body)
    assert prepared.image_is_historical == (True, False, False, False)
    assert [s.path for s in prepared.image_sources] == ['messages.0.content[0]'] + [
        'messages.2.content.0.tool_result'] * 3
    identities = request.image_identities
    assert [i['sha256'] for i in identities] == [sha256(image_bytes(s)).hexdigest() for s in prepared.image_sources]
    assert identities[0]['sha256'] == identities[1]['sha256'] == identities[3]['sha256']
    assert identities[0]['sha256'] != identities[2]['sha256']
    assert [i['start'] for i in identities] == sorted({i['start'] for i in identities})
    assert request.required_image_frontier == identities[0]['start'] + identities[0]['length']
    prepared.multimodal.release_sources()


@pytest.mark.parametrize('origins', [None, (), (False, False)])
def test_missing_or_misaligned_messages_origins_reject_before_queue(origins):
    from types import SimpleNamespace as NS
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    body, _ = image_results_body(['red'])
    prepared, _ = enqueue_images(body)
    prepared.image_is_historical = origins
    queued = []
    backend = NS(fatal_error=None, load=lambda: None,
                 scheduler=NS(add_request=queued.append), request_max_tokens=lambda p: 256)
    with pytest.raises(ValueError):
        ProductionMTPBackend._enqueue(backend, prepared, None, 'invalid-origin')
    assert not queued
    prepared.multimodal.release_sources()


@pytest.mark.parametrize('historical', [False, True])
@pytest.mark.parametrize('cache', ['missing', 'retained', 'evicted', 'changed'])
def test_converted_image_frontier_at_real_paired_admission(monkeypatch, historical, cache):
    from types import SimpleNamespace as NS
    import mlx.core as mx
    from ds41f_mlx.serving.paired_checkpoint import PairedCheckpointAuthority
    from test_metadata_first_vision import scheduler_request
    from test_production_mtp import MX, pair
    body, _ = image_results_body(['red', None], historical=historical, identical=True)
    retained_images = None
    if historical and cache == 'changed':
        previous, _ = enqueue_images(body)
        retained_images = previous.multimodal.identities()[:1]
        previous.multimodal.release_sources()
        # A real wire-history edit, converted by Recipe, not just a forged hash.
        body['messages'][0]['content'][0] = inline_image('blue')
    prepared, enqueued = enqueue_images(body)
    inp = prepared.multimodal
    authority = PairedCheckpointAuthority('identity', MX)
    frontier = inp.spans[-1].start
    if cache != 'missing':
        images = inp.identities()[:int(historical)] if retained_images is None else retained_images
        authority.publish(authority.capture(*pair(frontier)[:2], inp.token_ids[:frontier], images))
        if cache == 'evicted':
            authority.clear()
    scheduler, request = scheduler_request(inp, authority)
    request.prepared = prepared
    request.required_image_frontier = enqueued.required_image_frontier
    errors, starts, encoded = [], [], []
    def start(rec, prepared, tokenizer, trace, **kwargs):
        starts.append(len(rec.canonical))
        rows = inp.encode_new(NS(config=inp._config), len(rec.canonical),
            span_encoder=lambda pixels, h, w, kinds: mx.ones((len(kinds), 4)))
        encoded.extend(s for s, _ in rows.rows)
        scheduler.batch_generator = object()
        rec.owner = NS(uid=17, _bg=scheduler.batch_generator)
    scheduler.backend = NS(_start=start)
    scheduler._retire = lambda req, *, error: errors.append(error)
    scheduled, _ = scheduler._schedule_waiting()
    if historical and cache != 'retained':
        assert not scheduled and not starts and not encoded
        assert len(errors) == 1 and 'historical image pair unavailable' in str(errors[0])
        assert not request.native_started
    else:
        assert not errors, errors
        assert scheduled == [request]
        assert encoded == [inp.spans[-1].start]
    assert not inp._sources
    authority.clear()


@pytest.mark.parametrize('inventory', [False, True])
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('thinking', [False, True])
def test_ordinary_http_native_transport(monkeypatch, stream, thinking, inventory):
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
            if inventory:
                body['dsh_plugin_packages'] = {'packages': [{'name': 'example'}]}
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
            assert client.post('/v1/messages', json=body | {'dsh_session_log': {}}).status_code == 400
            assert client.post('/v1/messages', json=body | {'model': 'unadmitted'}).status_code == 400
            assert client.post('/v1/sessions/x/messages', json=body).status_code == 400
    finally:
        backend.scheduler = None
        backend.close()
        os.environ.clear()
        os.environ.update(previous)
