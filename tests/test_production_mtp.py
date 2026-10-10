"""Regression for paired immutable cache authority and ordinary MTP ingress."""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest

from ds41f_mlx.serving.paired_checkpoint import PairedCheckpoint, validate_pair
from ds41f_mlx.serving.capacity import TEXT_QUALIFIED_ENVELOPE


class Cache:
    compress_ratio = 0
    meta_state = ('deepseek_v41', '3', '0')
    def __init__(self, C):
        self.cache = [np.array([C])] + [np.full((1, 2, 3), i) for i in range(1, 7)]
    def size(self):
        return int(self.cache[0][0])
    @classmethod
    def from_state(cls, state, meta):
        result = cls(0)
        result.cache = list(state)
        return result


class Ring:
    def __init__(self, capacity):
        self.max_size = capacity
        self.offset = 0
        self.keys = None


def pair(C=193):
    target = [Cache(C) for _ in range(40)]
    rings = [Ring(128) for _ in range(3)]
    for stage, ring in enumerate(rings):
        ring.offset = C
        # Absolute modulo slot contents after wrap, NOT chronological rotation.
        positions = np.arange(C-128, C)
        physical = np.empty(128, dtype=np.int64)
        physical[positions % 128] = positions + 1000*stage
        ring.keys = physical.reshape(1, 1, 128, 1)
    return target, rings, list(range(C))


MX = SimpleNamespace(eval=lambda *arrays: None)


def test_paired_checkpoint_copies_both_directions_and_physical_wrap():
    target, rings, tokens = pair()
    saved = PairedCheckpoint.capture('identity', target, rings, tokens, MX)
    target[0].cache[1][...] = -1
    rings[0].keys[...] = -1
    restored, restored_rings, restored_tokens = saved.restore('identity', tokens+[999], MX)
    assert restored_tokens == tokens
    assert np.array_equal(restored_rings[0].keys, pair()[1][0].keys)
    assert np.all(restored[0].cache[1] == 1)
    restored_rings[1].keys[...] = -2
    restored[0].cache[6][...] = -2
    again, again_rings, _ = saved.restore('identity', tokens+[999], MX)
    assert np.all(again[0].cache[6] == 6)
    assert np.array_equal(again_rings[1].keys, pair()[1][1].keys)


def test_pair_rejects_frontier_identity_and_branch_mismatch():
    target, rings, tokens = pair()
    saved = PairedCheckpoint.capture('identity', target, rings, tokens, MX)
    with pytest.raises(ValueError):
        saved.restore('different', tokens, MX)
    with pytest.raises(ValueError):
        saved.restore('identity', tokens[:-1]+[-1], MX)
    rings[1].offset -= 1
    with pytest.raises(ValueError):
        validate_pair(target, rings, tokens)


def test_upstream_lookup_eviction_and_invalid_pair_miss():
    pytest.importorskip('mlx.core')
    pytest.importorskip('omlx')
    from ds41f_mlx.serving.paired_checkpoint import PairedCheckpointAuthority
    authority = PairedCheckpointAuthority('identity', MX, retained_checkpoints=2)
    first = authority.capture(*pair(193))
    second = authority.capture(*pair(200))
    authority.publish(first)
    authority.publish(second)
    timings = {}
    assert len(authority.acquire(list(range(202)), timings=timings)[2]) == 200
    assert set(timings) == {'cache_lookup_s', 'paired_restore_s'}
    assert all(value >= 0 for value in timings.values())
    # Earlier edit cannot borrow a later ring. Longest intact earlier pair wins.
    branch = list(range(202))
    branch[195] = -1
    assert len(authority.acquire(branch)[2]) == 193
    assert authority.acquire(list(range(192))) is None
    from dataclasses import replace, FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        second.rings[0].offset = 199
    # Simulate damaged internal payload; malformed pairs must fail closed.
    key = authority.paged.find_cached_block(list(second.tokens)).block_hash
    authority._payloads[key] = replace(second, rings=(replace(second.rings[0], offset=199), *second.rings[1:]))
    assert len(authority.acquire(list(range(202)))[2]) == 193
    authority.publish(authority.capture(*pair(210)))
    authority.publish(authority.capture(*pair(220)))
    assert len(authority._payloads) <= 2
    assert authority.paged.get_stats().total_tokens_cached == sum(len(c.tokens) for c in authority._payloads.values())
    authority.clear()
    assert not authority._payloads
    assert authority.acquire(list(range(222))) is None


@pytest.mark.parametrize('frontier', [193, 32768, TEXT_QUALIFIED_ENVELOPE - 3])
def test_pinned_mlx_native_arrays_and_wrapped_ring_snapshot_are_stable(frontier):
    mx = pytest.importorskip('mlx.core')
    pytest.importorskip('omlx')
    from omlx.patches.deepseek_v41.cache import DeepseekV41Cache
    from omlx.patches.mlx_lm_mtp.deepseek_v4_dspark import DSparkContextCache
    target, rings, tokens = pair(frontier)
    target = [DeepseekV41Cache.from_state([mx.array(v) for v in c.cache], c.meta_state) for c in target]
    native_rings = []
    for ring in rings:
        item = DSparkContextCache(128)
        item.keys, item.offset = mx.array(ring.keys), ring.offset
        native_rings.append(item)
    saved = PairedCheckpoint.capture('identity', target, native_rings, tokens, mx)
    before = [np.array(r.keys) for r in saved.rings]
    for c in target:
        c.cache[1][:] = 999
    for r in native_rings:
        r.append(mx.full((1, 1, 7, 1), -1), start_offset=len(tokens))
    restored, restored_rings, _ = saved.restore('identity', tokens+[999], mx)
    for got, expected in zip(restored_rings, before):
        assert np.array_equal(np.array(got.keys), expected)
        got.append(mx.full((1, 1, 3, 1), -2), start_offset=len(tokens))
    for got, expected in zip(saved.rings, before):
        assert np.array_equal(np.array(got.keys), expected)
    assert np.all(np.array(restored[0].cache[1]) == 1)


def test_runtime_source_inventory_matches_repository_admission():
    # Source edits must ship with the reviewed inventory, not require disabling
    # startup identity checks or blindly resealing an unapproved source tree.
    from ds41f_mlx.mtp_identity import qualified, runtime_inventory
    actual, expected = runtime_inventory(), qualified()['runtime']
    changed = sorted(path for path in actual.keys() | expected.keys()
                     if actual.get(path) != expected.get(path))
    assert not changed, f'unadmitted runtime source changes: {changed}'


def test_ordinary_request_has_no_session_sequence_and_keeps_execution_bounds():
    pytest.importorskip('mlx.core')
    pytest.importorskip('omlx')
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    import json
    body = {'model': 'deepseek-v4.1-flash', 'messages': [{'role': 'user', 'content': 'Hi'}]}
    ProductionMTPBackend.validate_ordinary(json.dumps(body).encode())
    for unsupported in ({'temperature': 1}, {'seed': 1}, {'reasoning_effort': 'high'}):
        with pytest.raises(ValueError):
            ProductionMTPBackend.validate_ordinary(json.dumps(body | unsupported).encode())


@pytest.mark.parametrize('host,authority', [
    ('127.0.0.1', '127.0.0.1:8000'),
    ('0.0.0.0', '192.168.68.56:8000'),
    ('192.168.68.56', 'mac-studio.local:8000'),
    ('::', '[fd12::56]:8000'),
])
def test_ordinary_http_recipe_route_has_no_compulsory_public_session(monkeypatch, host, authority):
    pytest.importorskip('mlx.core')
    pytest.importorskip('omlx')
    import os
    from fastapi.testclient import TestClient
    from fastapi.responses import JSONResponse
    from ds41f_mlx.mtp_identity import config
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    from ds41f_mlx.serving.server import create_app
    previous = dict(os.environ)
    backend = ProductionMTPBackend(runtime_config=config(host, profile='mtp-serving-v1'))
    backend.dependency_identity = 'fixture-admitted-identity'
    seen = []
    async def response(prepared, *, tokenizer, http_request):
        seen.append(prepared.token_ids)
        assert prepared.protocol == 'chat_completions'
        phases = prepared.phase_timings
        assert phases['request_prepare_s'] >= phases['recipe_convert_or_encode_s'] >= 0
        assert phases['recipe_convert_or_encode_s'] == sum(phases[k] for k in
            ('recipe_convert_s', 'recipe_render_s', 'recipe_encode_s'))
        assert prepared.arrival_t0 > 0
        assert not backend.sessions
        return JSONResponse({'choices': [{'message': {'role': 'assistant', 'content': 'OK'}}]})
    monkeypatch.setattr(backend, 'ordinary_response', response)
    from ds41f_mlx.serving import ordinary_admission
    parsed = []
    native_request = ordinary_admission.ChatCompletionRequest
    def tracked_request(raw):
        parsed.append(raw)
        return native_request(raw)
    monkeypatch.setattr(ordinary_admission, 'ChatCompletionRequest', tracked_request)
    try:
        app = create_app(profile='mtp-serving-v1', runtime_config=backend.runtime_config, backend=backend)
        with TestClient(app, base_url=f'http://{authority}') as client:
            advertised = client.get('/v1/models')
            assert advertised.status_code == 200
            assert advertised.json()['data'][0]['context_length'] == backend.context_tokens
            assert advertised.json()['data'][0]['max_output_tokens'] == 393216
            for headers in ({'Host': ''}, {'Host': 'bad host'}, {'Origin': 'http://other.invalid'}):
                assert client.get('/v1/models', headers=headers).status_code == 400
            assert client.get('/v1/models', headers=[('Host', authority), ('Host', authority)]).status_code == 400
            result = client.post('/v1/chat/completions', json={
                'model': 'deepseek-v4.1-flash', 'messages': [{'role': 'user', 'content': 'Hello'}]})
            assert result.status_code == 200
            for maximum in (769, 32768, 393216):
                result = client.post('/v1/chat/completions', json={
                    'model': 'deepseek-v4.1-flash', 'max_tokens': maximum,
                    'messages': [{'role': 'user', 'content': 'Hello'}]})
                assert result.status_code == 200
            result = client.post('/v1/chat/completions', json={
                'model': 'deepseek-v4.1-flash', 'max_tokens': 393217,
                'messages': [{'role': 'user', 'content': 'Hello'}]})
            assert result.status_code == 400
            assert 'output capability ceiling' in result.json()['error']['message']
            for changes in (
                {'messages': [{'role': 'user', 'content': [
                    {'type': 'text', 'text': 'こんにちは'},
                    {'type': 'text', 'text': '<system-reminder>Plan mode</system-reminder>'}]}]},
                {'messages': [{'role': 'latest_reminder', 'content': 'Read only'}]},
                {'stream': True, 'stream_options': {'include_usage': False}},
                {'stop': ['END'], 'n': 1, 'top_p': 1, 'client_metadata': {}},
                {'max_tokens': 'auto'},
                {'messages': [
                    {'role': 'user', 'content': 'Review the repository'},
                    {'role': 'assistant', 'content': None, 'tool_calls': [
                        {'id': 'read-source', 'type': 'function', 'function': {
                            'name': 'read', 'arguments': '{"filePath":"ds41f_mlx/mtp_profile.py"}'}}]},
                    {'role': 'tool', 'tool_call_id': 'read-source', 'content': (
                        Path(__file__).resolve().parents[1] / 'ds41f_mlx/mtp_profile.py').read_text()},
                ]},
            ):
                before = len(parsed)
                result = client.post('/v1/chat/completions', json={
                    'model': 'deepseek-v4.1-flash',
                    'messages': [{'role': 'user', 'content': 'Hello'}]} | changes)
                assert result.status_code == 200, result.text
                assert len(parsed) == before + 1  # no duplicate recipe conversion
            for changes in (
                {'temperature': 1}, {'n': 2}, {'seed': 1},
                {'messages': [{'role': 'user', 'content': [{'type': 'image_url',
                    'image_url': {'url': 'https://example.com/a.png'}}]}]},
                {'messages': [{'role': 'tool', 'tool_call_id': 'foreign', 'content': 'OK'}]},
            ):
                before = len(seen)
                result = client.post('/v1/chat/completions', json={
                    'model': 'deepseek-v4.1-flash',
                    'messages': [{'role': 'user', 'content': 'Hello'}]} | changes)
                assert result.status_code == 400
                assert len(seen) == before  # never reaches execution admission
            assert seen and not backend.sessions
            assert client.post('/v1/sessions', json={}).status_code == 404
            assert client.post('/v1/responses', json={}).status_code == 400
    finally:
        os.environ.clear()
        os.environ.update(previous)
        backend.close()


def test_json_delivery_stops_disconnect_probe_even_when_probe_consumes_cancellation(monkeypatch):
    pytest.importorskip('mlx.core')
    pytest.importorskip('omlx')
    import asyncio
    import sys
    import time
    from pathlib import Path
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    async def run():
        backend = ProductionMTPBackend(omlx_path=Path(sys.prefix)/'lib/python3.13/site-packages')
        scheduler = SimpleNamespace(requests={})
        def enqueue(prepared, tokenizer, rid):
            scheduler.requests[rid] = True
        def step():
            time.sleep(.03)
            rid = next(iter(scheduler.requests))
            scheduler.requests.clear()
            return SimpleNamespace(outputs=[SimpleNamespace(request_id=rid, finished=True, error=None,
                recipe_events=(), recipe_response={'choices': []})])
        scheduler.step = step
        backend.scheduler = scheduler
        monkeypatch.setattr(backend, '_enqueue', enqueue)
        class Probe:
            async def is_disconnected(self):
                try:
                    await asyncio.sleep(.1)
                except asyncio.CancelledError:
                    pass  # Models Starlette's inner cancelling AnyIO scope.
                return False
        prepared = SimpleNamespace(protocol='chat_completions', image_sources=None, token_ids=[1,2,3],
                                   stream=False, inference_options=SimpleNamespace(max_tokens=16))
        try:
            response = await asyncio.wait_for(backend.ordinary_response(prepared, tokenizer=None, http_request=Probe()), 2)
            assert response.body == b'{"choices":[]}'
            assert not backend._deliveries and not backend._settled
        finally:
            backend.close()
    asyncio.run(run())


@pytest.mark.parametrize('cancel,fail,frontier', [
    (False, False, 200), (True, False, 32768), (False, True, 32768),
    (False, False, 8192), (False, False, TEXT_QUALIFIED_ENVELOPE - 3),
    (False, False, TEXT_QUALIFIED_ENVELOPE - 2), (False, False, TEXT_QUALIFIED_ENVELOPE)])
def test_scheduler_adoption_settlement_publication_and_burn(monkeypatch, cancel, fail, frontier):
    pytest.importorskip('mlx.core')
    pytest.importorskip('omlx')
    from collections import deque
    from time import perf_counter
    from omlx.request import Request, SamplingParams
    import importlib
    generate = importlib.import_module('mlx_lm.generate')
    from ds41f_mlx.serving.production_mtp import ProductionScheduler
    from ds41f_mlx.serving.internal_mtp import QualificationSession
    from ds41f_mlx.serving.paired_checkpoint import PairedCheckpointAuthority
    transitions = []
    scheduler = object.__new__(ProductionScheduler)
    scheduler.waiting, scheduler.running, scheduler.requests = deque(), {}, {}
    scheduler.request_id_to_uid, scheduler.uid_to_request_id = {}, {}
    scheduler._pending_abort_ids, scheduler._completed = set(), []
    scheduler.batch_generator = None
    scheduler.total_prompt_tokens = scheduler.total_completion_tokens = 0
    scheduler.checkpoints = PairedCheckpointAuthority('identity', MX)
    monkeypatch.setattr(generate, 'BatchGenerator', lambda *a, **kw: SimpleNamespace())
    def start(rec, prepared, tokenizer, trace, *, checkpoint_capture, batch_generator_factory):
        rec.cache, rec.rings, rec.canonical = pair(200)
        checkpoint_capture(rec.cache, rec.rings, rec.canonical)
        bg = batch_generator_factory(None)
        rec.owner = SimpleNamespace(uid=17, _bg=bg)
        transitions.append('P5/native')
    def settle(rec, trace):
        transitions.append('settlement')
        if fail:
            raise RuntimeError('semantic settlement failed')
        assert not scheduler.checkpoints._payloads
        rec.cache, rec.rings, rec.canonical = pair(frontier)
        rec.owner = None
        trace['quiescence'], trace['canonical_frontier'] = {}, len(rec.canonical)
        trace['response'] = {'choices': [{'finish_reason': 'length' if frontier == TEXT_QUALIFIED_ENVELOPE else 'stop'}]}
    def advance(rec, trace):
        transitions.append('generation')
    def retire(rec):
        transitions.append('retirement')
        rec.owner = None
    scheduler.backend = SimpleNamespace(_start=start, _settle=settle, _advance_application=advance,
                                         _retire=retire, fatal_error=None)
    request = Request(request_id='ordinary', prompt=list(range(202)), sampling_params=SamplingParams())
    request.prompt_token_ids = list(range(202))
    request.execution = QualificationSession('ordinary')
    request.trace = dict(t0=perf_counter())
    request.prepared, request.recipe_tokenizer = None, None
    request.prompt_checkpoint, request.delivery_cursor = None, 0
    scheduler.requests[request.request_id] = request
    scheduler.waiting.append(request)
    scheduler.step()
    assert transitions == ['P5/native', 'generation']
    assert scheduler.batch_generator is request.execution.owner._bg
    assert scheduler.request_id_to_uid == {'ordinary': 17}
    assert not scheduler.checkpoints._payloads  # Backend/SSE is not authority.
    if cancel:
        scheduler.abort_request('ordinary')
    else:
        request.trace['application_terminal'] = True
    output = scheduler.step()
    assert transitions[-2:] == ['settlement', 'retirement']
    assert output.outputs[-1].finished
    assert bool(output.outputs[-1].error) == fail
    if frontier == TEXT_QUALIFIED_ENVELOPE:
        from omlx.request import RequestStatus
        assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
        assert output.outputs[-1].finish_reason == 'length'
    assert not scheduler.requests and not scheduler.running
    assert not scheduler.request_id_to_uid and not scheduler.uid_to_request_id
    assert scheduler.batch_generator is None and request.execution.owner is None
    assert bool(scheduler.checkpoints._payloads) != fail
    assert scheduler.last_settlement['cache_published'] != fail
    if not fail:
        assert len(scheduler.checkpoints.acquire(list(range(202)))[2]) == 200
        expected = frontier if frontier + 3 <= TEXT_QUALIFIED_ENVELOPE else 200
        assert len(scheduler.checkpoints.acquire(list(range(frontier + 2)))[2]) == expected
