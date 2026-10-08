"""Standard HTTP exact-byte reservation integration; execution doubles labeled."""
import asyncio
import json

import httpx
import pytest

from test_stateful_live_delivery import setup
from ds41f_mlx.serving.server import create_app
from ds41f_mlx.serving.response_reservation import observe_retry, reserve


@pytest.mark.parametrize('streaming', [False, True])
def test_standard_http_retry_returns_identical_bytes_without_generation(monkeypatch, streaming):
    backend, tokenizer, body, prepared = setup(monkeypatch)
    payload = json.loads(body)
    payload['stream'] = streaming
    body = json.dumps(payload).encode()
    app = create_app(backend=backend, runtime_config=backend.runtime_config)
    async def run():
        rec = await backend.create_stateful_session()
        url = f'/v1/sessions/{rec.session_id}/chat/completions'
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            headers = {'X-DS41F-Request-Sequence': '1', 'Content-Type': 'application/json'}
            first = await client.post(url, content=body, headers=headers)
            assert first.status_code == 200, first.text
            consumed, count = rec.m11.m8.consumed, rec.request_count
            second = await client.post(url, content=body, headers=headers)
            assert second.status_code == 200 and second.content == first.content
            assert second.headers['X-DS41F-Response-Replay'] == 'exact-bytes'
            assert rec.m11.m8.consumed == consumed and rec.request_count == count == 1
            # Semantic equivalence is NOT byte identity.
            changed = await client.post(url, content=body + b' ', headers=headers)
            assert changed.status_code == 400
            stale = await client.post(url, content=body, headers={**headers, 'X-DS41F-Request-Sequence':'3'})
            assert stale.status_code == 400
            unfenced = await client.post(url, content=body)
            assert unfenced.status_code == 400
            assert rec.request_count == 1
    try:
        asyncio.run(run())
    finally:
        backend.close()


def test_partial_sse_close_freezes_response_and_retry_does_not_decode(monkeypatch):
    backend, tokenizer, body, prepared = setup(monkeypatch)
    async def run():
        rec = await backend.create_stateful_session()
        slot = reserve(rec, 1, body, 'text/event-stream')
        stream = await backend.open_stateful_stream(rec.session_id, prepared, tokenizer=tokenizer, body=body)
        prefix = (await anext(stream)).encode()
        prefix += (await anext(stream)).encode()
        consumed = rec.m11.m8.consumed
        await stream.aclose()
        assert slot.state == 'completed', rec.to_json()
        assert b''.join(slot.chunks).startswith(prefix)
        assert b''.join(slot.chunks).endswith(b'data: [DONE]\n\n')
        assert rec.request_count == 1 and not rec.busy
        assert observe_retry(rec, 1, body) is slot
        assert rec.m11.m8.consumed == consumed
        assert rec.last_turn['cancelled'] and rec.last_turn['reconstruction']['representable']
    try:
        asyncio.run(run())
    finally:
        backend.close()


@pytest.mark.parametrize('streaming', [False, True])
def test_direct_development_entry_cannot_bypass_reservation(monkeypatch, streaming):
    from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend
    control, tokenizer, body, prepared = setup(monkeypatch)
    prepared.resolved_max_tokens = 32
    backend = DeepSeekRecipeRuntimeBackend(runtime_config=control.runtime_config,
        execution_strategy='first-party-mtp-development')
    monkeypatch.setattr(backend, 'load', lambda: pytest.fail('unreserved entry loaded model'))
    async def run():
        rec = await backend.create_stateful_session()
        with pytest.raises(ValueError, match='active exact-byte'):
            if streaming:
                await backend.open_stateful_stream(rec.session_id, prepared, tokenizer=tokenizer, body=body)
            else:
                await backend.run_stateful_chat_turn(rec.session_id, prepared, tokenizer=tokenizer)
        assert not rec.busy and rec.m11 is None and rec.request_count == 0
    try:
        asyncio.run(run())
    finally:
        backend.close()
        control.close()


def test_development_budget_rejects_before_conversion_or_image_preparation(monkeypatch):
    from ds41f_mlx.config import load_runtime_config
    from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend
    backend = DeepSeekRecipeRuntimeBackend(runtime_config=load_runtime_config(),
        execution_strategy='first-party-mtp-development')
    app = create_app(backend=backend, runtime_config=backend.runtime_config)
    monkeypatch.setattr('ds41f_mlx.serving.server.prepare_request',
        lambda *args, **kwargs: pytest.fail('unqualified budget prepared a request'))
    async def run():
        rec = await backend.create_stateful_session()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            response = await client.post(f'/v1/sessions/{rec.session_id}/budget', content=b'invalid: must not convert')
            assert response.status_code == 400
            assert rec.m11 is None and not rec.busy and rec.response_reservation is None
    try:
        asyncio.run(run())
    finally:
        backend.close()


def test_unqualified_reasoning_rejected_before_execution(monkeypatch):
    from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend
    control, _, body, _ = setup(monkeypatch)
    backend = DeepSeekRecipeRuntimeBackend(runtime_config=control.runtime_config,
        execution_strategy='first-party-mtp-development')
    payload = json.loads(body)
    payload['reasoning_effort'] = 'high'
    try:
        with pytest.raises(ValueError, match='disabled reasoning'):
            backend.validate_development_body(json.dumps(payload).encode())
        assert backend._runtime is None and not backend.sessions
    finally:
        backend.close()
        control.close()


def test_busy_get_never_reads_half_adopted_execution_history():
    from types import SimpleNamespace
    from ds41f_mlx.serving.deepseek_recipe_backend import StatefulSessionRecord
    def forbidden():
        pytest.fail('busy GET read worker-owned execution history')
    rec = StatefulSessionRecord('s', busy=True, m11=SimpleNamespace(diagnostics=forbidden))
    assert rec.to_json()['diagnostics'] is None
    assert rec.to_json()['state'] == 'busy'


def test_certificate_hashing_uses_one_prefix_snapshot(monkeypatch):
    from hashlib import sha256 as real_sha256
    from ds41f_mlx.serving.response_reservation import ResponseReservation
    slot = ResponseReservation(1, b'request', 'text/event-stream')
    slot.append(b'head')
    calls = [0]
    def interleaved_hash(data):
        calls[0] += 1
        if calls[0] == 1:
            slot.append(b'tail')
            slot.complete()
        return real_sha256(data)
    monkeypatch.setattr('ds41f_mlx.serving.response_reservation.sha256', interleaved_hash)
    cert = slot.certificate()
    assert cert['outcome_state'] == 'active' and cert['response_bytes'] == 4
    assert cert['response_sha256'] == real_sha256(b'head').hexdigest()
    assert slot.state == 'completed' and slot.size == 8


def test_active_and_uncertain_reservations_fail_closed(monkeypatch):
    backend, _, body, _ = setup(monkeypatch)
    async def run():
        rec = await backend.create_stateful_session()
        slot = reserve(rec, 1, body, 'application/json')
        with pytest.raises(RuntimeError, match='never regenerate'):
            observe_retry(rec, 1, body)
        slot.burn()
        with pytest.raises(RuntimeError, match='never regenerate'):
            observe_retry(rec, 1, body)
        with pytest.raises(RuntimeError, match='not coherent'):
            reserve(rec, 2, body, 'application/json')
    try:
        asyncio.run(run())
    finally:
        backend.close()
