"""Focused publication/cancellation regressions; execution doubles are labeled."""
import asyncio
import json

import httpx
import pytest

from test_m54_authorized_cycles import setup as cycle_setup
from test_stateful_live_delivery import setup as delivery_setup
from test_web_application_boundary import CounterTool, RuntimeDouble
from ds41f_mlx.serving.response_reservation import reserve, observe_retry
from ds41f_mlx.serving.server import prepare_request


def test_predicted_terminal_anchor_forbids_proposal_and_verify(monkeypatch):
    gen, _, producer, _, m8, p, adapter = cycle_setup(monkeypatch, terminal=12)
    def forbidden(*args, **kwargs):
        pytest.fail('protected terminal attempted proposal/verify')
    monkeypatch.setattr(producer, 'propose', forbidden)
    monkeypatch.setattr(gen, 'speculative_cycle', forbidden)
    try:
        reports = adapter.advance(p.observe)
        assert [r.token for r in reports] == [12]
        assert gen.current_token_history() == m8.token_history
        assert p.semantic_terminal is not None
    finally:
        gen.close()


def test_retired_producer_cannot_be_mistaken_for_protected_short_ring(monkeypatch):
    gen, _, producer, _, m8, p, adapter = cycle_setup(monkeypatch)
    producer.retire()
    before = gen.current_token_history()
    try:
        with pytest.raises(RuntimeError, match='not a short-ring'):
            adapter.advance(p.observe)
        assert gen.current_token_history() == m8.token_history == before
        assert not p.tokens and adapter.failed
    finally:
        gen.close()


def test_inexact_preview_retires_derived_state_before_target_mutation(monkeypatch):
    gen, _, producer, _, m8, p, adapter = cycle_setup(monkeypatch)
    original = p.preview_tokens
    def inexact(ids):
        result = original(ids)
        result.mapping_exact = False
        return result
    p.preview_tokens = inexact
    before = gen.current_token_history()
    try:
        with pytest.raises(RuntimeError, match='inexact'):
            adapter.advance(p.observe)
        assert gen.current_token_history() == m8.token_history == before
        assert adapter.failed and not producer.active
        with pytest.raises(RuntimeError):
            adapter.advance(p.observe)
    finally:
        gen.close()


def test_json_task_cancellation_after_reservation_keeps_exact_outcome(monkeypatch):
    backend, tokenizer, body, _ = delivery_setup(monkeypatch, delay=.001)
    payload = json.loads(body)
    payload['stream'] = False
    body = json.dumps(payload).encode()
    prepared = prepare_request('chat_completions', body, tokenizer=tokenizer, recipe_path=backend.recipe_path)
    async def run():
        rec = await backend.create_stateful_session()
        slot = reserve(rec, 1, body, 'application/json')
        task = asyncio.create_task(backend.run_stateful_chat_turn(rec.session_id, prepared, tokenizer=tokenizer))
        await asyncio.sleep(.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not rec.busy and rec.request_count == 1 and slot.state == 'completed'
        assert json.loads(b''.join(slot.chunks)) == rec.last_turn['response_json']
        assert observe_retry(rec, 1, body) is slot
    try:
        asyncio.run(run())
    finally:
        backend.close()


def test_sse_byte_ceiling_releases_reservation_and_burns_retry(monkeypatch):
    backend, tokenizer, body, prepared = delivery_setup(monkeypatch)
    monkeypatch.setattr('ds41f_mlx.serving.response_reservation.MAX_RESPONSE_BYTES', 1)
    async def run():
        rec = await backend.create_stateful_session()
        slot = reserve(rec, 1, body, 'text/event-stream')
        stream = await backend.open_stateful_stream(rec.session_id, prepared, tokenizer=tokenizer, body=body)
        with pytest.raises(RuntimeError, match='byte ceiling'):
            await anext(stream)
        assert not rec.busy and not backend._lock.locked() and rec.active_stream is None
        assert slot.state == 'uncertain' and rec.recovery_state == 'unrecoverable'
        with pytest.raises(RuntimeError):
            observe_retry(rec, 1, body)
    try:
        asyncio.run(run())
    finally:
        backend.close()


def test_worker_startup_error_never_freezes_successful_eof(monkeypatch):
    backend, tokenizer, body, prepared = delivery_setup(monkeypatch)
    def fail_load():
        raise RuntimeError('injected startup failure')
    monkeypatch.setattr(backend, 'load', fail_load)
    async def run():
        rec = await backend.create_stateful_session()
        slot = reserve(rec, 1, body, 'text/event-stream')
        stream = await backend.open_stateful_stream(rec.session_id, prepared, tokenizer=tokenizer, body=body)
        with pytest.raises(RuntimeError, match='startup failure'):
            await anext(stream)
        assert not rec.busy and rec.m11 is None and rec.request_count == 0
        assert slot.state == 'uncertain' and not slot.chunks
        with pytest.raises(RuntimeError, match='never regenerate'):
            observe_retry(rec, 1, body)
    try:
        asyncio.run(run())
    finally:
        backend.close()


def test_effect_execution_then_transport_failure_never_reexecutes(monkeypatch):
    # Real canonical Web effect ledger and registry; runtime observation and
    # transport loss are doubles. Not an MTP tool-workflow qualification.
    from ds41f_mlx import web
    from ds41f_mlx.web_tools import ToolRegistry
    tool = CounterTool()
    monkeypatch.setattr(web, 'RuntimeClient', RuntimeDouble)
    monkeypatch.setattr(web, 'registry_from_env', lambda: ToolRegistry([tool]))
    app = web.create_app()
    class DropOnce(httpx.AsyncBaseTransport):
        def __init__(self):
            self.inner = httpx.ASGITransport(app=app)
            self.drop = True
        async def handle_async_request(self, request):
            response = await self.inner.handle_async_request(request)
            if self.drop:
                self.drop = False
                await response.aclose()
                raise httpx.ReadError('injected response delivery loss')
            return response
    async def run():
        async with httpx.AsyncClient(transport=DropOnce(), base_url='http://127.0.0.1') as client:
            payload = dict(session_id='s', request_count=1)
            with pytest.raises(httpx.ReadError):
                await client.post('/api/tools', json=payload)
            assert tool.executions == 1
            retried = await client.post('/api/tools', json=payload)
            observed = await client.get('/api/tools/result?session_id=s&request_count=1')
            assert retried.status_code == 200 and retried.json() == observed.json()['result']
            assert tool.executions == 1
    asyncio.run(run())
