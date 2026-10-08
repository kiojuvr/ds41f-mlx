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


def test_lost_awaited_sse_worker_result_keeps_every_canonical_delta(monkeypatch):
    import threading
    from ds41f_mlx.runtime.live_turn import LiveRecipeTurn
    backend, tokenizer, body, prepared = delivery_setup(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    original = LiveRecipeTurn.advance
    def held(cursor):
        batch = original(cursor)
        if (not entered.is_set() and any(c.get('delta',{}).get('content')
            for e in batch for c in e.get('choices',[]))):
            entered.set()
            assert release.wait(5)
        return batch
    monkeypatch.setattr(LiveRecipeTurn,'advance',held)
    async def run():
        rec = await backend.create_stateful_session()
        slot = reserve(rec,1,body,'text/event-stream')
        stream = await backend.open_stateful_stream(rec.session_id,prepared,tokenizer=tokenizer,body=body)
        await anext(stream)  # ready
        task = asyncio.create_task(anext(stream))
        for _ in range(1000):
            if entered.is_set(): break
            await asyncio.sleep(.001)
        assert entered.is_set()
        task.cancel(); release.set()
        with pytest.raises(asyncio.CancelledError): await task
        assert slot.state == 'completed' and not rec.busy
        events = [json.loads(line[6:]) for line in b''.join(slot.chunks).splitlines()
                  if line.startswith(b'data: ') and line != b'data: [DONE]']
        text = ''.join(c.get('delta',{}).get('content','') or '' for e in events for c in e.get('choices',[]))
        assert text == rec.last_turn['response_json']['choices'][0]['message']['content']
        assert text and observe_retry(rec,1,body) is slot
    try: asyncio.run(run())
    finally: release.set(); backend.close()


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


@pytest.mark.parametrize('streaming', [False, True])
def test_direct_publication_body_mismatch_rejected_before_mutation(monkeypatch, streaming):
    backend, tokenizer, body, _ = delivery_setup(monkeypatch)
    payload = json.loads(body); payload['stream'] = streaming
    body = json.dumps(payload).encode()
    prepared = prepare_request('chat_completions',body,tokenizer=tokenizer,recipe_path=backend.recipe_path)
    async def run():
        rec = await backend.create_stateful_session()
        slot = reserve(rec,1,body,'text/event-stream' if streaming else 'application/json')
        with pytest.raises(ValueError, match='request-body identity'):
            if streaming:
                await backend.open_stateful_stream(rec.session_id,prepared,tokenizer=tokenizer,body=body+b' ')
            else:
                await backend.run_stateful_chat_turn(rec.session_id,prepared,tokenizer=tokenizer,body=body+b' ')
        assert rec.m11 is None and rec.request_count == 0 and not rec.busy
        assert slot.state == 'active'
        slot.complete()
        with pytest.raises(ValueError, match='active exact-byte'):
            await backend.run_stateful_chat_turn(rec.session_id,prepared,tokenizer=tokenizer,body=body)
        assert rec.m11 is None
    try:
        asyncio.run(run())
    finally:
        backend.close()


def test_json_publication_failure_burns_reservation_and_reentry(monkeypatch):
    backend, tokenizer, body, _ = delivery_setup(monkeypatch)
    payload = json.loads(body); payload['stream'] = False
    body = json.dumps(payload).encode()
    prepared = prepare_request('chat_completions',body,tokenizer=tokenizer,recipe_path=backend.recipe_path)
    def fail(*args, **kwargs): raise RuntimeError('injected application publication failure')
    monkeypatch.setattr('ds41f_mlx.serving.recipe_publication.publish_recipe_turn',fail)
    async def run():
        rec = await backend.create_stateful_session()
        slot = reserve(rec,1,body,'application/json')
        with pytest.raises(RuntimeError, match='publication failure'):
            await backend.run_stateful_chat_turn(rec.session_id,prepared,tokenizer=tokenizer)
        assert rec.m11.m8.closed and not rec.m11.m8.live_cache
        assert rec.request_count == 0 and rec.last_turn is None and not rec.busy
        assert rec.recovery_state == 'unrecoverable' and slot.state == 'uncertain'
        with pytest.raises(RuntimeError, match='never regenerate'):
            observe_retry(rec,1,body)
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
