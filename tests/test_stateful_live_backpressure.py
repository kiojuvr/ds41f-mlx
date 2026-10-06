"""Controlled saturated ASGI send; complements the real slow TCP reader."""
import asyncio

from ds41f_mlx.serving.server import InferenceStreamingResponse
from test_stateful_live_delivery import setup


def test_blocked_send_stops_native_demand_and_explicit_stop_still_settles(monkeypatch):
    backend, tokenizer, body, prepared = setup(monkeypatch)
    async def run():
        rec = await backend.create_stateful_session()
        stream = await backend.open_stateful_stream(rec.session_id, prepared, tokenizer=tokenizer, body=body)
        blocked, release = asyncio.Event(), asyncio.Event()
        async def send(message):
            if message['type'] == 'http.response.body' and message.get('body'):
                blocked.set()
                await release.wait()
        async def receive():
            await asyncio.Event().wait()
        scope = {'type':'http', 'asgi':{'spec_version':'2.4'}}
        response = asyncio.create_task(InferenceStreamingResponse(stream)(scope, receive, send))
        await blocked.wait()
        count = rec.m11.m8.consumed
        await asyncio.sleep(.1)
        assert rec.m11.m8.consumed == count
        await backend.cancel_stateful_stream(rec.session_id, stream.request_id)
        assert rec.m11.m8.generation is None and not rec.busy and not backend._lock.locked()
        assert rec.last_turn['cancelled']
        assert stream.pending == __import__('collections').deque()
        release.set()
        await response
    asyncio.run(run()); backend.close()


def test_header_send_failure_closes_unstarted_response_reservation(monkeypatch):
    backend, tokenizer, body, prepared = setup(monkeypatch)
    async def run():
        rec = await backend.create_stateful_session()
        stream = await backend.open_stateful_stream(rec.session_id, prepared, tokenizer=tokenizer, body=body)
        async def send(message): raise OSError('socket lost before headers')
        async def receive(): await asyncio.Event().wait()
        try:
            await InferenceStreamingResponse(stream)({'type':'http','asgi':{'spec_version':'2.4'}}, receive, send)
        except Exception: pass  # Starlette translates the expected socket failure
        assert not rec.busy and rec.active_stream is None and not backend._lock.locked()
        assert rec.m11 is None and stream.prepared is None
    asyncio.run(run()); backend.close()
