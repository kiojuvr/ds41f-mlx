import asyncio
import gc
import weakref

import pytest

from test_stateful_live_delivery import setup


def test_completed_parser_and_request_release_without_cyclic_gc(monkeypatch):
    backend, tokenizer, body, prepared = setup(monkeypatch)
    async def run():
        rec = await backend.create_stateful_session()
        stream = await backend.open_stateful_stream(rec.session_id, prepared, tokenizer=tokenizer, body=body)
        await anext(stream)
        cursor = weakref.ref(stream.cursor)
        gc.disable()
        try:
            await stream.aclose()
            assert stream.cursor is None and stream.prepared is None and stream.body is None
            assert cursor() is None
        finally: gc.enable()
    asyncio.run(run()); backend.close()


def test_burnt_state_failure_releases_generation_even_if_idle_extraction_fails(monkeypatch):
    backend, tokenizer, body, prepared = setup(monkeypatch)
    async def run():
        rec = await backend.create_stateful_session()
        stream = await backend.open_stateful_stream(rec.session_id, prepared, tokenizer=tokenizer, body=body)
        await anext(stream)
        closed = []
        class Burnt:
            def close(self): closed.append('policy/resources retired')
        rec.m11.m8.generation = Burnt()
        def fail(): raise RuntimeError('protected failure')
        monkeypatch.setattr(rec.m11.m8, 'next_token', fail)
        monkeypatch.setattr(rec.m11.m8, 'close', fail)
        with pytest.raises(RuntimeError, match='protected failure'): await anext(stream)
        assert closed == ['policy/resources retired']
        assert rec.m11.m8.generation is None and rec.m11.m8.closed
        assert not rec.busy and not backend._lock.locked()
        assert rec.recovery_state == 'unrecoverable'
        assert rec.last_turn is None  # no invented/cancel-finalized response after a fault
        assert stream.cursor is None
    asyncio.run(run())
    backend.sessions.clear(); backend.close()
