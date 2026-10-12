"""Bounded resource policy, without replacing execution or effect authority."""
import asyncio
from types import SimpleNamespace as NS

import httpx

from ds41f_mlx import web
from ds41f_mlx.serving.production_sessions import ProductionSessions
from ds41f_mlx.web_client import RuntimeHTTPError
from ds41f_mlx.web_tools import ToolRegistry


def test_idle_policy_skips_active_queued_and_closing_and_renews_get():
    sessions = ProductionSessions(NS(fatal_error=None))
    idle = sessions.create('idle'); active = sessions.create('active'); closing = sessions.create('closing')
    for rec in (idle, active, closing): rec.touched_at = 0
    active.active_request_id = 'queued-or-active'
    closing.closing = True
    assert sessions.expired(now=1000) == ['idle']
    sessions.get('idle')
    assert sessions.expired(now=1000) == []
    assert active.busy and closing.busy


def test_idle_retirement_releases_only_expired_logical_pair():
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    async def check():
        backend = object.__new__(ProductionMTPBackend)
        backend.fatal_error = None
        backend.production_sessions = ProductionSessions(backend)
        backend.sessions = backend.production_sessions.records
        old = backend.production_sessions.create('old'); old.touched_at = 0
        live = backend.production_sessions.create('live')
        busy = backend.production_sessions.create('busy'); busy.touched_at = 0; busy.active_request_id = 'r1'
        released = []
        backend.scheduler = NS(release_session=released.append)
        async def call(fn, *args): return fn(*args)
        backend._call = call
        await backend.retire_idle_sessions()
        assert released == ['old'] and old.closed
        assert set(backend.sessions) == {live.session_id, busy.session_id}
        assert backend.production_sessions.issued == {'old', 'live', 'busy'}
    asyncio.run(check())


def test_web_orphan_release_defers_busy_and_clears_only_confirmed_retirement(monkeypatch):
    class Runtime:
        base_url = 'http://127.0.0.1:8000'
        def __init__(self, *_): self.records = {}; self.busy = set(); self.closed = []
        def health(self): return {'status': 'ready', 'profile': 'mtp-serving-v1'}
        def create_session(self, sid):
            self.records[sid] = {'id': sid, 'state': 'empty'}; return self.records[sid]
        def get_session(self, sid):
            if sid not in self.records: raise RuntimeHTTPError(404, 'absent')
            return self.records[sid]
        def close_session(self, sid):
            if sid in self.busy: raise RuntimeHTTPError(409, 'busy')
            if sid not in self.records: raise RuntimeHTTPError(404, 'absent')
            self.closed.append(sid); del self.records[sid]; return {'id': sid, 'state': 'closed'}
    monkeypatch.setattr(web, 'RuntimeClient', Runtime)
    monkeypatch.setattr(web, 'registry_from_env', lambda: ToolRegistry([]))
    app = web.create_app()
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://localhost') as client:
            for sid in ('a', 'b'):
                assert (await client.post('/api/session', json={'id': sid})).status_code == 200
            app.state.runtime.busy.add('b')
            response = await client.post('/api/release', json={'ids': ['a', 'b']})
            assert response.status_code == 200
            assert app.state.runtime.closed == ['a']
            assert set(app.state.owned_conversations) == {'b'}
            app.state.runtime.busy.clear()
            await app.state.retire_owned()
            assert app.state.runtime.closed == ['a', 'b']
            assert not app.state.owned_conversations
            assert (await client.post('/api/keepalive', json={'ids': ['b']})).json()['alive'] == []
    asyncio.run(check())
