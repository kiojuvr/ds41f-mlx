"""Content-free projection and disposable lifecycle/control regressions."""
import json
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from ds41f_mlx.observability import Projection
from ds41f_mlx.operator_control import Control, tui_guard
from ds41f_mlx.operator_tui import render


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def test_projection_is_copy_and_content_free():
    p = Projection()
    p.lifecycle('READY')
    p.begin('opaque-1')
    p.phase('opaque-1', 'ENCODING')
    time.sleep(.002)
    p.phase('opaque-1', 'CACHE_LOOKUP')
    p.metrics('opaque-1', dict(prompt='SECRET', response={'text': 'SECRET'},
        canonical_generated=[123], recipe_encode_s=.01, cached_tokens=90,
        generated=10, decode_s=.5, suffix_append_s=.01, total_ttft_s=.1,
        prompt_replay=0, full_cache_repack=0), 100)
    s = p.snapshot()
    assert 'SECRET' not in json.dumps(s)
    assert s['current_request']['completed_phase_durations']['encoding_ms'] > 0
    s['current_request']['metrics']['cached_tokens'] = 0
    assert p.snapshot()['current_request']['metrics']['cached_tokens'] == 90
    p.finish('opaque-1', 'tool_calls')
    assert p.snapshot()['active_requests'] == 0
    text = '\n'.join(render(p.snapshot()))
    assert 'IDLE' in text and 'WAITING FOR CLIENT' in text
    assert '90.0%' in text and '20.0 tok/s' in text
    assert '100.0 ms' in text
    from ds41f_mlx.operator_dashboard import Dashboard
    assert Dashboard(p.snapshot()).new == 10
    assert Dashboard(p.snapshot()).work == 9


def test_queue_age_and_local_elapsed():
    p = Projection()
    p.begin('a'); p.phase('a', 'DECODING')
    p.begin('b'); p.phase('b', 'QUEUED', queue_reason='http_preparation')
    s = p.snapshot()
    assert s['active_requests'] == s['queued_requests'] == 1
    assert s['current_request']['request_id'] == 'a'
    t0 = s['sampled_mono']
    before, after = '\n'.join(render(s, t0)), '\n'.join(render(s, t0+5))
    assert before != after
    assert '5.00 s' in after and 'QUEUE' in after
    p.finish('a', 'stop')
    assert p.snapshot()['current_request']['current_phase'] == 'QUEUED'


def test_last_settled_mtp_and_alignment():
    p = Projection(); p.begin('a')
    p.metrics('a', dict(mtp_stats={'depth_drafted': [5, 5], 'depth_accepted': [4, 4]},
        canonical_frontier=42, target_offsets=[42]*40, dspark_offsets=[42]*5))
    p.finish('a', 'stop')
    text = '\n'.join(render(p.snapshot()))
    assert '80.0%' in text and '✓ ALIGNED' in text


def test_control_local_boundary_and_graceful_callback():
    p = Projection(); port = free_port(); c = Control(p, port)
    try:
        with pytest.raises(OSError):
            Control(Projection(), port)
        base = f'http://127.0.0.1:{port}'
        with urlopen(base+'/ds41f/status') as r:
            assert json.load(r)['schema'] == 'ds41f.live.v1'
        for headers in ({'Origin': 'http://hostile.example'}, {'Host': 'hostile.example'},
                        {'Origin': 'null', 'X-Forwarded-For': '127.0.0.1'}):
            with pytest.raises(HTTPError) as exc:
                urlopen(Request(base+'/ds41f/control/shutdown', data=b'', headers=headers))
            assert exc.value.code == 403
        c.server = SimpleNamespace(should_exit=False)
        with urlopen(Request(base+'/ds41f/control/shutdown', data=b'')) as r:
            assert r.status == 202
        assert c.server.should_exit and c.shutdown_requested
        assert p.snapshot()['state'] == 'STOPPING'
    finally:
        c.close()
    replacement = Control(Projection(), port)
    replacement.close()


def test_sigkill_releases_guard_without_cleanup():
    port = free_port()
    source = f'''from ds41f_mlx.operator_control import Control
from ds41f_mlx.observability import Projection
import time
c=Control(Projection(), {port})
print('READY', flush=True)
time.sleep(60)
'''
    child = subprocess.Popen([sys.executable, '-c', source], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'READY'
        child.kill(); child.wait(timeout=5)
        c = Control(Projection(), port); c.close()
    finally:
        if child.poll() is None:
            child.kill(); child.wait()


def test_tui_kernel_singleton():
    guard = tui_guard()
    try:
        with pytest.raises(OSError):
            tui_guard()
    finally:
        guard.close()
    tui_guard().close()


def test_cli_no_args_dispatches_tui_without_changing_headless(monkeypatch):
    from ds41f_mlx import ops, operator_tui
    monkeypatch.setattr(operator_tui, 'main', lambda: 17)
    assert ops.main([]) == 17
    with pytest.raises(SystemExit) as exc:
        ops.main(['--help'])
    assert exc.value.code == 0


def test_update_failure_is_best_effort():
    class BrokenLock:
        def __enter__(self):
            raise RuntimeError('diagnostic failure')
        def __exit__(self, *args):
            pass
    p = Projection(); p._lock = BrokenLock()
    p.begin('a'); p.phase('a', 'DECODING'); p.metrics('a', {}, 10)
    p.finish('a'); p.lifecycle('READY'); p.cache_summary({'hits': 1})


def test_restart_composes_shutdown_disappearance_and_canonical_start(monkeypatch):
    import asyncio
    from ds41f_mlx import operator_tui as tui
    endpoint = dict(host='0.0.0.0', port=8001)
    events = []
    alive = True
    def call(port, path='/ds41f/status', method='GET'):
        nonlocal alive
        events.append(method)
        if method == 'POST':
            alive = False
            return {'state': 'STOPPING'}
        if not alive:
            raise ConnectionRefusedError()
        return {'endpoint': endpoint}
    monkeypatch.setattr(tui, 'call', call)
    monkeypatch.setattr(socket, 'create_connection', lambda *a, **kw: (_ for _ in ()).throw(ConnectionRefusedError()))
    monkeypatch.setattr(tui, 'launch', lambda chat, cfg: events.append(('launch', chat, cfg)) or SimpleNamespace(poll=lambda: None))
    operator = tui.Operator()
    async def poll():
        pass
    monkeypatch.setattr(operator, 'poll', poll)
    asyncio.run(operator.lifecycle('restart'))
    assert events == ['GET', 'POST', 'GET', ('launch', False, endpoint)]


def test_canonical_launcher_registers_load_on_actual_fastapi_router(monkeypatch):
    import asyncio
    pytest.importorskip('omlx')
    fastapi = pytest.importorskip('fastapi')
    from ds41f_mlx import serve, mtp_identity
    from ds41f_mlx.serving import server, production_mtp
    events = []
    cfg = SimpleNamespace(host='127.0.0.1', port=8000, apply_environment=lambda: None)
    monkeypatch.setattr(mtp_identity, 'config', lambda *args, **kwargs: cfg)
    monkeypatch.setattr(mtp_identity, 'inspect', lambda cfg: {'identity_sha256': 'fixture'})
    monkeypatch.setattr(server, 'create_app', lambda **kwargs: fastapi.FastAPI())
    class Backend:
        context_tokens = 128
        def __init__(self, **kwargs):
            pass
        def load(self):
            events.append('load')
        async def _call(self, fn):
            return fn()
        async def shutdown_serving(self):
            events.append('retire')
        def close(self):
            events.append('close')
    monkeypatch.setattr(production_mtp, 'ProductionMTPBackend', Backend)
    class ControlFixture:
        shutdown_requested = False
        def reserve_service(self, host, port):
            events.append('reserve')
        def run(self, app, **kwargs):
            # This checks the real framework's registration API, not a fake app.
            async def lifespan():
                async with app.router.lifespan_context(app):
                    pass
            asyncio.run(lifespan())
    projection = Projection()
    assert serve._main(['--profile', 'mtp-serving-v1'], control=ControlFixture(), projection=projection) == 0
    assert events == ['reserve', 'load', 'retire', 'close']
    assert projection.snapshot()['state'] == 'READY'


def test_service_reserved_before_loading():
    port = free_port()
    c = Control(Projection(), free_port())
    try:
        c.reserve_service('0.0.0.0', port)
        with socket.socket() as s, pytest.raises(OSError):
            s.bind(('127.0.0.1', port))
    finally:
        c.close()
