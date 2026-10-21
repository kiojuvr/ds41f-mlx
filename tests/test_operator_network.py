"""Runtime-only operator launch, lifecycle and retired-surface regressions.

Former Web network tests were removed, not promoted to Runtime qualification.
"""
import asyncio
from pathlib import Path
import tomllib

import pytest
from ds41f_mlx import operator_tui as tui
from ds41f_mlx.operator_control import RUNTIME_PORT
from ds41f_mlx.operator_dashboard import render_screen, command_strip, Viewport


def test_runtime_launch_uses_admitted_interpreter_not_application_environment(monkeypatch):
    monkeypatch.setenv('DS41F_OPERATOR_RUNTIME_PYTHON', '/installed/bin/python')
    monkeypatch.setenv('DS41F_CHAT_BIND', 'lan')
    monkeypatch.setenv('DS41F_WEB_PORT', '8090')
    monkeypatch.setenv('DS41F_EXA_API_KEY', 'application-only-secret')
    monkeypatch.setenv('DS41F_DOCUMENT_VISION_URL', 'http://127.0.0.1:8001')
    monkeypatch.setenv('DS41F_CHECKPOINT', '/official/checkpoint')
    calls = []
    monkeypatch.setattr(tui.subprocess, 'Popen', lambda args, **kwargs: calls.append((args, kwargs)))
    tui.launch()
    args, options = calls[0]
    assert args == ['/installed/bin/python', '-I', '-m', 'ds41f_mlx.ops', 'start', '--profile', 'mtp-serving-v1']
    assert options['env']['DS41F_CHECKPOINT'] == '/official/checkpoint'
    assert not any(key in options['env'] for key in ('DS41F_CHAT_BIND', 'DS41F_WEB_PORT', 'DS41F_EXA_API_KEY', 'DS41F_DOCUMENT_VISION_URL', 'DS41F_OPERATOR_RUNTIME_PYTHON'))
    assert options['start_new_session']
    with pytest.raises(TypeError):
        tui.launch(chat=True)


def test_runtime_start_stop_and_conflict(monkeypatch):
    state = {'running': False, 'occupied': False}
    calls = []
    child = type('Child', (), {'poll': lambda self: None})()
    def call(port, path='/ds41f/status', method='GET'):
        assert port == RUNTIME_PORT
        calls.append((path, method))
        if not state['running']:
            raise ConnectionRefusedError()
        if path.endswith('shutdown'):
            state['running'] = False
        return {'state': 'READY'}
    def connect(*args, **kwargs):
        if not state['occupied']:
            raise ConnectionRefusedError()
        return type('Socket', (), {'__enter__': lambda self: self, '__exit__': lambda *args: None})()
    launches = []
    monkeypatch.setattr(tui, 'call', call)
    monkeypatch.setattr('socket.create_connection', connect)
    monkeypatch.setattr(tui, 'launch', lambda **kwargs: launches.append(kwargs) or child)
    op = tui.Operator()
    async def logs(): pass
    op.poll_logs = logs
    async def exercise():
        await op.lifecycle('start')
        assert launches == [{'default_reasoning_effort': op.default_reasoning_effort}]
        assert op.children == {RUNTIME_PORT: child}
        state['running'] = True
        await op.poll()
        assert op.snapshot['state'] == 'READY'
        await op.lifecycle('toggle')
        assert ('/ds41f/control/shutdown', 'POST') in calls
        assert op.snapshot is None
        state['occupied'] = True
        await op.lifecycle('start')
        assert len(launches) == 1 and 'CONFLICT' in op.message
        await op.poll()
        assert RUNTIME_PORT in op.conflicts
    asyncio.run(exercise())


def test_retired_product_not_packaged_or_rendered():
    root = Path(__file__).resolve().parents[1]
    assert not (root/'ds41f_mlx/web.py').exists()
    assert not list((root/'ds41f_mlx/web_static').glob('*'))
    metadata = tomllib.loads((root/'pyproject.toml').read_text())
    assert 'web-binary' not in metadata['project'].get('optional-dependencies', {})
    assert 'web_static/*' not in metadata['tool']['setuptools']['package-data']['ds41f_mlx']
    assert all('web' not in entry for entry in metadata['project']['scripts'].values())
    from ds41f_mlx import web_client
    assert not hasattr(web_client, 'StatefulToolChatClient')
    for mode in ('dashboard', 'settings', 'help', 'diagnostics'):
        text = '\n'.join(row.text for row in render_screen({'state': 'READY'}, Viewport(132, 57), mode=mode))
        assert 'Web Chat' not in text and 'Open Browser' not in text
        assert 'Web' not in command_strip(mode, 60)
        assert 'Web' not in command_strip(mode, 132)
