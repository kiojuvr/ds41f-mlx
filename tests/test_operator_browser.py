import asyncio
import subprocess

import pytest

from ds41f_mlx import operator_tui as tui


@pytest.mark.parametrize('code', [0, 1])
def test_macos_browser_uses_detached_launchservices(monkeypatch, code):
    calls = []
    class Process:
        async def wait(self):
            return code
    async def create(*args, **kwargs):
        calls.append((args, kwargs))
        return Process()
    monkeypatch.setattr(tui.sys, 'platform', 'darwin')
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', create)
    async def check():
        operator = tui.Operator()
        operator.chat_snapshot = {'endpoint': {'local_url': 'https://127.0.0.1:8090'}}
        await operator.open_browser()
        assert calls == [(('/usr/bin/open', 'https://127.0.0.1:8090'), {
            'stdin': subprocess.DEVNULL, 'stdout': subprocess.DEVNULL,
            'stderr': subprocess.DEVNULL, 'start_new_session': True})]
        assert ('failed' if code else 'requested') in operator.message
        assert not hasattr(operator, 'conversation')
    asyncio.run(check())


def test_browser_timeout_kills_only_helper_and_reports_url(monkeypatch):
    events = []
    class Process:
        async def wait(self):
            events.append('wait')
            return 0
        def kill(self):
            events.append('kill')
    async def create(*args, **kwargs):
        return Process()
    async def timeout(awaitable, seconds):
        assert seconds == 10
        awaitable.close()
        raise asyncio.TimeoutError
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', create)
    monkeypatch.setattr(asyncio, 'wait_for', timeout)
    async def check():
        operator = tui.Operator()
        operator.chat_snapshot = {'endpoint': {'port': 8090}}
        await operator.open_browser()
        assert events == ['kill', 'wait']
        assert 'timed out' in operator.message
        assert 'open manually: http://127.0.0.1:8090' in operator.message
    asyncio.run(check())
