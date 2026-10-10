"""Network boundary only; never inference or external tool effects."""
import asyncio
import sys
from types import SimpleNamespace

import httpx
import pytest

from ds41f_mlx import web


class Runtime:
    base_url = 'http://127.0.0.1:8000'
    def __init__(self, *args): pass
    def health(self): return {'status': 'ready'}


def test_explicit_private_ranges():
    for value in ('127.0.0.1', '::1', '10.3.4.5', '172.16.0.1', '172.31.255.255', '192.168.1.2', 'fd12::2', '::ffff:192.168.2.2'):
        assert web.private_address(value), value
    for value in ('0.0.0.0', '::', '8.8.8.8', '172.32.0.1', '169.254.1.2', '100.64.1.2', '192.0.2.1', '2001:db8::1', 'example.com'):
        assert not web.private_address(value), value


def test_lan_peer_host_origin_and_forwarding(monkeypatch):
    monkeypatch.setattr(web, 'RuntimeClient', Runtime)
    app = web.create_app(allow_private_lan=True)
    async def run():
        async def get(peer, url, headers=None):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=(peer, 1234)), base_url=url) as client:
                return await client.get('/', headers=headers)
        assert (await get('192.168.1.8', 'http://192.168.1.2:8080')).status_code == 200
        assert (await get('fd12::8', 'http://[fd12::2]:8080')).status_code == 200
        assert (await get('8.8.8.8', 'http://192.168.1.2', {'X-Forwarded-For': '192.168.1.8'})).status_code == 403
        assert (await get('192.168.1.8', 'http://evil.example')).status_code == 400
        assert (await get('192.168.1.8', 'http://8.8.8.8')).status_code == 400
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=('192.168.1.8', 1234)), base_url='http://192.168.1.2:8080') as client:
            assert (await client.post('/api/tools', json={}, headers={'Origin': 'http://evil.example'})).status_code == 403
            assert (await client.post('/api/tools', json={}, headers={'Origin': 'http://192.168.1.2:8080', 'Sec-Fetch-Site': 'cross-site'})).status_code == 403
            assert (await client.post('/api/tools', content='x')).status_code == 415
            # Same-origin reached route validation, not a relaxed tool certificate.
            assert (await client.post('/api/tools', json={}, headers={'Origin': 'http://192.168.1.2:8080'})).status_code == 400
    asyncio.run(run())


def test_default_stays_localhost(monkeypatch):
    monkeypatch.setattr(web, 'RuntimeClient', Runtime)
    app = web.create_app()
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://192.168.1.2') as client:
            assert (await client.get('/')).status_code == 400
    asyncio.run(run())


def test_bind_option_requires_consent_and_disables_proxy_trust(monkeypatch):
    calls = []
    monkeypatch.setattr(web, 'RuntimeClient', Runtime)
    from ds41f_mlx.operator_control import Control
    monkeypatch.setattr(Control, 'reserve_service', lambda *args: None)
    monkeypatch.setattr(Control, 'run', lambda self, app, **kwargs: calls.append(kwargs))
    with pytest.raises(SystemExit): web.main(['--host', '0.0.0.0'])
    with pytest.raises(SystemExit): web.main(['--host', '8.8.8.8', '--allow-private-lan'])
    web.main(['--host', '0.0.0.0', '--allow-private-lan'])
    assert calls[-1]['host'] == '0.0.0.0'
    assert calls[-1]['proxy_headers'] is False
    with pytest.raises(SystemExit): web.main(['--ssl-certfile', 'cert.pem'])
    web.main(['--host', '0.0.0.0', '--allow-private-lan', '--ssl-certfile', 'cert.pem', '--ssl-keyfile', 'key.pem'])
    assert calls[-1]['ssl_certfile'] == 'cert.pem'
    assert calls[-1]['ssl_keyfile'] == 'key.pem'
