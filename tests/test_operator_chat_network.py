"""TUI Chat LAN launch is explicit, scoped and has usable connection URLs."""
import pytest
from ds41f_mlx import operator_tui as tui, operator_network as network
from ds41f_mlx.operator_dashboard import render_screen, Viewport


@pytest.mark.parametrize('bind,host,lan', [(None, '0.0.0.0', True), ('lan', '0.0.0.0', True), ('loopback', '127.0.0.1', False)])
def test_chat_launch_bind(monkeypatch, bind, host, lan):
    monkeypatch.delenv('DS41F_CHAT_BIND', raising=False)
    monkeypatch.delenv('DS41F_WEB_PORT', raising=False)
    if bind:
        monkeypatch.setenv('DS41F_CHAT_BIND', bind)
    calls = []
    monkeypatch.setattr(tui.subprocess, 'Popen', lambda args, **kw: calls.append(args))
    tui.launch(chat=True)
    args = calls[0]
    assert args[args.index('--host') + 1] == host
    assert args[args.index('--port') + 1] == '8080'
    assert ('--allow-private-lan' in args) is lan


def test_restart_preserves_endpoint_and_runtime(monkeypatch):
    calls = []
    monkeypatch.setenv('DS41F_CHAT_BIND', 'lan')
    monkeypatch.setattr(tui.subprocess, 'Popen', lambda args, **kw: calls.append(args))
    tui.launch(True, dict(host='127.0.0.1', port=8090, allow_private_lan=False, runtime_url='http://127.0.0.1:8001'))
    assert '--allow-private-lan' not in calls[0]
    assert calls[0][-2:] == ['--runtime-url', 'http://127.0.0.1:8001']
    tui.launch(True, dict(host='0.0.0.0', port=8090))
    assert '--allow-private-lan' in calls[1]
    tui.launch()
    assert '--host' not in calls[2] and '--allow-private-lan' not in calls[2]
    monkeypatch.setenv('DS41F_CHAT_BIND', 'invalid')
    with pytest.raises(ValueError, match='lan or loopback'):
        tui.launch(True)
    with pytest.raises(ValueError, match='HTTP downgrade'):
        tui.launch(True, dict(host='0.0.0.0', port=8080, scheme='https'))
    assert len(calls) == 3


def test_connection_urls_and_dashboard(monkeypatch):
    monkeypatch.setattr(network, 'lan_addresses', lambda: ['192.168.68.54'])
    endpoint = network.chat_endpoint('0.0.0.0', 8080, allow_private_lan=True)
    assert endpoint['local_url'] == 'http://127.0.0.1:8080'
    assert endpoint['lan_urls'] == ['http://192.168.68.54:8080']
    rows = render_screen({'state': 'READY'}, Viewport(132, 57), chat={'state': 'READY', 'endpoint': endpoint})
    text = '\n'.join(row.text for row in rows)
    assert 'Listen 0.0.0.0:8080' in text
    assert 'Local http://127.0.0.1:8080' in text
    assert 'Chat LAN http://192.168.68.54:8080' in text
    assert 'trusted local network only' in text
    local = network.chat_endpoint('127.0.0.1', 8080)
    assert local['lan_urls'] == []
    assert network.chat_endpoint('::1', 8090, tls=True)['local_url'] == 'https://[::1]:8090'


def test_route_discovery_never_sends_packets_and_filters_public(monkeypatch):
    monkeypatch.setattr(network.socket, 'gethostbyname_ex', lambda name: (name, [], ['127.0.0.1', '8.8.8.8', '192.168.68.54']))
    class Route:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def connect(self, address): assert address == ('192.168.0.1', 9)
        def getsockname(self): return ('192.168.68.54', 5000)
    monkeypatch.setattr(network.socket, 'socket', lambda *args: Route())
    assert network.lan_addresses() == ['192.168.68.54']
    monkeypatch.setattr(network.socket, 'gethostbyname_ex', lambda name: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(network.socket, 'socket', lambda *args: (_ for _ in ()).throw(OSError()))
    assert network.lan_addresses() == []
