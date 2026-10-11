"""Consumer disconnects are not operator/runtime failures."""
import json
import socket
import struct
from threading import Event
from types import SimpleNamespace
from urllib.request import urlopen

import pytest

from ds41f_mlx.observability import Projection
from ds41f_mlx.operator_control import Control
from ds41f_mlx.runtime_logging import History
from test_operator_observability import free_port


@pytest.mark.parametrize('stage', ['headers', 'body'])
@pytest.mark.parametrize('error', [BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError])
def test_reply_tolerates_only_transport_loss(stage, error):
    # Exercise both write sites deterministically; TCP may reject either one.
    control = Control(Projection(), free_port())
    try:
        handler = object.__new__(control.httpd.RequestHandlerClass)
        def fail(*args):
            raise error('consumer disconnected')
        handler.send_response = lambda *args: None
        handler.send_header = lambda *args: None
        handler.end_headers = fail if stage == 'headers' else lambda: None
        handler.wfile = SimpleNamespace(write=fail)
        handler.close_connection = False
        handler.reply(200, {'state': 'READY'})
        assert handler.close_connection
        with pytest.raises(ValueError):
            handler.reply(200, {'invalid': float('nan')})
        handler.end_headers = lambda: None
        handler.wfile = SimpleNamespace(write=lambda body: (_ for _ in ()).throw(OSError('unexpected I/O failure')))
        with pytest.raises(OSError, match='unexpected I/O'):
            handler.reply(200, {})
    finally:
        control.close()


@pytest.mark.parametrize('path,method', [('/ds41f/status', 'GET'), ('/ds41f/logs', 'GET'),
                                       ('/ds41f/control/shutdown', 'POST')])
def test_reset_consumer_leaves_control_healthy_and_shutdown_committed(path, method):
    entered, release, completed = Event(), Event(), Event()
    projection, history = Projection(), History()
    def gate(fn):
        def blocked(*args, **kwargs):
            entered.set()
            assert release.wait(3)
            return fn(*args, **kwargs)
        return blocked
    if path == '/ds41f/status':
        projection.snapshot = gate(projection.snapshot)
    elif path == '/ds41f/logs':
        history.retrieve = gate(history.retrieve)
    else:
        projection.lifecycle = gate(projection.lifecycle)
    port = free_port()
    control = Control(projection, port, logs=history)
    control.server = SimpleNamespace(should_exit=False)
    errors = []
    control.httpd.handle_error = lambda *args: errors.append(args)
    finish = control.httpd.finish_request
    def finished(*args):
        try:
            return finish(*args)
        finally:
            completed.set()
    control.httpd.finish_request = finished
    client = socket.create_connection(('127.0.0.1', port), timeout=2)
    try:
        client.sendall(f'{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Length: 0\r\n\r\n'.encode())
        assert entered.wait(3)
        # Reset before the server sends headers/body, rather than relying on timing.
        client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
        client.close()
        release.set()
        assert completed.wait(3)
        with urlopen(f'http://127.0.0.1:{port}/ds41f/status', timeout=3) as response:
            snapshot = json.load(response)
        assert not errors  # socketserver would otherwise print the raw traceback
        if method == 'POST':
            assert control.shutdown_requested and control.server.should_exit
            assert snapshot['state'] == 'STOPPING'
    finally:
        client.close()
        release.set()
        control.close()
