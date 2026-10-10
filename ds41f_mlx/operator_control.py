"""Kernel-owned singleton socket and local control, inside the runtime process."""
from __future__ import annotations

import ipaddress
import json
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

RUNTIME_PORT = 18441
TUI_PORT = 18442
CHAT_PORT = 18443


def tui_guard():
    sock = socket.socket()
    try:
        sock.bind(('127.0.0.1', TUI_PORT))
        sock.listen(1)
        return sock
    except BaseException:
        sock.close()
        raise


class Control:
    def __init__(self, projection, port=RUNTIME_PORT):
        self.projection = projection
        self.shutdown_requested = False
        self.server = None
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, code, value):
                body = json.dumps(value, allow_nan=False).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)

            def allowed(self):
                # Kernel peer address, never Forwarded/X-Forwarded-For. Origin
                # and numeric authority checks also reject browser cross-origin actions.
                return (ipaddress.ip_address(self.client_address[0]).is_loopback
                        and self.headers.get('Origin') is None
                        and self.headers.get('Host') == f'127.0.0.1:{port}')

            def do_GET(self):
                if not self.allowed():
                    return self.reply(403, {'error': 'local_operator_only'})
                if self.path != '/ds41f/status':
                    return self.reply(404, {'error': 'not_found'})
                return self.reply(200, owner.projection.snapshot())

            def do_POST(self):
                if not self.allowed():
                    return self.reply(403, {'error': 'local_operator_only'})
                if self.path != '/ds41f/control/shutdown':
                    return self.reply(404, {'error': 'not_found'})
                if self.headers.get('Content-Length', '0') != '0' or self.headers.get('Transfer-Encoding'):
                    return self.reply(400, {'error': 'empty_body_required'})
                owner.shutdown_requested = True
                owner.projection.lifecycle('STOPPING')
                if owner.server is not None:
                    owner.server.should_exit = True
                return self.reply(202, {'state': 'STOPPING'})

        class LocalServer(ThreadingHTTPServer):
            # SO_REUSEADDR permits immediate restart after TCP TIME_WAIT;
            # no SO_REUSEPORT, so a second live listener still fails.
            allow_reuse_address = True
            daemon_threads = True

            def get_request(self):
                sock, addr = super().get_request()
                sock.settimeout(2)
                return sock, addr

        self.httpd = LocalServer(('127.0.0.1', port), Handler)
        self.thread = Thread(target=self.httpd.serve_forever, daemon=True, name='ds41f-control')
        self.thread.start()

    def reserve_service(self, host, port):
        # Reserve before model load, including conflict with a pre-upgrade server.
        sock = socket.socket(socket.AF_INET6 if ':' in host else socket.AF_INET)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            sock.listen(8)
            sock.setblocking(False)
            self.service_socket = sock
        except BaseException:
            sock.close()
            raise

    def run(self, app, **kwargs):
        import uvicorn
        self.server = uvicorn.Server(uvicorn.Config(app, **kwargs))
        self.server.should_exit = self.shutdown_requested
        self.server.run(sockets=[self.service_socket] if hasattr(self, 'service_socket') else None)

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()
        if hasattr(self, 'service_socket'):
            self.service_socket.close()
