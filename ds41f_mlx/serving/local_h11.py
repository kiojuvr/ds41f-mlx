"""Pinned uvicorn 0.54.0 adapter: finite local socket/header authority.

Request-concurrency limits alone do not bound connections waiting for headers.
This private adapter supplies that missing bound; it owns no model lifecycle.
"""
import socket
from uvicorn.protocols.http.h11_impl import H11Protocol
from ds41f_mlx.mtp_profile import LIMITS


class LocalH11Protocol(H11Protocol):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._admitted = False
        self._header_deadline = None

    def _arm_header_deadline(self):
        if self._header_deadline is None:
            self._header_deadline = self.loop.call_later(LIMITS['body_timeout_s'], self.transport.close)

    def _cancel_header_deadline(self):
        if self._header_deadline is not None:
            self._header_deadline.cancel()
            self._header_deadline = None

    def connection_made(self, transport):
        self.transport = transport
        if len(self.connections) >= LIMITS['connections']:
            transport.close()
            return
        self._admitted = True
        super().connection_made(transport)
        transport.set_write_buffer_limits(high=65536, low=16384)
        sock = transport.get_extra_info('socket')
        if sock is not None:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
        self._arm_header_deadline()

    def handle_events(self):
        try:
            return super().handle_events()
        except ValueError:
            # e.g. an overlong decimal Content-Length exceeds Python's integer
            # parser limit inside pinned h11, before an ASGI scope is created.
            self._cancel_header_deadline()
            try:
                self.send_400_response('Invalid HTTP header value.')
            except Exception:
                self.transport.close()

    def data_received(self, data):
        if not self._admitted:
            return
        previous = self.cycle
        if previous is None or previous.response_complete:
            self._arm_header_deadline()
        super().data_received(data)
        if self.cycle is not previous:
            self._cancel_header_deadline()

    def connection_lost(self, exc):
        self._cancel_header_deadline()
        super().connection_lost(exc)
