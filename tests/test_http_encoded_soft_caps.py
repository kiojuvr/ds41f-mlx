"""Finite audit of unchanged production-size ingress and occurrence guards."""
import asyncio
import base64
import json
import struct
import zlib

import pytest

from test_image_files import http, upload, body, ref, inline, png
from ds41f_mlx.serving.capacity import ORDINARY_BODY_BYTES
from ds41f_mlx.serving.ordinary_admission import admit_structure


def sized_png(size):
    """Valid small-pixel PNG with an opaque ancillary chunk; no pixel bomb."""
    original = png()
    payload = b'x' * (size - len(original) - 12)
    kind = b'npAD'
    chunk = struct.pack('>I', len(payload)) + kind + payload
    chunk += struct.pack('>I', zlib.crc32(kind + payload))
    return original[:-12] + chunk + original[-12:]


@pytest.mark.parametrize('path', ['/v1/messages', '/v1/chat/completions'])
def test_real_body_boundary(http, path):
    client, _, prepared = http
    wire = json.dumps(body([])).encode()
    below = wire + b' ' * (ORDINARY_BODY_BYTES - len(wire))
    assert client.post(path, content=below, headers={'content-type': 'application/json'}).status_code == 200
    assert len(prepared) == 1
    above = below + b' '
    assert client.post(path, content=above, headers={'content-type': 'application/json'}).status_code == 413
    with pytest.raises(ValueError, match='body limit'):
        admit_structure(above)
    assert len(prepared) == 1


@pytest.mark.parametrize('wire', [b'{', b'{"model":"a","model":"b"}', b'[]', b'{"x":NaN}', b'{"x":1e999}'])
def test_strict_json_rejections(http, wire):
    client, _, prepared = http
    for path in ['/v1/messages', '/v1/chat/completions']:
        assert client.post(path, content=wire, headers={'content-type': 'application/json'}).status_code == 400
    assert not prepared


@pytest.mark.parametrize('mixed', [False, True])
def test_actual_encoded_boundary_before_next_base64(http, monkeypatch, mixed):
    import ds41f_mlx.serving.image_files as files
    client, store, prepared = http
    data = sized_png(2 * 1024 * 1024)
    key = upload(client, data=data)['id']
    calls = []
    original = base64.b64encode

    def counted(value):
        calls.append(len(value))
        return original(value)

    content = ([inline(data)] + [ref(key) for _ in range(32)] if mixed
               else [ref(key) for _ in range(33)])
    monkeypatch.setattr(files.base64, 'b64encode', counted)
    response = client.post('/v1/messages', json=body(content))
    assert response.status_code == 400 and 'aggregate encoded' in response.text
    assert len(calls) == (31 if mixed else 32)
    assert not prepared  # No model preparation or Scheduler publication.
    assert store.snapshot(key).data == data
    assert client.post('/v1/messages', json=body([ref(key)])).status_code == 200
    assert len(prepared) == 1  # Rejection released the preparation lease.


def test_inline_harness_sized_body_mismatch(http):
    client, _, prepared = http
    image = inline(sized_png(1024 * 1024))
    wire = json.dumps(body([image] * 13)).encode()
    represented = 13 * len(image['source']['data'])
    assert ORDINARY_BODY_BYTES < len(wire)
    assert represented < 20 * 1024 * 1024
    assert client.post('/v1/messages', content=wire, headers={'content-type': 'application/json'}).status_code == 413
    assert not prepared


@pytest.mark.parametrize('error', [RuntimeError, asyncio.CancelledError])
def test_source_owner_clears_on_encoder_baseexception(http, monkeypatch, error):
    import gc
    import weakref
    from types import SimpleNamespace
    import ds41f_mlx.runtime.multimodal as mm
    client, _, prepared = http
    assert client.post('/v1/messages', json=body([inline(png())])).status_code == 200
    request = prepared.pop()
    image = request.multimodal
    identities = image.identities()
    assert image._sources
    def fail(*args):
        raise error('controlled encoder exit; no model execution')
    monkeypatch.setattr(mm, '_encode_span', fail)
    with pytest.raises(error):
        image.encode_new(SimpleNamespace(), span_encoder=object())
    assert not image._sources and image.identities() == identities
    request_ref, image_ref = weakref.ref(request), weakref.ref(image)
    del request, image
    gc.collect()
    assert request_ref() is None and image_ref() is None


@pytest.mark.parametrize('mode', ['timeout', 'cancel', 'disconnect'])
def test_ingress_cleanup(mode):
    from ds41f_mlx.serving.mtp_public import LocalBoundary
    from starlette.requests import Request, ClientDisconnect

    async def app(scope, receive, send):
        await Request(scope, receive).body()

    async def run():
        boundary = LocalBoundary(app, authority='localhost', ordinary=True, body_timeout=0.02)
        scope = dict(type='http', method='POST', path='/v1/messages', query_string=b'',
                     headers=[(b'host', b'localhost'), (b'content-type', b'application/json')])
        sent = []
        async def send(event):
            sent.append(event)
        async def receive():
            if mode == 'disconnect':
                return {'type': 'http.disconnect'}
            await asyncio.sleep(10)
        task = asyncio.create_task(boundary(scope, receive, send))
        if mode == 'cancel':
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif mode == 'disconnect':
            with pytest.raises(ClientDisconnect):
                await task
        else:
            await task
            assert sent[0]['status'] == 408
        assert not boundary.preparing and not boundary.preparation_lock.locked()
    asyncio.run(run())
