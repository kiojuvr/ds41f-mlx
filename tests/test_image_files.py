"""Gate W2: real HTTP ingress/Recipe/Vision preparation, no numerical backend."""
import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import os
from types import SimpleNamespace as NS

from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from ds41f_mlx.serving.image_files import FileError, FileLimits, ImageFiles, MAX_OCCURRENCES


def png(color='red'):
    out = BytesIO()
    Image.new('RGB', (32, 32), color).save(out, format='PNG')
    return out.getvalue()


def form(data=None, mime='image/png', seconds='604800', anchor='created_at', filename='original.png'):
    return dict(files={'file': (filename, png() if data is None else data, mime)},
                data={'expires_after[anchor]': anchor, 'expires_after[seconds]': seconds})


def ref(key):
    return dict(type='image', source=dict(type='file', file_id=key))


def inline(data):
    return dict(type='image', source=dict(type='base64', media_type='image/png',
                                        data=base64.b64encode(data).decode()))


def body(content):
    return dict(model='deepseek-flash', max_tokens=64, thinking={'type': 'disabled'},
                messages=[dict(role='user', content=content + [dict(type='text', text='describe this image')])])


@pytest.fixture
def http(monkeypatch):
    from ds41f_mlx.mtp_identity import config
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    from ds41f_mlx.serving.server import create_app
    previous = dict(os.environ)
    backend = ProductionMTPBackend(runtime_config=config(profile='mtp-serving-v1'))
    backend.dependency_identity = 'controlled-model-free'
    backend.scheduler = NS(requests={}, close_serving=lambda: None)
    prepared = []

    async def respond(request, **kwargs):
        prepared.append(request)
        return JSONResponse({'prepared': True})

    monkeypatch.setattr(backend, 'ordinary_response', respond)
    try:
        app = create_app(backend=backend, runtime_config=backend.runtime_config)
        with TestClient(app, base_url='http://127.0.0.1:8000') as client:
            yield client, app.state.image_files, prepared
    finally:
        backend.scheduler = None
        backend.close()
        os.environ.clear()
        os.environ.update(previous)


def upload(client, **kwargs):
    result = client.post('/v1/files', **form(**kwargs))
    assert result.status_code == 200, result.text
    return result.json()


def test_contract_and_deletion(http):
    client, store, prepared = http
    receipt = upload(client)
    assert set(receipt) == {'id', 'type', 'mime_type', 'size_bytes', 'created_at', 'filename'}
    assert receipt | {'id': 'opaque', 'created_at': 'time'} == dict(id='opaque', type='file',
        mime_type='image/png', size_bytes=len(png()), created_at='time', filename='original.png')
    assert receipt['created_at'].endswith('Z')
    snapshot = store.snapshot(receipt['id'])
    assert snapshot.data == png() and snapshot.mime == 'image/png'
    assert client.delete('/v1/files/' + receipt['id']).json() == dict(id=receipt['id'], type='file_deleted')
    assert snapshot.data == png()  # immutable acquired snapshot outlives deletion
    assert client.delete('/v1/files/' + receipt['id']).status_code == 404
    assert not prepared


@pytest.mark.parametrize('kwargs', [dict(data=b'invalid'), dict(mime='image/jpeg'),
    dict(mime='image/gif'), dict(anchor='last_active_at'), dict(seconds='3599'),
    dict(seconds='2592001'), dict(seconds='1.5'), dict(seconds='3600.0'), dict(filename='')])
def test_invalid_upload_no_publication(http, kwargs):
    client, store, _ = http
    assert client.post('/v1/files', **form(**kwargs)).status_code == 400
    assert not store._files


@pytest.mark.parametrize('fmt,mime', [('PNG', 'image/png'), ('JPEG', 'image/jpeg'), ('WEBP', 'image/webp')])
def test_qualified_binary_formats_preserved(http, fmt, mime):
    client, store, _ = http
    output = BytesIO()
    Image.new('RGB', (32, 32), 'blue').save(output, format=fmt)
    data = output.getvalue()
    receipt = upload(client, data=data, mime=mime)
    assert receipt['mime_type'] == mime and receipt['size_bytes'] == len(data)
    assert store.snapshot(receipt['id']).data == data


def test_small_wire_reference_can_resolve_beyond_json_wire_limit(http):
    client, _, prepared = http
    output = BytesIO()
    Image.new('RGB', (2048, 2048), 'red').save(output, format='PNG', compress_level=0)
    data = output.getvalue()
    assert len(base64.b64encode(data)) > 16 * 1024 * 1024
    key = upload(client, data=data)['id']
    response = client.post('/v1/messages', json=body([ref(key)]))
    assert response.status_code == 200, response.text
    assert list(prepared[-1].multimodal._sources.values()) == [data]


def test_resolved_byte_budget_before_recipe(http, monkeypatch):
    import ds41f_mlx.serving.image_files as files
    client, _, prepared = http
    key = upload(client)['id']
    monkeypatch.setattr(files, 'MAX_REQUEST_IMAGE_BYTES', len(png()) * 2)
    r = client.post('/v1/messages', json=body([ref(key)] * 3))
    assert r.status_code == 400 and 'aggregate' in r.text
    assert not prepared


def test_bounds_and_malformed_partial(http):
    client, store, _ = http
    assert client.post('/v1/files', files={'file': ('x.png', png(), 'image/png')}).status_code == 400
    f = form()
    f['data']['purpose'] = 'user_data'
    assert client.post('/v1/files', **f).status_code == 400
    assert client.post('/v1/files', content=b'--broken\r\n',
                       headers={'content-type': 'multipart/form-data; boundary=broken'}).status_code == 400
    assert client.post('/v1/files', content=b'x' * (16 * 1024 * 1024 + 65537),
                       headers={'content-type': 'multipart/form-data; boundary=x'}).status_code == 413
    assert not store._files
    store.limits = FileLimits(file_bytes=len(png()) - 1)
    assert client.post('/v1/files', **form()).status_code == 413
    store.limits = FileLimits(count=1)
    receipt = upload(client)
    assert client.post('/v1/files', **form()).status_code == 429
    client.delete('/v1/files/' + receipt['id'])
    store.limits = FileLimits(storage_bytes=len(png()) - 1)
    assert client.post('/v1/files', **form()).status_code == 429
    assert not store._files


def test_inline_file_and_replacement_identity_and_origins(http):
    client, store, prepared = http
    a, b = upload(client), upload(client)
    assert a['id'] != b['id']
    for content in ([inline(png())], [ref(a['id'])], [ref(b['id'])]):
        assert client.post('/v1/messages', json=body(content)).status_code == 200
    baseline = prepared[0]
    for p in prepared[1:]:
        assert p.token_ids == baseline.token_ids
        assert p.multimodal.identities() == baseline.multimodal.identities()
        assert p.image_is_historical == baseline.image_is_historical == (False,)
        assert p.multimodal._sources == baseline.multimodal._sources
    # Repeated historical occurrences stay separate and ordered. Two encoded
    # bytes variants, then repeat the first; no provider ID enters identity.
    c = upload(client, data=png('blue'))
    history = body([ref(a['id']), ref(c['id']), ref(a['id'])])
    history['messages'] += [dict(role='assistant', content='observed'), dict(role='user', content='continue')]
    assert client.post('/v1/messages', json=history).status_code == 200
    p = prepared[-1]
    assert p.image_is_historical == (True, True, True)
    ids = p.multimodal.identities()
    assert ids[0]['sha256'] == ids[2]['sha256'] != ids[1]['sha256']
    assert [i['start'] for i in ids] == sorted(i['start'] for i in ids)


def test_nested_exact_tool_id_and_equivalent_order(http):
    client, _, prepared = http
    key = upload(client)['id']
    call = 'opaque:original-call-123'
    def request(image):
        b = body([])
        b['tools'] = [dict(name='read_image', input_schema={'type': 'object'})]
        b['messages'] += [dict(role='assistant', content=[dict(type='tool_use', id=call, name='read_image', input={})]),
            dict(role='user', content=[dict(type='tool_result', tool_use_id=call,
                content=[dict(type='text', text='result'), image]), dict(type='text', text='describe')])]
        return b
    for image in [inline(png()), ref(key)]:
        r = client.post('/v1/messages', json=request(image))
        assert r.status_code == 200, r.text
    a, b = prepared
    assert a.token_ids == b.token_ids
    assert a.multimodal.identities() == b.multimodal.identities()
    assert b.image_is_historical == (False,)
    assistant = next(m for m in b.conversation_request.conversation.messages if m.role == 'assistant')
    assert assistant.tool_calls[0].id == call


def test_stale_scope_expiry_delete_before_execution(http):
    client, store, prepared = http
    foreign = ImageFiles()
    # An actual ID uploaded in another deployment instance is unavailable here.
    foreign_key = upload(client)['id']
    foreign._files[foreign_key] = store._files.pop(foreign_key)
    keys = [foreign_key, 'unknown', '../weights']
    deleted = upload(client)['id']
    client.delete('/v1/files/' + deleted)
    keys.append(deleted)
    expired = upload(client)['id']
    store.clock = lambda: 10**12
    keys.append(expired)
    for key in keys:
        r = client.post('/v1/messages', json=body([ref(key)]))
        assert r.status_code == 404, r.text
        assert r.json()['error']['code'] == 'file_not_found'
        assert 'file not found or expired' in r.json()['error']['message']
    assert not store._files and not prepared


@pytest.mark.parametrize('location', ['assistant', 'system', 'source_extra', 'block_extra', 'nested_result'])
def test_invalid_locations_and_fields(http, location):
    client, _, prepared = http
    image = ref(upload(client)['id'])
    b = body([image])
    if location == 'assistant':
        b['messages'][0]['role'] = 'assistant'
    elif location == 'system':
        b['system'] = [image]
        b['messages'][0]['content'] = 'hi'
    elif location == 'source_extra':
        image['source']['url'] = '/tmp/image.png'
    elif location == 'block_extra':
        image['cache_control'] = {}
    else:
        b['messages'][0]['content'] = [dict(type='tool_result', tool_use_id='x', content=[
            dict(type='tool_result', tool_use_id='y', content=[image])])]
    r = client.post('/v1/messages', json=b)
    assert r.status_code == 400, r.text
    assert not prepared


def test_resolution_budget_and_unrelated_extensions(http):
    client, store, prepared = http
    key = upload(client)['id']
    assert client.post('/v1/messages', json=body([ref(key)] * (MAX_OCCURRENCES + 1))).status_code == 400
    # Pre-materialization aggregate pixel budget counts repeat occurrences.
    out = BytesIO()
    Image.new('RGB', (2048, 2048)).save(out, format='PNG')
    large = upload(client, data=out.getvalue())['id']
    assert client.post('/v1/messages', json=body([ref(large)] * 5)).status_code == 400
    for extension in ({'dsh_session_log': {}}, {'unknown': True}):
        assert client.post('/v1/messages', json=body([ref(key)]) | extension).status_code == 400
    assert not prepared
    assert client.get('/v1/files').status_code == 404
    assert client.get('/v1/files/' + key).status_code == 404


def test_concurrent_snapshot_and_delete(http):
    client, store, _ = http
    key = upload(client)['id']
    with ThreadPoolExecutor(2) as pool:
        snapshot = pool.submit(store.snapshot, key).result()
        pool.submit(store.delete, key).result()
    assert snapshot.data == png()
    with pytest.raises(FileError):
        store.snapshot(key)


def test_disconnect_partial_upload_releases_lease(http):
    from starlette.requests import ClientDisconnect
    client, store, _ = http
    request = client.build_request('POST', '/v1/files', **form())
    wire = request.read()
    events = iter([{'type': 'http.request', 'body': wire[:len(wire)//2], 'more_body': True},
                   {'type': 'http.disconnect'}])
    scope = dict(type='http', asgi={'version': '3.0'}, http_version='1.1', method='POST',
                 scheme='http', path='/v1/files', raw_path=b'/v1/files', root_path='',
                 query_string=b'', headers=request.headers.raw, client=('127.0.0.1', 1),
                 server=('127.0.0.1', 8000))
    async def run():
        async def receive():
            return next(events)
        async def send(event):
            pass
        with pytest.raises(ClientDisconnect):
            await client.app(scope, receive, send)
    client.portal.call(run)
    assert not store._files
    upload(client)  # lease was released; no partial object ever published


def test_historical_file_still_requires_paired_state(http):
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    from ds41f_mlx.serving.paired_checkpoint import PairedCheckpointAuthority
    from test_metadata_first_vision import scheduler_request
    from test_production_mtp import MX
    client, _, prepared = http
    key = upload(client)['id']
    b = body([ref(key)])
    b['messages'] += [dict(role='assistant', content='seen'), dict(role='user', content='continue')]
    assert client.post('/v1/messages', json=b).status_code == 200
    p = prepared[-1]
    queued = []
    backend = NS(fatal_error=None, load=lambda: None,
                 scheduler=NS(add_request=queued.append), request_max_tokens=lambda p: 64)
    ProductionMTPBackend._enqueue(backend, p, None, 'file-history')
    authority = PairedCheckpointAuthority('identity', MX)
    scheduler, request = scheduler_request(p.multimodal, authority)
    request.prepared = p
    request.required_image_frontier = queued[0].required_image_frontier
    errors = []
    def forbidden_start(*args, **kwargs):
        pytest.fail('historical images must not start inference on a missing pair')
    scheduler.backend = NS(_start=forbidden_start)
    scheduler._retire = lambda req, *, error: errors.append(error)
    scheduled, _ = scheduler._schedule_waiting()
    assert not scheduled and not request.native_started
    assert len(errors) == 1 and 'historical image pair unavailable' in str(errors[0])
    authority.clear()


def test_delete_during_http_resolution_preserves_acquired_bytes(http, monkeypatch):
    client, store, prepared = http
    key = upload(client)['id']
    snapshot = store.snapshot
    def resolve_then_delete(file_id):
        image = snapshot(file_id)
        # A concurrent owner removes storage after resolution has acquired bytes.
        monkeypatch.setattr(store, 'snapshot', snapshot)
        with ThreadPoolExecutor(1) as pool:
            pool.submit(store.delete, file_id).result()
        return image
    monkeypatch.setattr(store, 'snapshot', resolve_then_delete)
    assert client.post('/v1/messages', json=body([ref(key)])).status_code == 200
    assert list(prepared[-1].multimodal._sources.values()) == [png()]
    monkeypatch.setattr(store, 'snapshot', snapshot)
    assert client.post('/v1/messages', json=body([ref(key)])).status_code == 404


def test_busy_upload_and_raw_messages_body_fence(http):
    client, store, prepared = http
    # Exercise the actual no-queue preparation/upload lease.
    boundary = client.app.middleware_stack
    while not hasattr(boundary, 'preparation_lock'):
        boundary = boundary.app
    assert boundary.preparation_lock is not None
    client.portal.call(boundary.preparation_lock.acquire)
    try:
        assert client.post('/v1/files', **form()).status_code == 503
    finally:
        client.portal.call(boundary.preparation_lock.release)
    assert client.post('/v1/messages', content=b' ' * (16 * 1024 * 1024 + 1),
                       headers={'content-type': 'application/json'}).status_code == 413
    assert not store._files and not prepared
