"""Finite model-free Files listing and installed official quota recovery."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
from threading import Thread
import time

import pytest
import uvicorn

from test_image_files import http, upload, png
from ds41f_mlx.serving.image_files import FileError, FileLimits


@pytest.mark.parametrize('query', ['limit=0', 'limit=1001', 'limit=-1', 'limit=1.0',
    'limit=', 'limit=1&limit=2', 'after_id=', 'after_id=missing', 'order=asc',
    'limit=1&', 'limit=%31', 'limit=1&purpose=image', 'x=' + 'a' * 200])
def test_invalid_listing_query(http, query):
    client, _, _ = http
    assert client.get('/v1/files?' + query).status_code == 400
    assert client.get('/v1/models?limit=1').status_code == 400
    assert client.get('/health?limit=1').status_code == 400


def test_pagination_expiry_and_concurrent_changes(http):
    client, store, _ = http
    assert client.get('/v1/files').json() == dict(data=[], first_id=None, last_id=None, has_more=False)
    receipts = [upload(client, filename=f'{i}.png') for i in range(1005)]
    single = client.get('/v1/files?limit=1').json()
    assert single == dict(data=receipts[:1], first_id=receipts[0]['id'],
                          last_id=receipts[0]['id'], has_more=True)
    first = client.get('/v1/files?limit=1000').json()
    assert first == dict(data=receipts[:1000], first_id=receipts[0]['id'],
                         last_id=receipts[999]['id'], has_more=True)
    cursor = first['last_id']
    second = client.get('/v1/files?limit=1000&after_id=' + cursor).json()
    assert second == dict(data=receipts[1000:], first_id=receipts[1000]['id'],
                          last_id=receipts[-1]['id'], has_more=False)
    assert client.get('/v1/files?after_id=' + cursor + '&limit=5').json() == second
    assert client.get('/v1/files?after_id=' + second['last_id']).json()['data'] == []
    # Surviving cursors tolerate deletes before them and additions after them.
    snapshot = store.snapshot(receipts[0]['id'])
    client.delete('/v1/files/' + receipts[0]['id'])
    extra = upload(client)
    assert client.get('/v1/files?after_id=' + cursor).json()['data'] == receipts[1000:] + [extra]
    client.delete('/v1/files/' + cursor)
    assert client.get('/v1/files?after_id=' + cursor).status_code == 400
    with pytest.raises(FileError):
        store.snapshot(receipts[0]['id'])
    assert snapshot.data == png()
    assert client.get('/v1/files?after_id=file_' + '0' * 32).status_code == 400
    store.clock = lambda: 10**12
    assert client.get('/v1/files').json()['data'] == []
    assert client.get('/v1/files?after_id=' + extra['id']).status_code == 400
    assert snapshot.data == png()


def test_listing_concurrent_deletion(http):
    from concurrent.futures import ThreadPoolExecutor
    client, store, _ = http
    receipts = [upload(client) for _ in range(32)]
    snapshots = [store.snapshot(r['id']) for r in receipts]

    def list_pages():
        for _ in range(32):
            page = store.listing(b'limit=7')
            assert len(page['data']) <= 7
            assert len({r['id'] for r in page['data']}) == len(page['data'])
            assert all(r in receipts for r in page['data'])

    def delete_files():
        for receipt in receipts:
            store.delete(receipt['id'])

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(list_pages), pool.submit(delete_files), pool.submit(list_pages)]
        for future in futures:
            future.result()
    assert store.listing()['data'] == []
    assert all(snapshot.data == png() for snapshot in snapshots)


@pytest.fixture
def wire(http):
    client, store, _ = http
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    server = uvicorn.Server(uvicorn.Config(client.app, lifespan='off', log_level='error'))
    thread = Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    thread.start()
    for _ in range(500):
        if server.started:
            break
        time.sleep(.01)
    assert server.started
    try:
        yield client, store, f'http://127.0.0.1:{sock.getsockname()[1]}/v1'
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()
        assert not thread.is_alive()


@pytest.mark.parametrize('scenario', ['default-quota', 'multi-page-oldest', 'none-owned', 'insufficient', 'cancel', 'cancel-list', 'delete-race'])
def test_installed_official_recovery(wire, tmp_path, scenario):
    root = Path(os.environ.get('HARNESS_NODE_MODULES', '/Users/kioju/.local/share/deepseek-harness/node_modules'))
    if not shutil.which('node') or not (root / '@deepseek-ai/dsh-llm-deepseek/lib/index.js').is_file():
        pytest.skip('installed official Harness unavailable; not an end-to-end qualification')
    client, store, url = wire
    data = png()
    options = dict(name=scenario, url=url, success=True)
    unrelated = []
    snapshots = []
    if scenario == 'default-quota':
        from PIL import Image
        from io import BytesIO
        out = BytesIO()
        Image.new('RGB', (2048, 2048), 'blue').save(out, format='PNG', compress_level=0)
        data = out.getvalue()
        unrelated.append(upload(client)['id'])
        options['fill'] = True
    elif scenario == 'multi-page-oldest':
        now = int(time.time())
        store.clock = lambda: now
        unrelated = [upload(client, filename=('DSH-not-owned.png' if i == 0 else
                     'prefix-dsh-not-owned.png' if i == 1 else f'other-{i}.png'))['id'] for i in range(1001)]
        upload(client, filename='dsh-newer.png')
        store.clock = lambda: now - 10
        oldest = upload(client, filename='dsh-oldest.png')['id']
        snapshots.append(store.snapshot(oldest))
        store.clock = lambda: now
        # Deliberately oldest on page two, not first in provider order.
        store.limits = FileLimits(storage_bytes=store.retained_bytes(store._files) + 128)
        options.update(pages=2, deleted=[oldest])
    elif scenario in ('cancel-list', 'delete-race'):
        owned = upload(client, filename='dsh-racing.png')['id']
        store.limits = FileLimits(storage_bytes=store.retained_bytes(store._files))
        options.update(success=False, attempts=1, deleted=[owned] if scenario == 'delete-race' else [])
        if scenario == 'delete-race':
            options['raceDelete'] = owned
        else:
            options['cancelDuringList'] = True
    elif scenario in ('none-owned', 'insufficient'):
        unrelated.append(upload(client)['id'])
        owned = upload(client, filename='dsh-small.png')['id'] if scenario == 'insufficient' else None
        store.limits = FileLimits(storage_bytes=store.retained_bytes(store._files))
        options.update(success=False, attempts=2 if owned else 1, deleted=[owned] if owned else [])
        # Reclaiming one tiny file cannot admit this larger valid image.
        if owned:
            from PIL import Image
            from io import BytesIO
            out = BytesIO()
            Image.new('RGB', (128, 128)).save(out, format='PNG', compress_level=0)
            data = out.getvalue()
    else:
        options['cancel'] = True
    image = tmp_path / 'image.png'
    image.write_bytes(data)
    options['image'] = str(image)
    config = tmp_path / 'input.json'
    config.write_text(json.dumps(options))
    result = subprocess.run(['node', 'tools/check_harness_files_recovery.mjs', str(root), str(config)],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)
    assert all(store.snapshot(key).data == png() for key in unrelated)
    for key in options.get('deleted', []):
        with pytest.raises(FileError):
            store.snapshot(key)
        # A new Messages request fails, never substitutes or replays an image.
        from test_image_files import body, ref
        assert client.post('/v1/messages', json=body([ref(key)])).status_code == 404
    assert all(snapshot.data == png() for snapshot in snapshots)
    assert store.retained_bytes(store._files) <= store.limits.storage_bytes
    print(json.dumps(dict(scenario=scenario, retained_entries=len(store._files),
        retained_bytes=store.retained_bytes(store._files), budget=store.limits.storage_bytes)))
