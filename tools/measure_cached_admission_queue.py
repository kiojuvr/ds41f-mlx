#!/usr/bin/env python3
"""Bounded model-free retention probe: actual Recipe objects and Scheduler queue.
Numerical loading/preflight are excluded, not replaced with another executor.
Run in the disposable installed environment with -I.
"""
import argparse
import base64
from collections import deque
import ctypes
import gc
from io import BytesIO
import json
from pathlib import Path
import time
import tracemalloc
from types import SimpleNamespace as NS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--files', action='store_true', help='exercise the full 64 MiB resolved Files occurrence envelope')
    a = p.parse_args()
    from ds41f_mlx.mtp_identity import config
    from ds41f_mlx.execution_identity import inspect_payload
    inspect_payload()
    from ds41f_mlx.serving.server import prepare_request, load_v41_tokenizer
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend, ProductionScheduler
    from ds41f_mlx.serving.capacity import ORDINARY_BODY_BYTES
    from omlx.scheduler import SchedulerConfig
    from omlx.exceptions import SchedulerQueueFullError
    from PIL import Image
    cfg = config(profile='mtp-serving-v1')
    tokenizer = load_v41_tokenizer(cfg.recipe_path)
    image = BytesIO()
    Image.new('RGB', (1, 1), 'red').save(image, format='PNG')
    url = base64.b64encode(image.getvalue()).decode()
    # Max occurrence metadata; virtually all context positions remain text.
    content = [dict(type='image', source=dict(type='base64', media_type='image/png', data=url)) for _ in range(600)]
    store = None
    if a.files:
        from ds41f_mlx.serving.image_files import ImageFiles
        store = ImageFiles()
        png = image.getvalue()
        # Four large containers plus 596 tiny occurrences exactly fill 64 MiB.
        remaining = 64*1024**2 - 596*len(png)
        sizes = [remaining//4]*4
        sizes[-1] += remaining%4
        def upload(data):
            fields = (b'--probe\r\nContent-Disposition: form-data; name="expires_after[anchor]"\r\n\r\ncreated_at\r\n'
                b'--probe\r\nContent-Disposition: form-data; name="expires_after[seconds]"\r\n\r\n3600\r\n'
                b'--probe\r\nContent-Disposition: form-data; name="file"; filename="probe.png"\r\n'
                b'Content-Type: image/png\r\n\r\n')
            result = store.upload(fields + data + b'\r\n--probe--\r\n', 'multipart/form-data; boundary=probe')
            return dict(type='image', source=dict(type='file', file_id=result['id']))
        content = [upload(png + b'\0'*(size-len(png))) for size in sizes]
        content += [upload(png)]*596
    payload = dict(model=cfg.model_id, max_tokens=1, thinking={'type':'disabled'}, messages=[
        dict(role='user', content=content), dict(role='assistant', content='OK'),
        dict(role='user', content='')])
    def raw(n):
        payload['messages'][-1]['content'] = ' x'*n
        b = json.dumps(payload, separators=(',', ':')).encode()
        assert len(b) < ORDINARY_BODY_BYTES
        # JSON whitespace is valid ingress and retained by the application.
        return b + b' '*(ORDINARY_BODY_BYTES-len(b))
    def prepare(b):
        return prepare_request('messages', b, tokenizer=tokenizer, recipe_path=cfg.recipe_path,
                               checkpoint=cfg.checkpoint_path, ordinary=True, image_files=store)
    baseline = prepare(raw(0))
    n = 1048576 - len(baseline.token_ids) - 16
    del baseline
    b = raw(n)
    check = prepare(b)
    assert 1048500 <= len(check.token_ids) < 1048576
    geometry = dict(prompt_tokens=len(check.token_ids), occurrences=len(check.multimodal.spans),
                    patches=sum(s.grid[0]*s.grid[1] for s in check.multimodal.spans), body_bytes=len(b),
                    encoded_occurrence_bytes=sum(len(x) for x in check.multimodal._sources.values()),
                    transport='Files' if a.files else 'inline')
    del check, b
    gc.collect()
    # Actual queue/add/abort code; no model or physical preflight is instantiated.
    scheduler = object.__new__(ProductionScheduler)
    scheduler.config = SchedulerConfig(max_num_seqs=1)
    scheduler.requests, scheduler.running, scheduler.waiting = {}, {}, deque()
    scheduler.block_aware_cache = NS()
    scheduler._detect_boundary_snapshot_need = lambda: None
    scheduler._completed = []
    backend = object.__new__(ProductionMTPBackend)
    backend.scheduler, backend.fatal_error = scheduler, None
    backend.load = lambda: None
    scheduler.backend = backend
    lib = ctypes.CDLL('/usr/lib/libproc.dylib')
    observations = []
    tracemalloc.start(1)
    def sample(stage):
        buf = ctypes.create_string_buffer(1024)
        assert lib.proc_pid_rusage(__import__('os').getpid(), 4, ctypes.byref(buf)) == 0
        current, peak = tracemalloc.get_traced_memory()
        observations.append(dict(stage=stage, python_current=current, python_peak=peak,
            footprint=ctypes.c_uint64.from_buffer(buf, 72).value,
            lifetime_max_footprint=ctypes.c_uint64.from_buffer(buf, 240).value,
            waiting=len(scheduler.waiting), requests=len(scheduler.requests)))
    sample('baseline')
    retained = []
    started = time.monotonic()
    for i in range(33):
        body = raw(n)  # Independent complete ingress/body/Recipe references.
        prepared = prepare(body)
        backend._enqueue(prepared, tokenizer, str(i))
        retained.append((body, json.loads(body), prepared))
        if i == 0:
            # One active object outside the 32 waiting slots; never native work.
            active = scheduler.waiting.popleft()
            sample('active-proxy')
        if i in (1, 16, 32):
            sample('retained-'+str(i+1))
    overflow_body = raw(n)
    overflow_prepared = prepare(overflow_body)
    try:
        backend._enqueue(overflow_prepared, tokenizer, 'overflow')
    except SchedulerQueueFullError:
        pass
    else:
        raise AssertionError('queue overflow admitted')
    sample('overflow-prepared-34th-retained')
    # A conservative coexistence bound: separate the entire encoded-source
    # allowance for every request, even though real Files immutable references
    # share bytes and inline bodies already include their base64 representation.
    sources = [bytes(bytearray(64*1024**2)) for _ in range(34)]
    sample('full-queue-plus-34x64MiB-source-upper-bound')
    sources.clear()
    overflow_prepared.multimodal.release_sources()
    overflow_body = overflow_prepared = None
    for rid in list(scheduler.requests):
        assert scheduler._do_abort_request(rid)
    sample('queued-aborted-app-still-retained')
    assert not scheduler.requests and not scheduler.waiting
    assert all(not prepared.multimodal._sources for _, _, prepared in retained)
    assert not scheduler.running
    scheduler._completed.clear()
    retained.clear()
    body = prepared = active = None
    gc.collect()
    sample('application-released')
    if store is not None:
        store.clear()
    gc.collect()
    sample('store-released')
    a.output.write_text(json.dumps(dict(status='PASS', geometry=geometry,
        seconds=time.monotonic()-started, observations=observations,
        scope='actual Recipe and queue metadata; no model/native/HTTP socket'), indent=2)+'\n')


if __name__ == '__main__':
    main()
