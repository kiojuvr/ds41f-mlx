#!/usr/bin/env python3
"""Isolated finite pixel-admission probe, synthetic images, no Harness/tool effects.

Run -I in an admitted non-editable candidate. Never installs or touches Production.
Default measures preparation only; --native adds one 20-image incremental history.
"""
import argparse
import asyncio
import base64
import ctypes
from dataclasses import replace
import gc
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import tracemalloc
from unittest.mock import patch
import weakref

SIZES = ((694,698), (694,698), (694,698), (1110,1110), (720,720),
         (1040,1040), (1302,1302), (694,698), (694,698), (1120,1280),
         (990,1080), (900,720), (900,1602), (694,698), (1080,1080),
         (880,800), (880,800), (1080,1080), (694,698), (880,800))
JPEG = {2, 5, 6, 7, 8, 13, 14, 17}
GiB = 1024**3


def fixture_images():
    """Same dimensions/formats as the supplied incident, not its image contents."""
    from PIL import Image, ImageDraw
    images = []
    for i, size in enumerate(SIZES):
        if i == 1:
            images.append(images[0])  # repeated occurrence still pays encoded bytes
            continue
        fmt = 'JPEG' if i in JPEG else 'PNG'
        with Image.new('RGBA' if i == 0 else 'RGB', size, 'white') as image:
            ImageDraw.Draw(image).rectangle((20, 20, 100, 80), fill=(30+i*7, 60, 100))
            out = BytesIO()
            image.save(out, format=fmt)
            images.append(out.getvalue())
    return images


def inline(data, i):
    return dict(type='image', source=dict(type='base64',
        media_type='image/jpeg' if i in JPEG else 'image/png', data=base64.b64encode(data).decode()))


def upload_fixtures(store, images):
    import httpx
    ids = []
    for i, data in enumerate(images):
        mime = 'image/jpeg' if i in JPEG else 'image/png'
        request = httpx.Request('POST', 'http://fixture/v1/files',
            files={'file': (f'fixture-{i:02}', data, mime)},
            data={'expires_after[anchor]': 'created_at', 'expires_after[seconds]': '604800'})
        ids.append(store.upload(request.read(), request.headers['content-type'])['id'])
    return ids


def file_ref(key):
    return dict(type='image', source=dict(type='file', file_id=key))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--native', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (a.output/name).write_text(json.dumps(value, indent=2)+'\n')
    def cmd(*args):
        return subprocess.check_output(args, text=True)
    save('host-before.json', dict(vm=cmd('vm_stat'), swap=cmd('sysctl', 'vm.swapusage'),
        pressure=cmd('memory_pressure', '-Q'), hardware=cmd('sysctl', 'hw.memsize')))
    from ds41f_mlx.mtp_identity import config, inspect
    from ds41f_mlx.execution_identity import inspect_payload
    cfg = replace(config(profile='mtp-serving-v1'), kv_root=a.output/'unused-kv')
    authority, _ = inspect_payload()
    import ds41f_mlx
    installed = Path(ds41f_mlx.__file__).parent
    sources = {}
    for src in (a.source/'ds41f_mlx').rglob('*.py'):
        rel = src.relative_to(a.source/'ds41f_mlx')
        data = (installed/rel).read_bytes()
        assert data == src.read_bytes(), str(rel)
        sources[str(rel)] = sha256(data).hexdigest()
    save('sources.json', sources)
    save('provenance.json', authority['provenance'])
    cfg.apply_environment()
    from ds41f_mlx.serving.server import prepare_request, load_v41_tokenizer
    from ds41f_mlx.serving.image_files import ImageFiles
    from ds41f_mlx.runtime import multimodal as mm
    tokenizer = load_v41_tokenizer(cfg.recipe_path)
    lib = ctypes.CDLL('/usr/lib/libproc.dylib')
    def footprint():
        buf = ctypes.create_string_buffer(1024)
        assert lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buf)) == 0
        return dict(footprint=ctypes.c_uint64.from_buffer(buf, 72).value,
                    lifetime_max_footprint=ctypes.c_uint64.from_buffer(buf, 240).value)
    images = fixture_images()
    store = ImageFiles()
    ids = upload_fixtures(store, images)
    save('workload.json', [dict(index=i, width=w, height=h, source_pixels=w*h,
        encoded_bytes=len(images[i]), sha256=sha256(images[i]).hexdigest(),
        mime='image/jpeg' if i in JPEG else 'image/png') for i, (w, h) in enumerate(SIZES)])
    assert sum(w*h for w, h in SIZES) == 16955588
    def prepare(history):
        raw = json.dumps(dict(model=cfg.model_id, max_tokens=16, temperature=0,
            thinking={'type':'disabled'}, stream=False, messages=history)).encode()
        assert len(raw) < 8*1024**2
        return prepare_request('messages', raw, tokenizer=tokenizer, recipe_path=cfg.recipe_path,
            checkpoint=cfg.checkpoint_path, ordinary=True, image_files=store)
    # Instrument only source validation, not admission, arrays or numerical work.
    from PIL import Image
    open_image, validate = Image.open, mm.validate_encoded_image
    refs, validation = [], []
    max_live = 0
    def opened(*args, **kwargs):
        nonlocal max_live
        image = open_image(*args, **kwargs)
        refs.append(weakref.ref(image))
        max_live = max(max_live, sum(r() is not None for r in refs))
        return image
    def checked(*args, **kwargs):
        assert not any(r() is not None for r in refs), 'preceding decoded image retained'
        wall, cpu = time.perf_counter(), time.process_time()
        try:
            return validate(*args, **kwargs)
        finally:
            validation.append(dict(wall_s=time.perf_counter()-wall, cpu_s=time.process_time()-cpu))
    measured, identity = {}, None
    for mode in ('file', 'inline'):
        history = []
        for i, data in enumerate(images):
            image = file_ref(ids[i]) if mode == 'file' else inline(data, i)
            history.extend([dict(role='user', content=[image, dict(type='text', text='Reply only OK.')]),
                            dict(role='assistant', content='OK')])
        history.append(dict(role='user', content='Reply only OK.'))
        refs.clear()
        validation.clear()
        max_live = 0
        gc.collect()
        before = footprint()
        tracemalloc.start()
        wall, cpu = time.perf_counter(), time.process_time()
        with patch.object(Image, 'open', opened), patch.object(mm, 'validate_encoded_image', checked):
            prepared = prepare(history)
        duration = dict(wall_s=time.perf_counter()-wall, cpu_s=time.process_time()-cpu)
        traced_current, traced_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert len(validation) == 20 and len(refs) == 40 and max_live <= 2
        assert not any(r() is not None for r in refs)
        assert prepared.image_is_historical == (True,)*20
        current_identity = (prepared.token_ids, prepared.multimodal.identities())
        if identity is not None:
            assert identity == current_identity
        identity = current_identity
        measured[mode] = dict(**duration, validation_calls=len(validation), image_open_calls=len(refs),
            validation_wall_s=sum(r['wall_s'] for r in validation),
            validation_cpu_s=sum(r['cpu_s'] for r in validation), max_live_pil_handles=max_live,
            live_pil_after=0, traced_current=traced_current, traced_peak=traced_peak,
            before=before, after=footprint(), prompt_frontier=len(prepared.token_ids),
            total_pixels=sum(s.source_pixels for s in prepared.multimodal.spans),
            total_patches=sum(s.grid[0]*s.grid[1] for s in prepared.multimodal.spans))
        # Preparation itself is NOT permission for C=0 historical reconstruction.
        try:
            prepared.multimodal.validate_workload(0)
        except ValueError:
            pass
        else:
            raise AssertionError('cold reconstruction bypassed fresh workload limit')
        prepared.multimodal.release_sources()
        del prepared, history, current_identity
    identity = None
    save('preparation.json', measured)
    if not a.native:
        store.clear()
        save('result.json', dict(status='PREPARATION_ONLY'))
        return
    # Full checkpoint admission before backend construction/loading.
    identity = inspect(cfg)
    save('identity.json', identity)
    import mlx.core as mx
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    from ds41f_mlx.observability import TRACE_FIELDS
    backend = ProductionMTPBackend(runtime_config=cfg)
    backend.dependency_identity, backend.glm_profile = identity['identity_sha256'], identity['glm_profile']
    stop, unsafe = threading.Event(), []
    case = 'startup'
    def sample(event, **extra):
        row = dict(t=time.monotonic(), case=case, event=event, **footprint(),
            mlx_active=mx.get_active_memory(), mlx_cache=mx.get_cache_memory(),
            mlx_peak=mx.get_peak_memory(), **extra)
        with (a.output/'samples.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        if max(row['footprint'], row['mlx_active']+row['mlx_cache']) > 360*GiB:
            unsafe.append('360 GiB process/allocator stop')
    def check():
        assert not unsafe, unsafe
    def watch():
        try:
            while not stop.wait(2):
                vm, swap, pressure = cmd('vm_stat'), cmd('sysctl', 'vm.swapusage'), cmd('memory_pressure', '-Q')
                reserve = sum(int(re.search(n+r':\s+(\d+)', vm)[1])*16384
                    for n in ('Pages free', 'Pages inactive', 'Pages speculative'))
                with (a.output/'host.jsonl').open('a') as f:
                    f.write(json.dumps(dict(t=time.monotonic(), vm=vm, swap=swap,
                        pressure=pressure, reserve_proxy=reserve))+'\n')
                if reserve < 100*GiB or int(re.search(r'free percentage: (\d+)%', pressure)[1]) < 25:
                    unsafe.append('host reserve/pressure stop')
                if re.search(r'used = (?!0\.00M)\S+', swap):
                    unsafe.append('nonzero swap')
                sample('watch')
        except BaseException:
            unsafe.append('observer failed')
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    history, traces = [], []
    async def run(label, i=None):
        nonlocal case
        case = label
        check()
        content = [dict(type='text', text='Reply only OK.')]
        if i is not None:
            content.insert(0, file_ref(ids[i]))
        history.append(dict(role='user', content=content))
        mx.reset_peak_memory()
        prepared = prepare(history)
        mm_input = prepared.multimodal
        assert len(prepared.token_ids) < 24000
        await backend._call(backend._enqueue, prepared, tokenizer, label)
        output = None
        for _ in range(64):
            check()
            result = await backend._call(backend.scheduler.step)
            output = next((x for x in result.outputs if x.finished), None)
            if output is not None:
                break
        assert output is not None and not output.error, 'native request failed'
        trace = output.serving_trace
        C = trace['canonical_frontier']
        assert set(trace['target_offsets']+trace['dspark_offsets']) == {C}
        assert trace['prediction_retired'] and trace['queue_empty'] and trace['cache_published']
        assert trace['prompt_replay'] == trace['full_cache_repack'] == 0
        assert trace['vision_encoded_count'] == int(i is not None)
        assert not mm_input._sources
        cached = trace['cached_tokens']
        assert all(s.start+s.length <= cached for s, old in
                   zip(mm_input.spans, prepared.image_is_historical) if old)
        fields = (*TRACE_FIELDS, 'vision_encoded_count', 'cache_published')
        receipt = {k:v for k,v in trace.items() if k in fields and isinstance(v, (int,float,bool))}
        receipt.update(case=label, aligned=True, total_source_pixels=sum(s.source_pixels for s in mm_input.spans),
            fresh_source_pixels=sum(s.source_pixels for s in mm_input.spans if s.start >= cached),
            images=len(mm_input.spans))
        traces.append(receipt)
        save('requests.json', traces)
        history.append(dict(role='assistant', content=trace['response']['content']))
        sample('settled')
        check()
    async def native():
        await backend._call(backend.load)
        sample('loaded')
        check()
        for i in range(20):
            await run(f'image-{i+1:02}', i)
        await run('cached-20')
        assert traces[-2]['total_source_pixels'] == traces[-1]['total_source_pixels'] == 16955588
        assert traces[-2]['fresh_source_pixels'] == 704000 and traces[-1]['fresh_source_pixels'] == 0
    try:
        asyncio.run(native())
        save('result.json', dict(status='COMPLETED', unsafe=unsafe))
    except BaseException as exc:
        save('result.json', dict(status='STOPPED', exception_type=type(exc).__name__, unsafe=unsafe))
        raise
    finally:
        stop.set()
        watcher.join()
        scheduler = backend.scheduler
        asyncio.run(backend.shutdown_serving())
        assert scheduler is None or (not scheduler.requests and not scheduler.checkpoints._payloads)
        store.clear()
        sample('closed', roots=0, requests=0)
        save('host-after.json', dict(vm=cmd('vm_stat'), swap=cmd('sysctl', 'vm.swapusage'),
            pressure=cmd('memory_pressure', '-Q')))


if __name__ == '__main__':
    def expired(*_):
        raise TimeoutError('finite 1800 second budget')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(1800)
    try:
        main()
    finally:
        signal.alarm(0)
