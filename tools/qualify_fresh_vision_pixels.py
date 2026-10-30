#!/usr/bin/env python3
"""Finite isolated Flash-1017 pixel-boundary acceptance; no admission overrides.

Run with -I in an independently installed candidate. Never changes Production.
"""
import argparse
import asyncio
import base64
from dataclasses import replace
import ctypes
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
from unittest.mock import patch

GiB = 1024**3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (a.output/name).write_text(json.dumps(value, indent=2, default=str)+'\n')
    def command(*args):
        return subprocess.check_output(args, text=True)
    def host():
        return dict(vm=command('vm_stat'), pressure=command('memory_pressure', '-Q'),
                    swap=command('sysctl', 'vm.swapusage'))
    save('host-before.json', host())
    from ds41f_mlx.mtp_identity import config, inspect
    from ds41f_mlx.execution_identity import inspect_payload
    cfg = replace(config(profile='mtp-serving-v1'), kv_root=a.output/'unused-kv')
    identity = inspect(cfg)
    authority, _ = inspect_payload()
    assert authority['checkpoint_variant'] == 'flash-1017'
    import ds41f_mlx
    installed = Path(ds41f_mlx.__file__).parent
    hashes = {}
    for src in (a.source/'ds41f_mlx').rglob('*.py'):
        rel = src.relative_to(a.source/'ds41f_mlx')
        data = (installed/rel).read_bytes()
        assert data == src.read_bytes(), str(rel)
        hashes[str(rel)] = sha256(data).hexdigest()
    save('sources.json', hashes)
    save('identity.json', identity)
    cfg.apply_environment()
    import mlx.core as mx
    from PIL import Image
    from ds41f_mlx.runtime import multimodal as mm
    from ds41f_mlx.serving.server import prepare_request, load_v41_tokenizer
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    backend = ProductionMTPBackend(runtime_config=cfg)
    backend.dependency_identity, backend.glm_profile = identity['identity_sha256'], identity['glm_profile']
    tokenizer = load_v41_tokenizer(cfg.recipe_path)
    lib = ctypes.CDLL('/usr/lib/libproc.dylib')
    case = 'startup'
    stop, unsafe = threading.Event(), []
    lock = threading.Lock()
    def check():
        if unsafe:
            raise RuntimeError('qualification safety stop: '+repr(unsafe))
    def sample(event, **extra):
        buf = ctypes.create_string_buffer(1024)
        assert lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buf)) == 0
        row = dict(t=time.monotonic(), case=case, event=event,
            footprint=ctypes.c_uint64.from_buffer(buf, 72).value,
            lifetime_peak=ctypes.c_uint64.from_buffer(buf, 240).value,
            mlx_active=mx.get_active_memory(), mlx_cache=mx.get_cache_memory(),
            mlx_peak=mx.get_peak_memory(), **extra)
        with lock, (a.output/'samples.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        if max(row['footprint'], row['mlx_active']+row['mlx_cache']) > 360*GiB:
            unsafe.append('360 GiB process/allocator stop')
    def watch():
        try:
            while not stop.wait(2):
                row = host()
                row['t'] = time.monotonic()
                row['reserve_proxy'] = sum(int(re.search(n+r':\s+(\d+)', row['vm'])[1])*16384
                    for n in ('Pages free', 'Pages inactive', 'Pages speculative'))
                with (a.output/'host.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
                if row['reserve_proxy'] < 100*GiB or int(re.search(r'free percentage: (\d+)%', row['pressure'])[1]) < 25:
                    unsafe.append('host reserve/pressure stop')
                if re.search(r'used = (?!0\.00M)\S+', row['swap']):
                    unsafe.append('nonzero swap')
                sample('watch')
        except BaseException as exc:
            unsafe.append('observer failed: '+repr(exc))
    def image(size, color='red'):
        out = BytesIO()
        with Image.new('RGB', size, color) as im:
            im.save(out, format='PNG')
        return dict(type='image', source=dict(type='base64', media_type='image/png',
                                             data=base64.b64encode(out.getvalue()).decode()))
    large, small = image((2048, 2048)), image((512, 512))
    def conversation(blocks):
        return [dict(role='user', content=[*blocks, dict(type='text', text='Reply only OK.')])]
    def prepare(history):
        raw = json.dumps(dict(model=cfg.model_id, max_tokens=16, temperature=0,
            thinking={'type': 'disabled'}, messages=history, stream=False)).encode()
        assert len(raw) < 1024**2
        prepared = prepare_request('messages', raw, tokenizer=tokenizer, recipe_path=cfg.recipe_path,
                                   checkpoint=cfg.checkpoint_path, ordinary=True)
        assert len(prepared.token_ids) < 5000
        return prepared
    try:
        prepare(conversation([large]*5))
    except Exception as exc:
        assert 'patch budget' in str(exc), str(exc)
        save('patch-rejection.json', dict(error=str(exc), model_loaded=False))
    else:
        raise AssertionError('retained patch protection did not reject')
    encoded = 0
    original = mm._encode_span
    def observed(*args, **kwargs):
        nonlocal encoded
        check()
        sample('image:before')
        value = original(*args, **kwargs)
        encoded += 1
        sample('image:after', embedding_bytes=value.nbytes)
        if case == 'injected-vision-failure' and encoded == 2:
            value = None  # do not retain the computed row in the injected traceback
            raise RuntimeError('qualification injected Vision failure')
        return value
    receipts = []
    async def run(label, history, expected_encoded, expected_cached, failure=False):
        nonlocal case, encoded
        case, encoded = label, 0
        check()
        mx.reset_peak_memory()
        sample('prepare:before')
        prepared = prepare(history)
        inp = prepared.multimodal
        sample('prepare:after')
        await backend._call(backend._enqueue, prepared, tokenizer, label)
        final = None
        for _ in range(64):
            check()
            result = await backend._call(backend.scheduler.step)
            final = next((o for o in result.outputs if o.finished), None)
            if final is not None:
                break
        assert final is not None
        trace = final.serving_trace
        assert not inp._sources and encoded == expected_encoded
        assert not backend.scheduler.requests and not backend.scheduler.running
        receipt = dict(case=label, pixels=sum(s.source_pixels for s in inp.spans),
            patches=sum(s.grid[0]*s.grid[1] for s in inp.spans), encoded=encoded,
            cached_tokens=trace.get('cached_tokens'), error=final.error,
            roots=len(backend.scheduler.checkpoints._payloads),
            trace={k: trace.get(k) for k in ('vision_encoded_count', 'cache_published',
                'native_started', 'canonical_settled', 'failure_stage', 'target_offsets',
                'dspark_offsets', 'canonical_frontier', 'prompt_replay', 'full_cache_repack',
                'queue_empty', 'prediction_retired')})
        if failure:
            assert 'qualification injected Vision failure' in final.error
            assert backend.fatal_error and not backend.scheduler.checkpoints._payloads
        else:
            assert not final.error and not backend.fatal_error
            assert encoded == trace['vision_encoded_count']
            assert (trace['cached_tokens'] > 0) == expected_cached
            C = trace['canonical_frontier']
            assert set(trace['target_offsets']+trace['dspark_offsets']) == {C}
            assert trace['queue_empty'] and trace['prediction_retired'] and trace['cache_published']
            assert trace['prompt_replay'] == trace['full_cache_repack'] == 0
        receipts.append(receipt)
        save('requests.json', receipts)
        sample('settled')
        check()
        return trace
    async def native():
        await backend._call(backend.load)
        sample('loaded')
        await run('below-former-both', conversation([small]), 1, False)
        history = conversation([large]*4+[small])
        trace = await run('above-former-pixels-cold', history, 5, False)
        assert receipts[-1]['pixels'] == 17_039_360 > mm.MAX_REQUEST_PIXELS
        assert receipts[-1]['patches'] == 36_117 <= mm.MAX_REQUEST_PATCHES
        history += [dict(role='assistant', content=trace['response']['content']),
                    dict(role='user', content='Reply only OK again.')]
        await run('complete-pair-reuse', history, 0, True)
        await run('injected-vision-failure', conversation([image((2048, 2048), 'blue')]+[large]*3+[small]),
                  2, False, failure=True)
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        with patch.object(mm, '_encode_span', observed):
            asyncio.run(native())
        save('result.json', dict(status='PASS_SCOPED_PIXELS', unsafe=unsafe, overrides=None))
    except BaseException as exc:
        save('result.json', dict(status='STOPPED', error=repr(exc), unsafe=unsafe))
        raise
    finally:
        stop.set()
        watcher.join()
        asyncio.run(backend.shutdown_serving())
        sample('closed')
        save('host-after.json', host())
        backend.close()


if __name__ == '__main__':
    def expired(*_):
        raise TimeoutError('finite 1800 second qualification deadline')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(1800)
    main()
