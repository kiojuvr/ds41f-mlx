#!/usr/bin/env python3
"""Finite isolated installed Messages probe. No admission overrides or tool calls."""
import argparse
import asyncio
import base64
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from dataclasses import replace
from unittest.mock import patch
from contextlib import ExitStack

GiB = 1024**3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False, parents=True)
    def save(name, value):
        (a.output/name).write_text(json.dumps(value, indent=2)+'\n')
    def cmd(*args):
        return subprocess.check_output(args, text=True)
    save('host-before.json', dict(vm=cmd('vm_stat'), swap=cmd('sysctl', 'vm.swapusage'),
        pressure=cmd('memory_pressure', '-Q'), hardware=cmd('sysctl', 'hw.memsize')))
    # Installed/checkpoint admission precedes numerical imports or model load.
    from ds41f_mlx.mtp_identity import config, inspect
    from ds41f_mlx.execution_identity import inspect_payload
    cfg = replace(config(profile='mtp-serving-v1'), kv_root=a.output/'unused-kv')
    identity = inspect(cfg)
    authority, _ = inspect_payload()
    import ds41f_mlx
    installed = Path(ds41f_mlx.__file__).parent
    hashes = {}
    for src in (a.source/'ds41f_mlx').rglob('*.py'):
        rel = src.relative_to(a.source/'ds41f_mlx')
        data = (installed/rel).read_bytes()
        assert data == src.read_bytes(), str(rel)
        hashes[str(rel)] = hashlib.sha256(data).hexdigest()
    save('sources.json', hashes)
    save('identity.json', identity)
    save('provenance.json', authority['provenance'])
    cfg.apply_environment()
    import mlx.core as mx
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    from ds41f_mlx.serving.paired_checkpoint import PairedCheckpointAuthority, pair_bytes
    from ds41f_mlx.serving.server import prepare_request, load_v41_tokenizer
    from ds41f_mlx.observability import TRACE_FIELDS
    backend = ProductionMTPBackend(runtime_config=cfg)
    backend.dependency_identity = identity['identity_sha256']
    backend.glm_profile = identity['glm_profile']
    lib = ctypes.CDLL('/usr/lib/libproc.dylib')
    unsafe, stop, lock = [], threading.Event(), threading.Lock()
    case = 'startup'
    def sample(event, **extra):
        buf = ctypes.create_string_buffer(1024)
        assert lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buf)) == 0
        footprint = ctypes.c_uint64.from_buffer(buf, 72).value
        row = dict(t=time.monotonic(), case=case, event=event, footprint=footprint,
            lifetime_max_footprint=ctypes.c_uint64.from_buffer(buf, 240).value,
            mlx_active=mx.get_active_memory(), mlx_cache=mx.get_cache_memory(),
            mlx_peak=mx.get_peak_memory(), **extra)
        if backend.scheduler is not None:
            roots = backend.scheduler.checkpoints._payloads.values()
            row.update(roots=len(backend.scheduler.checkpoints._payloads),
                       retained_logical_bytes=sum(pair_bytes(c) for c in roots))
        with lock, (a.output/'samples.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        if max(footprint, row['mlx_active']+row['mlx_cache']) > 360*GiB:
            unsafe.append('360 GiB process/allocator ceiling')
        return row
    def check():
        assert not unsafe, unsafe
    def watch():
        try:
            while not stop.wait(2):
                sample('watch')
                vm, swap, pressure = cmd('vm_stat'), cmd('sysctl', 'vm.swapusage'), cmd('memory_pressure', '-Q')
                pages = lambda name: int(re.search(name+r':\s+(\d+)', vm)[1])*16384
                reserve = sum(pages(n) for n in ('Pages free', 'Pages inactive', 'Pages speculative'))
                with (a.output/'host.jsonl').open('a') as f:
                    f.write(json.dumps(dict(t=time.monotonic(), vm=vm, swap=swap,
                        pressure=pressure, reserve_proxy=reserve))+'\n')
                if reserve < 100*GiB:
                    unsafe.append('100 GiB host reserve proxy')
                if re.search(r'used = (?!0\.00M)\S+', swap):
                    unsafe.append('nonzero swap')
                if int(re.search(r'free percentage: (\d+)%', pressure)[1]) < 25:
                    unsafe.append('pressure below 25 percent')
        except BaseException:
            unsafe.append('resource observer failed')
    def observe(cls, name):
        original = getattr(cls, name)
        def wrapped(self, *args, **kwargs):
            sample(name+':before')
            check()
            result = original(self, *args, **kwargs)
            extra = {}
            if name == 'capture':
                extra = dict(pair_logical_bytes=pair_bytes(result), frontier=len(result.tokens),
                    aligned=all(c.size() == len(result.tokens) for c in result.target)
                        and all(r.offset == len(result.tokens) for r in result.rings))
            sample(name+':after', **extra)
            return result
        return patch.object(cls, name, wrapped)
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    image = base64.b64encode(a.image.read_bytes()).decode()
    history, traces = [], []
    async def run(label, text, add_image=False):
        nonlocal case
        case = label
        check()
        content = [dict(type='text', text=text)]
        if add_image:
            content.append(dict(type='image', source=dict(type='base64', media_type='image/png', data=image)))
        history.append(dict(role='user', content=content))
        raw = json.dumps(dict(model=cfg.model_id, max_tokens=16, temperature=0,
            thinking={'type':'disabled'}, stream=False, messages=history)).encode()
        assert len(raw) < 8*1024**2
        mx.reset_peak_memory()
        prepared = prepare_request('messages', raw, tokenizer=tokenizer,
            recipe_path=cfg.recipe_path, checkpoint=cfg.checkpoint_path, ordinary=True)
        assert len(prepared.token_ids) < 28000
        sample('prepared', prompt_frontier=len(prepared.token_ids))
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
        assert trace['cache_published'] and trace['prediction_retired'] and trace['queue_empty']
        assert trace['prompt_replay'] == trace['full_cache_repack'] == 0
        assert trace.get('vision_encoded_count', 0) == int(add_image)
        assert not prepared.multimodal._sources
        fields = (*TRACE_FIELDS, 'vision_encoded_count', 'cache_published', 'prediction_retired', 'queue_empty')
        receipt = {k: v for k, v in trace.items() if k in fields and isinstance(v, (int, float, bool))}
        receipt.update(aligned=True, case=label)
        traces.append(receipt)
        save('requests.json', traces)
        history.append(dict(role='assistant', content=trace['response']['content']))
        sample('settled')
        check()
        return trace
    async def qualify():
        nonlocal tokenizer
        await backend._call(backend.load)
        sample('loaded')
        check()
        tokenizer = load_v41_tokenizer(cfg.recipe_path)
        for n in range(8):
            await run(f'image-{n+1}', 'State the dashboard title only, briefly.', True)
        # Genuine short appends, not fabricated roots or a large cold suffix.
        for n in range(12):
            trace = await run(f'continuation-{n+1}', ' x'*1800+' State the dashboard title only.')
            if len(history) > 0 and trace['canonical_frontier'] >= 25000:
                break
        assert trace['historical_planned_patches'] > 36864
        assert trace['legacy_pair_bytes_exceeded']
        assert trace['selected_pair_frontier'] > 0
        assert 25000 <= trace['canonical_frontier'] < 28000
    tokenizer = None
    try:
        with ExitStack() as observations:
            for cls, names in [(PairedCheckpointAuthority, ('acquire', 'capture', 'publish')),
                               (ProductionMTPBackend, ('_encode_new_images', '_start', '_settle'))]:
                for name in names:
                    observations.enter_context(observe(cls, name))
            asyncio.run(qualify())
        save('result.json', dict(status='COMPLETED', unsafe=unsafe))
    except BaseException as exc:
        save('result.json', dict(status='STOPPED', exception_type=type(exc).__name__, unsafe=unsafe))
        raise
    finally:
        stop.set()
        watcher.join()
        scheduler = backend.scheduler
        asyncio.run(backend.shutdown_serving())
        assert scheduler is None or (not scheduler.checkpoints._payloads and not scheduler.requests)
        sample('closed')
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
