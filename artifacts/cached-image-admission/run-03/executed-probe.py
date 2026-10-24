#!/usr/bin/env python3
"""Isolated installed-runtime resource probe; never a production admission selector.
Run with -I and an independently installed wheel matching --source. No HTTP/Harness
acceptance is claimed. Real Recipe requests use the existing Scheduler/settlement.
"""
import argparse
import asyncio
import base64
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import tracemalloc
from dataclasses import replace
from unittest.mock import patch  # lifecycle observation only; never admission overrides

GiB = 1024**3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--images', type=Path, nargs='+', required=True)
    p.add_argument('--max-images', type=int, default=12)
    p.add_argument('--expected-runtime-commit', default='36fa7b0e39374f700dfbf7c3c62cd26b2ce7e997')
    a = p.parse_args()
    assert 1 <= a.max_images <= 12
    a.output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (a.output/name).write_text(json.dumps(value, indent=2, default=str)+'\n')
    def command(*args):
        return subprocess.check_output(args, text=True)
    save('host-before.json', {k: command(*v) for k, v in {
        'ps': ['ps', '-axo', 'pid,ppid,%cpu,rss,command'],
        'vm': ['vm_stat'], 'pressure': ['memory_pressure'],
        'swap': ['sysctl', 'vm.swapusage'], 'os': ['sw_vers'],
        'hardware': ['sysctl', 'hw.memsize', 'machdep.cpu.brand_string']}.items()})
    # Full installed/checkpoint admission BEFORE numerical imports or backend load.
    from ds41f_mlx.mtp_identity import config, inspect
    from ds41f_mlx.execution_identity import inspect_payload
    cfg = replace(config(profile='mtp-serving-v1'), kv_root=a.output/'kv')
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
    assert authority['provenance']['source_commit'] == a.expected_runtime_commit
    save('identity.json', identity)
    save('sources.json', hashes)
    save('dependencies.txt.json', command(str(Path(os.sys.executable)), '-m', 'pip', 'freeze') if (Path(os.sys.executable).parent/'pip').exists() else command('uv', 'pip', 'freeze', '--python', os.sys.executable))
    cfg.apply_environment()
    import mlx.core as mx
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    from ds41f_mlx.serving.paired_checkpoint import (PairedCheckpoint, PairedCheckpointAuthority,
        pair_bytes, projected_pair_bytes, PATCH_DISCOUNT_PAIR_BYTES, PATCH_DISCOUNT_ROOT_BYTES)
    import ds41f_mlx.runtime.multimodal as mm
    from ds41f_mlx.serving.server import prepare_request, load_v41_tokenizer
    backend = ProductionMTPBackend(runtime_config=cfg)
    backend.dependency_identity = identity['identity_sha256']
    backend.glm_profile = identity['glm_profile']
    lib = ctypes.CDLL('/usr/lib/libproc.dylib')
    stop = threading.Event()
    unsafe = []
    phase = 'admitted'
    case = 'startup'
    samples_lock = threading.Lock()
    def sample(event, **extra):
        buf = ctypes.create_string_buffer(1024)
        rc = lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buf))
        # rusage_info_v4 fixed prefix: uuid, user/system/wakeups/pageins,
        # wired_size, resident_size, phys_footprint (byte offsets 56/64/72).
        nums = {name: ctypes.c_uint64.from_buffer(buf, off).value for name, off in
                [('wired', 56), ('resident', 64), ('footprint', 72), ('lifetime_max_footprint', 240)]} if rc == 0 else {}
        row = dict(t=time.monotonic(), case=case, event=event, phase=phase,
                   mlx_active=mx.get_active_memory(), mlx_cache=mx.get_cache_memory(),
                   mlx_peak=mx.get_peak_memory(), **nums, **extra)
        if tracemalloc.is_tracing():
            row['python_current'], row['python_peak'] = tracemalloc.get_traced_memory()
        if backend.scheduler is not None:
            roots = list(backend.scheduler.checkpoints._payloads.values())
            row.update(roots=len(roots), root_logical_bytes=sum(
                v.nbytes for c in roots for layer in c.target for v in layer.cache if v is not None)
                + sum(r.keys.nbytes for c in roots for r in c.rings if r.keys is not None))
        with samples_lock, (a.output/'samples.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        if max(row.get('footprint', 0), row['mlx_active']+row['mlx_cache']) > 360*GiB:
            unsafe.append('360 GiB process/allocator budget exceeded')
        return row
    def check():
        if unsafe:
            raise RuntimeError('SAFETY STOP: '+repr(unsafe))
    def host_watch():
        while not stop.wait(2):
            vm = command('vm_stat')
            swap = command('sysctl', 'vm.swapusage')
            pressure = command('memory_pressure', '-Q')
            with (a.output/'host.jsonl').open('a') as f:
                f.write(json.dumps(dict(t=time.monotonic(), vm=vm, swap=swap, pressure=pressure))+'\n')
            if re.search(r'used = (?!0\.00M)\S+', swap):
                unsafe.append('nonzero swap')
            pages = lambda name: int(re.search(name+r':\s+(\d+)', vm)[1])*16384
            # Free pages alone fall near zero while macOS reclaims file cache.
            # This is an availability proxy, NOT guaranteed allocatable bytes;
            # retain independent footprint, pressure and zero-swap gates.
            available = sum(pages(name) for name in ('Pages free', 'Pages inactive', 'Pages speculative'))
            if available < 100*GiB:
                unsafe.append('free+inactive+speculative proxy below 100 GiB reserve')
            percent = re.search(r'free percentage: (\d+)%', pressure)
            if percent and int(percent[1]) < 25:
                unsafe.append('memory-pressure free percentage below 25')
    watcher = threading.Thread(target=host_watch, daemon=True)
    watcher.start()
    # Observe existing lifecycle without replacing any authority or array operation.
    def instrument(cls, name):
        original = getattr(cls, name)
        def observed(self, *args, **kwargs):
            nonlocal phase
            previous = phase
            phase = name
            sample(name+':before')
            try:
                result = original(self, *args, **kwargs)
                if name == 'capture' and case.startswith('bounded-'):
                    assert pair_bytes(result) <= PATCH_DISCOUNT_PAIR_BYTES
                return result
            finally:
                sample(name+':after')
                phase = previous
        return patch.object(cls, name, observed)
    from contextlib import ExitStack
    histories = []
    images = [x.read_bytes() for x in a.images]
    save('workload.json', dict(images=[dict(path=str(x), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))
        for x, b in zip(a.images, images)], max_images=a.max_images,
        budgets=dict(process_and_mlx_bytes=360*GiB, free_reserve_bytes=100*GiB,
                     max_prompt=32768, output_tokens=64, total_seconds=1800),
        test_override=None))
    traces = []
    async def run_request(history, label, cancel=False, expect_error=False, running_cancel=False):
        nonlocal case, phase
        case, phase = label, 'preparation'
        check()
        body = dict(model=cfg.model_id, max_tokens=64, temperature=0,
                    thinking={'type':'disabled'}, stream=False, messages=history)
        raw = json.dumps(body, separators=(',', ':')).encode()
        assert len(raw) < 8*1024**2
        mx.reset_peak_memory()
        before = sample('prepare:before', body_bytes=len(raw))
        prepared = prepare_request('messages', raw, tokenizer=tokenizer,
            recipe_path=cfg.recipe_path, checkpoint=cfg.checkpoint_path, ordinary=True)
        assert mm.MAX_REQUEST_PATCHES == 36864 and mm.MAX_REQUEST_PIXELS == 16777216
        assert len(prepared.token_ids) <= 32768
        sample('prepare:after', prompt_tokens=len(prepared.token_ids),
               identities=list(prepared.multimodal.identities()) if prepared.multimodal else [],
               historical=prepared.image_is_historical)
        rid = label
        await backend._call(backend._enqueue, prepared, tokenizer, rid)
        if running_cancel:
            await backend._call(backend.scheduler._schedule_waiting)
            req = backend.scheduler.requests[rid]
            assert req.native_started and rid in backend.scheduler.running
            sample('native-cancel:before')
            await backend._call(backend.scheduler._do_abort_request, rid)
            assert not backend.scheduler.requests and not prepared.multimodal._sources
            trace = req.trace
            assert set(trace['target_offsets'] + trace['dspark_offsets']) == {trace['canonical_frontier']}
            assert trace['prediction_retired'] and trace['queue_empty'] and trace['cache_published']
            if label == 'bounded-native-cancel-seven':
                assert trace['vision_encoded_count'] == 7
                assert trace['historical_patch_discount']
            assert not backend.fatal_error
            save(label+'-trace.json', trace)
            backend.scheduler._completed.clear()
            sample('native-cancel:after')
            return
        if cancel:
            await backend._call(backend.scheduler._do_abort_request, rid)
            assert not backend.scheduler.requests
            assert not prepared.multimodal._sources
            span = prepared.multimodal.spans[0]
            try:
                prepared.multimodal.validate_prefix(span.start+1, [])
            except ValueError as exc:
                assert 'intersects' in str(exc)
            else:
                raise AssertionError('partial span accepted')
            backend.scheduler._completed.clear()
            sample('queued-cancel:after')
            return
        output = None
        for _ in range(256):
            check()
            result = await backend._call(backend.scheduler.step)
            output = next((x for x in result.outputs if x.finished), None)
            if output is not None:
                break
        assert output is not None, 'finite step budget exhausted'
        trace = output.serving_trace
        save(label+'-trace.json', trace)
        traces.append(dict(case=label, error=output.error, before=before))
        if expect_error:
            assert output.error and not trace.get('cache_published'), trace
            expected = 'paired-byte qualification fallback' if label.endswith('byte-fallback') else 'historical image pair unavailable'
            assert expected in output.error, output.error
            assert not trace.get('historical_patch_discount') and not trace.get('cached_tokens')
            assert not backend.fatal_error, backend.fatal_error
        else:
            assert not output.error, output.error
            C = trace['canonical_frontier']
            assert set(trace['target_offsets']+trace['dspark_offsets']) == {C}
            assert trace['cache_published'] and trace['prompt_replay'] == trace['full_cache_repack'] == 0
            identities = list(prepared.multimodal.identities()) if prepared.multimodal else []
            cached = trace['cached_tokens']
            assert trace.get('vision_encoded_count', 0) == sum(s['start']+s['length'] > cached for s in identities)
            assert all(not old or s['start']+s['length'] <= cached for old, s in zip(prepared.image_is_historical or (), identities))
            response = trace['response']
            history.append(dict(role='assistant', content=response['content']))
        del prepared, result, output, trace
        gc.collect()
        sample('request:clean')
        check()
    async def campaign():
        nonlocal case, tokenizer
        await backend._call(backend.load)
        sample('model:loaded')
        check()
        tracemalloc.start(1)
        tokenizer = load_v41_tokenizer(cfg.recipe_path)
        for n in range(1, a.max_images+1):
            check()
            histories.append(dict(role='user', content=[dict(type='text', text=f'Screenshot observation {n}. State the dashboard title only, briefly.'),
                dict(type='image', source=dict(type='base64', media_type='image/png', data=base64.b64encode(images[(n-1)%len(images)]).decode()))]))
            if n > 7:
                margin = sample('extended-case:headroom')
                assert margin.get('footprint', 360*GiB) < 320*GiB, 'insufficient extended-case headroom'
                assert margin['mlx_active']+margin['mlx_cache'] < 320*GiB
            await run_request(histories, f'history-{n:02}')
            if n in (7, 10, 12):
                histories.append(dict(role='user', content='What is the latest dashboard title? Answer briefly.'))
                await run_request(histories, f'cached-{n:02}')
        # Real queued cancellation releases source ownership, without native work.
        pending = histories+[dict(role='user', content='Briefly repeat the title.')]
        await run_request(pending, 'queued-cancel', cancel=True)
        await run_request(pending, 'running-cancel', running_cancel=True)
        # Exercise the byte boundary and longest suffix with seven fresh images
        # after a legitimate single-image cold seed. An old ordinary tail hint
        # is not guaranteed to survive the twelve-image run; do not republish or
        # relabel old state to force a hit. This is an explicit fresh conversation.
        # Padding follows executable packed-array bytes, not a context policy.
        import copy
        prefix = [copy.deepcopy(histories[0])]
        await run_request(prefix, 'boundary-cold-seed')
        extra = [dict(type='image', source=dict(type='base64', media_type='image/png',
            data=base64.b64encode(images[i % len(images)]).decode())) for i in range(7)]
        root = max(backend.scheduler.checkpoints._payloads.values(), key=lambda c: len(c.tokens))
        lo, hi = len(root.tokens), 32768
        while lo + 1 < hi:
            mid = (lo + hi)//2
            if projected_pair_bytes(root, mid) <= PATCH_DISCOUNT_PAIR_BYTES:
                lo = mid
            else:
                hi = mid
        # Leave bounded continuation headroom while approaching the byte ceiling.
        desired = lo - 64 - 5 - 256
        def branch(words, label):
            return copy.deepcopy(prefix) + [dict(role='user', content=[
                dict(type='text', text=label+' '+(' x'*words))] + copy.deepcopy(extra) +
                [dict(type='text', text=' State the dashboard title only.')])]
        trial = branch(0, 'Resource boundary A')
        trial_body = json.dumps(dict(model=cfg.model_id, max_tokens=64, temperature=0,
            thinking={'type':'disabled'}, stream=False, messages=trial)).encode()
        trial_prepared = prepare_request('messages', trial_body, tokenizer=tokenizer,
            recipe_path=cfg.recipe_path, checkpoint=cfg.checkpoint_path, ordinary=True)
        words = desired - len(trial_prepared.token_ids)
        assert words > 0
        trial_prepared.multimodal.release_sources()
        del trial_prepared
        boundary = branch(words, 'Resource boundary A')
        await run_request(boundary, 'bounded-native-cancel-seven', running_cancel=True)
        # Explicit no-tool cancellation re-entry: same canonical prompt pair,
        # no historical replay. This is not Runtime retry/recovery machinery.
        await run_request(boundary, 'bounded-cancel-reentry')
        excess = copy.deepcopy(boundary) + [dict(role='user', content=' x'*1024)]
        await run_request(excess, 'pair-byte-fallback', expect_error=True)
        # Populate distinct real native continuation roots, retaining their
        # actual functional-array costs. No synthetic roots/sharing assumption.
        closest = 0
        for i in range(18):
            roots = sum(pair_bytes(c) for c in backend.scheduler.checkpoints._payloads.values())
            closest = max(closest, roots)
            if roots > PATCH_DISCOUNT_ROOT_BYTES:
                await run_request(copy.deepcopy(boundary) + [dict(role='user', content='Repeat title.')],
                                  'root-byte-fallback', expect_error=True)
                break
            boundary.append(dict(role='user', content='Repeat title.'))
            await run_request(boundary, f'bounded-roots-{i:02}')
        else:
            raise AssertionError('retained-byte fallback not reached')
        save('byte-boundary.json', dict(projected_end=lo, desired_prompt=desired,
            closest_retained_bytes=closest, pair_limit=PATCH_DISCOUNT_PAIR_BYTES,
            root_limit=PATCH_DISCOUNT_ROOT_BYTES))
        # A changed historical image must miss safely; never encode/replay history.
        import copy
        changed = copy.deepcopy(pending)
        changed[0]['content'][1]['source']['data'] = base64.b64encode(images[1]).decode()
        await run_request(changed, 'identity-miss', expect_error=True)
        # Existing roots cleared by their sole authority; old bytes cannot recover KV.
        await backend._call(backend.scheduler.checkpoints.clear)
        sample('cache:cleared')
        await run_request(pending, 'historical-miss', expect_error=True)
        histories.clear()
        gc.collect()
        sample('history:released')
    tokenizer = None
    try:
        with ExitStack() as observations:
            for cls, names in [(PairedCheckpointAuthority, ['acquire', 'capture', 'publish']),
                               (PairedCheckpoint, ['restore']),
                               (ProductionMTPBackend, ['_encode_new_images', '_start', '_settle', '_retire'])]:
                for name in names:
                    observations.enter_context(instrument(cls, name))
            asyncio.run(campaign())
        save('result.json', dict(status='COMPLETED', traces=traces, unsafe=unsafe))
    except BaseException as exc:
        save('result.json', dict(status='STOPPED', error=repr(exc), traces=traces, unsafe=unsafe))
        raise
    finally:
        stop.set()
        watcher.join()
        # Retire even a safety-stopped active/queued request through its existing
        # owner before executor shutdown; never use process exit as settlement.
        retired_scheduler = backend.scheduler
        asyncio.run(backend.shutdown_serving())
        roots = 0 if retired_scheduler is None else len(retired_scheduler.checkpoints._payloads)
        assert roots == 0
        assert retired_scheduler is None or not retired_scheduler.requests
        gc.collect()
        sample('backend:closed', roots=roots)
        save('host-after.json', dict(vm=command('vm_stat'), swap=command('sysctl', 'vm.swapusage'), pressure=command('memory_pressure')))


if __name__ == '__main__':
    import signal
    def expired(*_):
        raise TimeoutError('finite 1800 second test budget exhausted')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(1800)
    try:
        main()
    finally:
        signal.alarm(0)
