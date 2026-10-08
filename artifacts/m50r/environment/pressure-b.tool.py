"""M50R audit only: installed public candidate, separate rate and topology runs.

No runtime edits, monkeypatches, extra device synchronization or profiling in
performance mode. Existing candidate telemetry remains enabled in both modes.
Trace counts are Python/native API regions, NOT GPU dispatches or transfer bytes.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
import traceback


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def pressure():
    import psutil
    import mlx.core as mx
    return dict(time=time.time(), rss_bytes=psutil.Process().memory_info().rss,
                available_bytes=psutil.virtual_memory().available,
                swap=psutil.swap_memory()._asdict(), load=list(__import__('os').getloadavg()),
                mlx_active_bytes=mx.get_active_memory(), mlx_cache_bytes=mx.get_cache_memory(),
                mlx_peak_bytes=mx.get_peak_memory(),
                vm_stat=subprocess.check_output(['vm_stat'], text=True))


def dispersion(values):
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                stdev=statistics.stdev(values), min=min(values), max=max(values),
                cv=statistics.stdev(values)/statistics.mean(values))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['identity', 'performance', 'trace'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--repeats', type=int, default=6)
    ap.add_argument('--rehash-weights', action='store_true')
    args = ap.parse_args()
    if args.repeats < 3:
        ap.error('at least three fresh measured sessions required')
    from ds41f_mlx.mtp_identity import config, inspect
    cfg = config()
    identity = inspect(cfg)
    out = dict(schema='ds41f.m50r.audit.v1', mode=args.mode, status='RUNNING',
               head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               tool_sha256=digest(__file__), identity=identity, rows=[],
               scope='bounded installed mtp-singleton-v1; not M50R PASS',
               instrumentation='sys.setprofile worker only' if args.mode == 'trace' else 'none added; candidate built-in telemetry retained')

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2) + '\n')

    save()
    args.output.with_suffix('.tool.py').write_bytes(Path(__file__).read_bytes())
    if args.mode == 'identity':
        import mlx.core as mx
        out['host'] = dict(executable=sys.executable, executable_sha256=digest(sys.executable),
                           python=sys.version, device_info=mx.device_info(),
                           os_build=subprocess.check_output(['sw_vers'], text=True),
                           hardware_model=subprocess.check_output(['sysctl','-n','hw.model'], text=True).strip(),
                           thermal=subprocess.check_output(['pmset','-g','therm'], text=True))
        out['checkpoint_bytes'] = []
        if args.rehash_weights:
            for shard in identity['checkpoint']['shards']:
                path = cfg.checkpoint_path / shard['name']
                before = path.stat()
                actual = digest(path)
                after = path.stat()
                assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
                assert actual == shard['lfs_sha256'], path
                out['checkpoint_bytes'].append(dict(name=path.name, bytes=after.st_size, sha256=actual))
                save()
        out['status'] = 'IDENTITY_CAPTURED'
        out['weight_verification'] = 'fresh SHA256 of all 48 shards' if args.rehash_weights else 'metadata only; byte identity NOT proven'
        save()
        return

    import httpx
    import uvicorn
    from ds41f_mlx.serving.mtp_public import LocalMTPBackend
    from ds41f_mlx.serving.server import create_app
    backend = LocalMTPBackend(runtime_config=cfg)
    backend.dependency_identity = identity['identity_sha256']
    counters = Counter()
    events = []
    active = set()
    start_trace = [0.0]

    def cache_layout(cache):
        # Metadata only. DeepseekV41Cache.size() calls .item(): never add that
        # host/device materialization inside a lazy physical region.
        return [dict(object_id=id(c), compress_ratio=c.compress_ratio,
                     slots=[None if x is None else dict(shape=list(x.shape), dtype=str(x.dtype))
                            for x in c.cache]) for c in cache]

    def state_snapshot(frame):
        st = frame.f_locals.get('state') or frame.f_locals.get('mtp_state')
        gb = frame.f_locals.get('gen_batch')
        rec = frame.f_locals.get('rec')
        value = {}
        if rec is not None and rec.owner is not None:
            gb = rec.owner._bg._generation_batch
            h = getattr(gb, '_omlx_semantic_horizon', None)
            st = st or getattr(gb, '_omlx_mtp_state', None) or (None if h is None else h.last_state)
            if h is not None:
                value['horizon'] = dict(completed=h.completed,
                    pending=None if h.pending is None else repr(h.pending))
            value['guard_finished'] = rec.guard.finished
        if st is not None and hasattr(st, 'queue'):
            value.update(queue=[(int(x[0]), x[2]) for x in st.queue], hist_offset=getattr(st, 'hist_offset', None),
                         next_main_shape=None if getattr(st, 'next_main', None) is None else list(st.next_main.shape),
                         drafts_shape=None if getattr(st, 'drafts', None) is None else list(st.drafts.shape),
                         ring_offsets=[] if st.mtp_cache is None else [c.offset for c in st.mtp_cache],
                         ring_shapes=[] if st.mtp_cache is None else [None if c.keys is None else list(c.keys.shape) for c in st.mtp_cache])
        if gb is not None:
            value['target_layout'] = cache_layout(gb.prompt_cache)
            if gb.prompt_cache:
                stash = getattr(gb.prompt_cache[0], '_mtp_draft_stash', None)
                if stash is not None:
                    value['stash_before'] = stash[2]
                    value['stash_input_shape'] = list(stash[0].shape)
        if rec is not None:
            value.update(canonical_frontier=len(rec.canonical), busy=rec.busy, poisoned=rec.poisoned)
            if rec.owner is not None:
                value['history'] = rec.owner.history.to_json()
        for key in ('k', 'm', 'm_gpu', 'accepted', 'num_drafts', 'backend_terminal', 'before', 'end', 'count'):
            x = frame.f_locals.get(key)
            if isinstance(x, (int, bool)):
                value[key] = x
        inputs = frame.f_locals.get('inputs')
        if inputs is not None and hasattr(inputs, 'shape'):
            value['input_shape'] = list(inputs.shape)
        return value

    boundaries = {'_run_verify_cycle_chain', '_call_backbone_captured', 'dspark_forward',
                  'proposal_forward', 'mtp_partial_rollback', 'canonical_quiesce_native_singleton',
                  '_next', '_settle', '_retire', 'preview', 'bind', 'observe_canonical_emit'}

    def profile(frame, event, arg):
        name = frame.f_code.co_name
        if name == '_run_verify_cycle_chain' and event == 'call':
            active.add(id(frame))
        if active:
            if event == 'call':
                counters[frame.f_code.co_filename.split('site-packages/')[-1] + ':' + name] += 1
                for key in ('inputs', 'x', 'q'):
                    x = frame.f_locals.get(key)
                    if hasattr(x, 'shape'):
                        counters[name + ':' + key + ':' + str(tuple(x.shape))] += 1
            elif event == 'c_call':
                native = 'native:' + str(getattr(arg, '__module__', '')) + ':' + getattr(arg, '__name__', '')
                counters[native] += 1
                owner = getattr(arg, '__self__', None)
                if hasattr(owner, 'shape') and getattr(arg, '__name__', '') in ('tolist', 'item'):
                    counters[native + ':shape:' + str(tuple(owner.shape)) + ':dtype:' + str(owner.dtype)] += 1
        if name in boundaries and event in ('call', 'return'):
            events.append(dict(t=time.perf_counter()-start_trace[0], name=name, event=event,
                               file=frame.f_code.co_filename, state=state_snapshot(frame)))
        if name == '_run_verify_cycle_chain' and event == 'return':
            active.discard(id(frame))

    async def run():
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0)); sock.listen(8)
        actual_cfg = replace(cfg, port=sock.getsockname()[1])
        from ds41f_mlx.serving.local_h11 import LocalH11Protocol
        server = uvicorn.Server(uvicorn.Config(create_app(backend=backend, runtime_config=actual_cfg,
                                profile='mtp-singleton-v1'), lifespan='off', log_level='warning',
                                workers=1, proxy_headers=False, ws='none', loop='asyncio',
                                limit_concurrency=8, backlog=8, timeout_keep_alive=5,
                                h11_max_incomplete_event_size=16384, http=LocalH11Protocol))
        out['server'] = dict(protocol='LocalH11Protocol', workers=1, proxy_headers=False,
                            ws='none', loop='asyncio', concurrency=8, backlog=8,
                            keep_alive_s=5, incomplete_headers_bytes=16384,
                            port=actual_cfg.port, lifespan='off: model explicitly preloaded on sole worker')
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            while not server.started:
                if task.done():
                    await task
                await asyncio.sleep(.01)
            out['before_load'] = pressure()
            t0 = time.perf_counter()
            await backend._call(backend.load)
            out['load_s'] = time.perf_counter()-t0
            out['after_load'] = pressure()
            out['model_config'] = backend._model.language_model._config.__dict__
            save()
            async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{actual_cfg.port}', timeout=600) as client:
                count = args.repeats + 1 if args.mode == 'performance' else 5
                for sample in range(count):
                    cancel = args.mode == 'trace' and sample in (2, 4)
                    cancel_after_frames = 8 if sample == 4 else 1
                    terminal = args.mode == 'trace' and sample == 1
                    # Diagnostic transport delay only, never a rate trial or a
                    # native physical phase modification. Make disconnect drain
                    # observable instead of racing an already completed reply.
                    backend.delivery_delay_s = .03 if cancel else 0.0
                    request = dict(model=cfg.model_id, messages=[
                        dict(role='system', content='A cache stores frequently used data. '*32),
                        dict(role='user', content='What is the weather in Paris? Use lookup_weather.' if terminal else
                             'Write OK exactly forty times separated by spaces and nothing else.')],
                        temperature=0, reasoning_effort='none', max_tokens=64, stream=True)
                    if args.mode == 'trace' and sample == 3:
                        request['messages'][-1]['content'] = (
                            'Write the exact sequence: 91, 17, 83, 29, 61, 43, 97, 11, 71, 37, '
                            '53, 19, 89, 23, 67, 31, 79, 41, 59, 13. No other text.')
                    if terminal:
                        from ds41f_mlx.mtp_profile import WEATHER
                        request.update(tools=[WEATHER], tool_choice=dict(type='function', function=dict(name='lookup_weather')))
                    body = json.dumps(request, separators=(',', ':')).encode()
                    response = await client.post('/v1/sessions', content=b'{}', headers={'Content-Type':'application/json'})
                    response.raise_for_status()
                    sid = response.json()['id']
                    rec = backend.sessions[sid]
                    counters.clear(); events.clear(); active.clear()
                    start_trace[0] = time.perf_counter()
                    if args.mode == 'trace':
                        await backend._call(lambda: sys.setprofile(profile))
                    before = pressure()
                    t0 = time.perf_counter(); first = None; first_token = None; chunks = []
                    async with client.stream('POST', f'/v1/sessions/{sid}/chat/completions', content=body,
                            headers={'Content-Type':'application/json', 'X-DS41F-Request-Sequence':'1'}) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if line.startswith('data: '):
                                if first is None:
                                    first = time.perf_counter()-t0
                                chunks.append(line)
                                if line != 'data: [DONE]':
                                    data = json.loads(line[6:])
                                    if any(c.get('delta', {}).get('content') for c in data.get('choices', [])) and first_token is None:
                                        first_token = time.perf_counter()-t0
                                if cancel and len(chunks) >= cancel_after_frames:
                                    break
                    http_s = time.perf_counter()-t0
                    deadline = time.perf_counter()+120
                    while rec.busy:
                        if time.perf_counter() > deadline:
                            raise TimeoutError('protected settlement did not drain')
                        await asyncio.sleep(.01)
                    trace = dict(rec.last_turn)
                    n = len(trace['canonical_generated'])
                    assert set(trace['target_offsets']) == {trace['canonical_frontier']}
                    assert set(trace['dspark_offsets']) == {trace['canonical_frontier']}
                    assert trace['queue_empty'] and trace['prediction_retired']
                    assert trace['prompt_replay'] == trace['full_cache_repack'] == 0
                    row = dict(sample=sample, warmup=args.mode == 'performance' and sample == 0,
                               cancel_requested=cancel, terminal_workload=terminal,
                               diagnostic_delivery_delay_s=backend.delivery_delay_s,
                               cancel_after_data_frames=cancel_after_frames if cancel else None,
                               request_hex=body.hex(), request_sha256=hashlib.sha256(body).hexdigest(),
                               http_s=http_s, first_sse_s=first, ttft_content_s=first_token,
                               decode_tok_s=n/trace['decode_s'], request_tok_s=n/http_s,
                               settled_decode_s=trace['elapsed_s']-trace['prefill_handoff_s']-trace['load_s'],
                               settled_decode_tok_s=n/(trace['elapsed_s']-trace['prefill_handoff_s']-trace['load_s']),
                               canonical_token_ids=list(rec.canonical),
                               before=before, after=pressure(), trace=trace, chunks=chunks,
                               settled_layout=cache_layout(rec.cache) if args.mode == 'trace' and rec.cache else None,
                               topology=dict(counters) if args.mode == 'trace' else None,
                               causal_events=list(events) if args.mode == 'trace' else None)
                    out['rows'].append(row)
                    if args.mode == 'trace':
                        await backend._call(lambda: sys.setprofile(None))
                    response = await client.delete(f'/v1/sessions/{sid}', headers={'Content-Type':'application/json'})
                    response.raise_for_status()
                    row['retired'] = sid not in backend.sessions
                    save()
                    print(json.dumps(dict(sample=sample, decode_tok_s=row['decode_tok_s'], http_s=http_s)), flush=True)
            if args.mode == 'performance':
                rows = out['rows'][1:]
                out['dispersion'] = {k: dispersion([r[k] for r in rows]) for k in
                                     ('decode_tok_s', 'settled_decode_tok_s', 'http_s', 'request_tok_s', 'ttft_content_s')}
                out['token_identity'] = all(r['trace']['canonical_generated'] == rows[0]['trace']['canonical_generated'] for r in rows)
            out['status'] = 'CAPTURED_NOT_GATE_PASS'
        finally:
            if args.mode == 'trace':
                await backend._call(lambda: sys.setprofile(None))
            server.should_exit = True
            await task
            sock.close()
            backend.close()
            out['after_close'] = pressure()
    try:
        asyncio.run(run())
    except BaseException:
        out['status'] = 'ERROR'
        out['error'] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
