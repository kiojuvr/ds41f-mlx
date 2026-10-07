"""Bounded, opt-in observer of the admitted MTP HTTP runtime (never a backend).

Run with the normally sealed MTP interpreter. No environment/admission bypass.
Metadata/route observation is limited to N verify cycles; route host transfers and
microbenchmarks happen AFTER server shutdown. --sync-phases is a separate,
perturbing diagnostic, not an end-to-end performance run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
import dataclasses
import functools
import inspect
import json
from pathlib import Path
import subprocess
import time


def occupancy(indices, experts):
    counts = Counter(indices)
    if any(i < 0 or i >= experts for i in counts):
        raise ValueError('expert index outside inventory')
    return dict(routes=len(indices), active_experts=len(counts), experts=experts,
                rows_per_active_expert_histogram=dict(sorted(Counter(counts.values()).items())),
                max_rows=max(counts.values(), default=0),
                mean_rows_per_active_expert=len(indices) / len(counts) if counts else 0,
                singleton_route_fraction=sum(n for n in counts.values() if n == 1) / len(indices) if indices else 0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--port', type=int)
    p.add_argument('--r1-output', type=Path,
                   help='launch unchanged R1 MTP fixture, substituting only its server command with this observer')
    p.add_argument('--cycles', type=int, default=8)
    p.add_argument('--sync-phases', action='store_true')
    p.add_argument('--microbench', action='store_true')
    p.add_argument('--export-operands', action='store_true',
                   help='export bounded real routed/dense/head and up to two packed-attention operand sets for isolated capture')
    a = p.parse_args(argv)
    import os
    if os.environ.get('MTL_CAPTURE_ENABLED') not in (None, '0'):
        p.error('capture-layer overhead is unsuitable here; use tools.capture_mtp_operands separately')
    if a.export_operands and not a.microbench:
        p.error('--export-operands requires --microbench')
    if not 1 <= a.cycles <= 32:
        p.error('cycles must be 1..32')
    if a.r1_output:
        import sys
        from reference.R1.mtp_model import main as r1_main
        original_popen = subprocess.Popen
        def launch(command, *args, **kwargs):
            if command[:4] == [sys.executable, '-m', 'ds41f_mlx.ops', 'start']:
                port = command[command.index('--port')+1]
                command = [sys.executable, '-m', 'tools.profile_mtp_verification',
                           '--port', port, '--output', str(a.output), '--cycles', str(a.cycles)]
                for flag, enabled in (('--sync-phases', a.sync_phases),
                                      ('--microbench', a.microbench),
                                      ('--export-operands', a.export_operands)):
                    if enabled:
                        command.append(flag)
            return original_popen(command, *args, **kwargs)
        subprocess.Popen = launch
        try:
            return r1_main(['--output', str(a.r1_output)])
        finally:
            subprocess.Popen = original_popen
    if a.port is None:
        p.error('--port is required unless --r1-output is used')
    import mlx.core as mx
    from omlx.patches.mlx_lm_mtp import batch_generator as bg
    from omlx.patches.deepseek_v41 import language, mtp
    from ds41f_mlx.serve import main as serve
    from ds41f_mlx import prefill_fp8_mlx as prefill_module
    from ds41f_mlx.prefill_fp8_mlx import DeferredPrefillAppend
    from ds41f_mlx.runtime.mtp_lifecycle import DSparkCommittedContext, OMLXMTPGenerationSession
    # These are the same modules the sealed runtime normally imports.
    result = dict(schema='ds41f.mtp.compute-observer.v1', status='RUNNING',
                  sync_phases=a.sync_phases, cycles=[], operations=[], routes=[], microbench=[], prefill=[])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    phase = ContextVar('phase', default=None)
    cycle = ContextVar('cycle', default=None)
    recent_prefill = ContextVar('recent_prefill', default=None)
    pending_routes = []
    samples = {}
    undo = []

    def patch(host, name, wrapper):
        original = getattr(host, name)
        descriptor = inspect.getattr_static(host, name)
        setattr(host, name, wrapper(original))
        undo.append((host, name, descriptor))

    @contextmanager
    def scope(name):
        token = phase.set(name)
        row = cycle.get()
        wait_start = time.perf_counter()
        if row is not None and a.sync_phases:
            mx.synchronize()
        prior_wait = time.perf_counter()-wait_start
        t = time.perf_counter()
        try:
            yield
        finally:
            if row is not None:
                if a.sync_phases:
                    mx.synchronize()
                row.setdefault('phases', []).append(dict(name=name, wall_s=time.perf_counter()-t,
                    prior_wait_s=prior_wait, timing='synchronized diagnostic' if a.sync_phases else 'host enqueue only'))
            phase.reset(token)

    def cycle_wrapper(original):
        @functools.wraps(original)
        def wrapped(batch, state, *args, **kwargs):
            if len(result['cycles']) >= a.cycles:
                return original(batch, state, *args, **kwargs)
            row = dict(index=len(result['cycles']), depth=int(state.drafts.shape[0]),
                       stats_before=dataclasses.asdict(state.stats))
            result['cycles'].append(row)
            token = cycle.set(row)
            t = time.perf_counter()
            try:
                return original(batch, state, *args, **kwargs)
            finally:
                row['wall_s'] = time.perf_counter()-t
                row['stats_after'] = dataclasses.asdict(state.stats)
                cycle.reset(token)
        return wrapped

    def phase_wrapper(name):
        def decorate(original):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                if name == 'target_verify' and cycle.get() is not None:
                    cycle.get()['backbone_input_shape'] = list(args[1].shape)
                with scope(name):
                    out = original(*args, **kwargs)
                    if cycle.get() is not None and a.sync_phases:
                        if isinstance(out, tuple):
                            mx.eval([v for v in out[:2] if isinstance(v, mx.array)])
                        elif isinstance(out, mx.array):
                            mx.eval(out)
                    return out
            return wrapped
        return decorate

    def gate_wrapper(original):
        @functools.wraps(original)
        def wrapped(self, x, *args, **kwargs):
            out = original(self, x, *args, **kwargs)
            row = cycle.get()
            if row is not None:
                pending_routes.append((dict(cycle=row['index'], phase=phase.get(),
                    input_shape=list(x.shape), experts=self._config.n_routed_experts), out[0]))
            return out
        return wrapped

    def operation_wrapper(name):
        def decorate(original):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                row = cycle.get()
                if row is not None:
                    shapes = [list(v.shape) if isinstance(v, mx.array) else None for v in args]
                    metadata = dict(cycle=row['index'], phase=phase.get(), operation=name,
                        shapes=shapes, dtypes=[str(v.dtype) if isinstance(v, mx.array) else None for v in args],
                        kwargs={k:(list(v.shape) if isinstance(v, mx.array) else v) for k,v in kwargs.items()})
                    result['operations'].append(metadata)
                    # One real input per signature, bounded to six shapes. Nothing
                    # is copied or evaluated on the observed execution path.
                    key = (name, phase.get(), tuple(tuple(v.shape) if isinstance(v, mx.array) else None for v in args),
                           kwargs.get('mode'), kwargs.get('bits'), kwargs.get('group_size'))
                    if a.microbench and key not in samples:
                        candidates = [(k,v) for k,v in samples.items() if v[3]['operation'] == name]
                        if name == 'quantized_matmul':
                            # Include the vocabulary head, not just the first
                            # small attention projections. Keep one real row
                            # shape per packed weight geometry, largest first.
                            geometry = tuple(args[1].shape)
                            if not any(tuple(v[1][1].shape) == geometry for _,v in candidates):
                                if len(candidates) < 3:
                                    samples[key] = (original, args, kwargs, metadata)
                                else:
                                    smallest_key, smallest = min(candidates, key=lambda kv: kv[1][1][1].size)
                                    if args[1].size > smallest[1][1].size:
                                        del samples[smallest_key]
                                        samples[key] = (original, args, kwargs, metadata)
                        elif name == 'packed_attention':
                            count = args[3].shape[-1]+args[4].shape[-1]
                            bucket = count >= 512
                            peers = [(k,v) for k,v in candidates if
                                     (v[1][3].shape[-1]+v[1][4].shape[-1] >= 512) == bucket]
                            if not peers:
                                samples[key] = (original, args, kwargs, metadata)
                            elif count > peers[0][1][1][3].shape[-1]+peers[0][1][1][4].shape[-1]:
                                del samples[peers[0][0]]
                                samples[key] = (original, args, kwargs, metadata)
                        elif len(candidates) < (1 if name == 'bf16_head' else 3):
                            samples[key] = (original, args, kwargs, metadata)
                return original(*args, **kwargs)
            return wrapped
        return decorate

    def draft_wrapper(original):
        @functools.wraps(original)
        def wrapped(batch, state, *args, **kwargs):
            with scope('draft_total'):
                out = original(batch, state, *args, **kwargs)
                if cycle.get() is not None and a.sync_phases:
                    mx.eval(state.drafts)
                return out
        return wrapped

    def prefill_wrapper(original):
        @functools.wraps(original)
        def wrapped(app, *args, **kwargs):
            t = time.perf_counter()
            out = original(app, *args, **kwargs)
            if len(result['prefill']) < 32:
                row = dict(wall_s=time.perf_counter()-t, C=app.C,
                           request_history_rows=len(app.request_token_history))
                runner = getattr(getattr(app, 'final_execution', None), 'runner', None)
                telemetry = getattr(getattr(runner, 'p8_optimizer', None), 'telemetry', None)
                if telemetry is not None:
                    row['existing_telemetry'] = telemetry.to_json()
                result['prefill'].append(row)
                recent_prefill.set((row, time.perf_counter()))
            return out
        return wrapped

    def context_ready_wrapper(original):
        def wrapped(cls, *args, **kwargs):
            recent = recent_prefill.get()
            t = time.perf_counter()
            first = recent is not None and 'post_prefix_to_context_ready_s' not in recent[0]
            if first:
                recent[0]['post_prefix_to_context_ready_s'] = t-recent[1]
            out = original(*args, **kwargs)
            if first:
                recent[0]['context_certificate_s'] = time.perf_counter()-t
            return out
        return classmethod(wrapped)

    def handoff_wrapper(original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            recent = recent_prefill.get()
            t = time.perf_counter()
            if recent is not None:
                recent[0]['post_prefix_to_handoff_entry_s'] = t-recent[1]
            out = original(*args, **kwargs)
            if recent is not None:
                recent[0]['handoff_total_s'] = time.perf_counter()-t
            return out
        return wrapped

    def bootstrap_wrapper(original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            t = time.perf_counter()
            out = original(*args, **kwargs)
            recent = recent_prefill.get()
            if recent is not None:
                recent[0]['terminal_bootstrap_s'] = time.perf_counter()-t
            return out
        return wrapped

    patch(DeferredPrefillAppend, 'execute_all', prefill_wrapper)
    patch(DSparkCommittedContext, 'from_native', context_ready_wrapper)
    patch(prefill_module, 'handoff_to_generation', handoff_wrapper)
    patch(OMLXMTPGenerationSession, 'start', bootstrap_wrapper)
    patch(bg, '_run_verify_cycle_chain', cycle_wrapper)
    patch(bg, '_dspark_next_drafts', draft_wrapper)
    patch(bg, '_call_backbone_captured', phase_wrapper('target_verify'))
    if a.sync_phases:
        # This deliberately serialized diagnostic is never the production
        # throughput receipt. Compare outer verify timing to the plain observer.
        patch(language.Attention, '__call__', phase_wrapper('attention'))
        patch(language.MoE, '__call__', phase_wrapper('moe'))
    patch(mtp.DSparkMixin, 'dspark_forward', phase_wrapper('draft'))
    patch(mtp.DSparkMixin, 'mtp_partial_rollback', phase_wrapper('rollback'))
    patch(language.Gate, '__call__', gate_wrapper)
    for name in ('quantized_matmul', 'gather_qmm'):
        patch(mx, name, operation_wrapper(name))
    patch(language, 'project_logits', operation_wrapper('bf16_head'))
    patch(language, 'packed_sparse_attention', operation_wrapper('packed_attention'))
    # Uvicorn replays captured termination signals after its graceful drain.
    # Its usual default SIGTERM action would kill an external observer before
    # deferred route export. Keep the drain unchanged and let its replay return.
    import signal
    previous_term = signal.signal(signal.SIGTERM, lambda *_: None)
    try:
        code = serve(['--profile', 'mtp-singleton-v1', '--port', str(a.port)])
        result['server_returncode'] = code
        for metadata, indices in pending_routes:
            flat = indices.reshape(-1).tolist()
            result['routes'].append(dict(metadata, **occupancy(flat, metadata['experts'])))
        if a.export_operands:
            exported = []
            candidates = list(samples.values())
            routed = [v for v in candidates if v[3]['operation'] == 'gather_qmm'][:1]
            dense = sorted((v for v in candidates if v[3]['operation'] == 'quantized_matmul'),
                           key=lambda v: v[1][1].size, reverse=True)[:3]
            heads = [v for v in candidates if v[3]['operation'] == 'bf16_head'][:1]
            attention = [v for v in candidates if v[3]['operation'] == 'packed_attention'][:2]
            for index, (_, args, kwargs, metadata) in enumerate(routed+dense+heads+attention):
                tensors = {}
                def encode(v, name):
                    if isinstance(v, mx.array):
                        tensors[name] = v
                        return {'array': name}
                    return v
                description = dict(operation=metadata['operation'], source=metadata,
                    args=[encode(v, f'arg{i}') for i,v in enumerate(args)],
                    kwargs={k:encode(v, f'kw_{k}') for k,v in kwargs.items()})
                base = a.output.parent / f'{a.output.stem}-operands{index}'
                mx.save_safetensors(str(base.with_suffix('.safetensors')), tensors)
                base.with_suffix('.json').write_text(json.dumps(description, indent=2)+'\n')
                exported.append(str(base.with_suffix('.json')))
            result['exported_operands'] = exported
        sweep_done = False
        route_probe_done = False
        for original, args, kwargs, metadata in samples.values():
            mx.eval([v for v in (*args, *kwargs.values()) if isinstance(v, mx.array)])
            for _ in range(3):
                mx.eval(original(*args, **kwargs)); mx.synchronize()
            times = []
            for _ in range(10):
                t = time.perf_counter()
                mx.eval(original(*args, **kwargs)); mx.synchronize()
                times.append(time.perf_counter()-t)
            bench = dict(metadata, samples_s=times,
                scope='isolated replay of real operands; not additive end-to-end attribution')
            result['microbench'].append(bench)
            if (not route_probe_done and metadata['operation'] == 'gather_qmm'
                and kwargs.get('rhs_indices') is not None):
                ids = kwargs['rhs_indices'].reshape(-1)
                flat_x = args[0].reshape(-1, args[0].shape[-1])
                if ids.size % flat_x.shape[0] == 0:
                    route_probe_done = True
                    t = time.perf_counter()
                    order = mx.argsort(ids)
                    inverse = mx.argsort(order)
                    lhs = kwargs.get('lhs_indices')
                    lhs = (lhs.reshape(-1) if lhs is not None else
                           mx.arange(ids.size) // (ids.size // flat_x.shape[0]))
                    sorted_x = flat_x[lhs[order]][:, None, :]
                    sorted_ids = ids[order]
                    mx.eval(sorted_x, sorted_ids, inverse); mx.synchronize()
                    sort_s = time.perf_counter()-t
                    probe_kwargs = dict(kwargs, rhs_indices=sorted_ids, sorted_indices=True)
                    probe_kwargs.pop('lhs_indices', None)
                    probe_args = (sorted_x, *args[1:])
                    for _ in range(3):
                        mx.eval(original(*probe_args, **probe_kwargs)); mx.synchronize()
                    elapsed = []
                    for _ in range(10):
                        t = time.perf_counter()
                        mx.eval(original(*probe_args, **probe_kwargs)); mx.synchronize()
                        elapsed.append(time.perf_counter()-t)
                    expected = original(*args, **kwargs).reshape(ids.size, -1)
                    actual = original(*probe_args, **probe_kwargs)[inverse].reshape(ids.size, -1)
                    error = float(mx.max(mx.abs(expected.astype(mx.float32)-actual.astype(mx.float32))).item())
                    result['sorted_route_probe'] = dict(samples_s=elapsed, setup_sort_s=sort_s,
                        max_abs_error=error, source_operation=metadata,
                        scope='sorted flattened real routes; setup excluded from kernel timing, no runtime change')
            if not sweep_done and metadata['operation'] == 'quantized_matmul' and args[1].ndim == 2:
                sweep_done = True
                result['dense_row_sweep'] = []
                flat = args[0].reshape(-1, args[0].shape[-1])
                for rows in (1, 2, 3, 4, 5, 6, 8, 16):
                    x = flat[mx.arange(rows) % flat.shape[0]]
                    mx.eval(x)
                    call_args = (x, *args[1:])
                    for _ in range(3):
                        mx.eval(original(*call_args, **kwargs)); mx.synchronize()
                    elapsed = []
                    for _ in range(10):
                        t = time.perf_counter()
                        mx.eval(original(*call_args, **kwargs)); mx.synchronize()
                        elapsed.append(time.perf_counter()-t)
                    sweep = dict(rows=rows, samples_s=elapsed,
                        scope='row-count control with real captured weights and repeated real activation rows')
                    result['dense_row_sweep'].append(sweep)
        result['status'] = 'PASS' if code == 0 else 'FAIL'
    except BaseException as exc:
        result.update(status='FAIL', error=repr(exc))
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        for host, name, original in reversed(undo):
            setattr(host, name, original)
        a.output.write_text(json.dumps(result, indent=2)+'\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
