#!/usr/bin/env python3
"""Finite #4354 same-process production transitions on the current baseline.
Three prompt sizes, two real prefix hits each and seeded ordinary sampling.
Factory instrumentation records specializations, not GPU dispatch counts.
Retained paired-state hashing is outside request timing. No compiler cache claim.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path


def run(sizes, output, *, hit_repeats=1):
    import mlx.core as mx
    seen, calls = set(), []
    make = mx.fast.metal_kernel
    affected = ('activation', 'sinkhorn', 'hc_post', 'unpermute_reduce', 'topk')
    def record(**kw):
        kernel = make(**kw)
        name = kw['name']
        def call(**args):
            if any(s in name for s in affected):
                template = tuple((k, str(v)) for k,v in args.get('template', []))
                inputs = tuple((str(x.dtype), 's' if x.ndim == 0 else 'c' if x.size < 8 else 'd') for x in args['inputs'])
                key = (name, template, inputs, tuple(map(str,args['output_dtypes'])))
                if key not in seen:
                    calls.append(dict(name=name, template=template, shapes=[x.shape for x in args['inputs']], inputs=inputs))
                    seen.add(key)
            return kernel(**args)
        return call
    mx.fast.metal_kernel = record
    from fastapi.testclient import TestClient
    from ds41f_mlx.mtp_identity import config, inspect
    cfg = config(profile='mtp-serving-v1')
    print('Admitting production identity', flush=True)
    identity = inspect(cfg)
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    from ds41f_mlx.serving.server import create_app, load_v41_tokenizer, prepare_request
    backend = ProductionMTPBackend(runtime_config=cfg)
    backend.dependency_identity = identity['identity_sha256']
    tokenizer = load_v41_tokenizer(cfg.recipe_path)
    app = create_app(profile='mtp-serving-v1', runtime_config=cfg, backend=backend)
    receipt = dict(identity=backend.dependency_identity, hit_repeats=hit_repeats, cases=[])
    def states():
        import mlx.core as mx
        pairs = []
        def digest(a):
            if a is None:
                return None
            import numpy as np
            if isinstance(a, mx.array):
                raw = a.view(mx.uint8)
                mx.eval(raw)
                data = np.asarray(raw).tobytes()
            else:
                data = np.asarray(a).tobytes()
            return dict(shape=a.shape, dtype=str(a.dtype), sha256=hashlib.sha256(data).hexdigest())
        for c in backend.scheduler.checkpoints._payloads.values():
            pairs.append(dict(tokens=c.tokens, target=[[digest(a) for a in t.cache] for t in c.target],
                              rings=[dict(offset=r.offset, keys=digest(r.keys)) for r in c.rings]))
        return sorted(pairs, key=lambda c: (len(c['tokens']), c['tokens']))
    output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        output.write_text(json.dumps(receipt, indent=2)+'\n')
    try:
        with TestClient(app) as client:
            for size in sizes:
                unit = ' The ledger records immutable committed checkpoints and independent continuation branches.\n'
                units = max(1, size // len(tokenizer.encode(unit)))
                def body(n):
                    return dict(model='deepseek-v4.1-flash', temperature=0, max_tokens=64,
                                messages=[dict(role='user', content=f'Synthetic ledger size {size}.\n'+unit*n+'\nWrite a detailed numbered explanation of checkpoint ownership with at least twenty items.')])
                request = body(units)
                # Fixture sizing only, outside timed requests; authoritative encoder.
                prepared = prepare_request('chat_completions', json.dumps(request).encode(), tokenizer=tokenizer,
                                           recipe_path=cfg.recipe_path, checkpoint=cfg.checkpoint_path, ordinary=True)
                units = max(1, units + (size-len(prepared.token_ids)) // len(tokenizer.encode(unit)))
                request = body(units)
                case = dict(requested_tokens=size, rows=[])
                receipt['cases'].append(case)
                for label in ('seed', *('hit' if i == 0 else f'hit_{i+1}' for i in range(hit_repeats))):
                    print(f'Running {size} {label}', flush=True)
                    identities_before = len(calls)
                    t0 = time.perf_counter()
                    response = client.post('/v1/chat/completions', json=request)
                    response.raise_for_status()
                    result = response.json()
                    trace = dict(backend.serving_traces[-1])
                    row = dict(label=label, wall_s=time.perf_counter()-t0, usage=result['usage'], trace=trace,
                               response=result, state=states(), new_custom_identities=calls[identities_before:])
                    case['rows'].append(row)
                    save()
                    print(json.dumps(dict(size=size, label=label, usage=result['usage'], phases={k:v for k,v in trace.items() if k.endswith('_s')})), flush=True)
                    assert result['usage']['completion_tokens'] == 64
                    assert trace['cache_published'] and not trace['cancelled']
                    assert trace['prompt_replay'] == trace['full_cache_repack'] == 0
                    assert set(trace['target_offsets']+trace['dspark_offsets']) == {trace['canonical_frontier']}
                    health = client.get('/health').json()
                    assert not health['fatal_error'] and not health['active_requests'] and not health['queued_requests']
                    if label.startswith('hit'):
                        assert trace['cached_tokens'] == result['usage']['prompt_tokens']-2
                        assert trace['remaining_suffix_tokens'] == 2
                # One seeded ordinary default-sampling request, using the same real retained prompt pair.
                request.pop('temperature')
                request['seed'] = 4360
                t0 = time.perf_counter()
                response = client.post('/v1/chat/completions', json=request)
                response.raise_for_status()
                result = response.json()
                trace = dict(backend.serving_traces[-1])
                case['rows'].append(dict(label='sampling', wall_s=time.perf_counter()-t0,
                                         response=result, usage=result['usage'], trace=trace, state=states()))
                assert trace['prompt_replay'] == trace['full_cache_repack'] == 0
                assert set(trace['target_offsets']+trace['dspark_offsets']) == {trace['canonical_frontier']}
                import mlx.core as mx
                receipt['peak_memory_bytes'] = mx.get_peak_memory()
                save()
    finally:
        backend.close()
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', type=int, default=[512, 640, 1152])
    parser.add_argument('--output', type=Path, default=Path('artifacts/template-reuse/production.json'))
    parser.add_argument('--hit-repeats', type=int, choices=[1, 2, 3], default=1,
                        help='Bounded repeat only for phase noise checks')
    args = parser.parse_args()
    run(args.sizes, args.output, hit_repeats=args.hit_repeats)
