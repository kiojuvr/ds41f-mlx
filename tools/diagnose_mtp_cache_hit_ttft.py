#!/usr/bin/env python3
"""Finite synthetic ordinary production HTTP TTFT diagnostic (no client session).

Run with the admitted normal-local Python; uses real create_app/production core.
One seed and one exact repeat per size; repeat retains the ordinary real suffix.
No mocks, cache injection, replay, alternative generation path, or soak.
"""
import argparse
import json
import time
from pathlib import Path


def run(sizes, output, *, hit_repeats=1):
    from fastapi.testclient import TestClient
    from ds41f_mlx.mtp_identity import config, inspect
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    from ds41f_mlx.serving.server import create_app, load_v41_tokenizer, prepare_request

    cfg = config(profile='mtp-serving-v1')
    print('Admitting production identity', flush=True)
    identity = inspect(cfg)
    backend = ProductionMTPBackend(runtime_config=cfg)
    backend.dependency_identity = identity['identity_sha256']
    tokenizer = load_v41_tokenizer(cfg.recipe_path)
    app = create_app(profile='mtp-serving-v1', runtime_config=cfg, backend=backend)
    receipt = dict(identity=backend.dependency_identity, hit_repeats=hit_repeats, cases=[])
    output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        output.write_text(json.dumps(receipt, indent=2)+'\n')
    try:
        with TestClient(app) as client:
            for size in sizes:
                unit = ' The ledger records immutable committed checkpoints and independent continuation branches.\n'
                units = max(1, size // len(tokenizer.encode(unit)))
                def body(n):
                    return dict(model='deepseek-v4.1-flash', temperature=0, max_tokens=8,
                                messages=[dict(role='user', content=f'Synthetic ledger size {size}.\n'+unit*n+'\nReply with exactly OK.')])
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
                    t0 = time.perf_counter()
                    response = client.post('/v1/chat/completions', json=request)
                    response.raise_for_status()
                    result = response.json()
                    trace = dict(backend.serving_traces[-1])
                    row = dict(label=label, wall_s=time.perf_counter()-t0, usage=result['usage'], trace=trace)
                    case['rows'].append(row)
                    save()
                    print(json.dumps(dict(size=size, label=label, usage=result['usage'], phases={k:v for k,v in trace.items() if k.endswith('_s')})), flush=True)
                    assert result['choices'][0]['message']['content'] == 'OK'
                    assert trace['cache_published'] and not trace['cancelled']
                    assert trace['prompt_replay'] == trace['full_cache_repack'] == 0
                    assert set(trace['target_offsets']+trace['dspark_offsets']) == {trace['canonical_frontier']}
                    health = client.get('/health').json()
                    assert not health['fatal_error'] and not health['active_requests'] and not health['queued_requests']
                    if label.startswith('hit'):
                        assert trace['cached_tokens'] == result['usage']['prompt_tokens']-2
                        assert trace['remaining_suffix_tokens'] == 2
                save()
    finally:
        backend.close()
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', type=int, default=[4096, 32768, 220000])
    parser.add_argument('--output', type=Path, default=Path('artifacts/mtp-cache-hit-ttft/diagnostic.json'))
    parser.add_argument('--hit-repeats', type=int, choices=[1, 2, 3], default=1,
                        help='Bounded repeat only for phase noise checks')
    args = parser.parse_args()
    run(args.sizes, args.output, hit_repeats=args.hit_repeats)
