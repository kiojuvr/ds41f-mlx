#!/usr/bin/env python3
"""Bounded public ordinary-MTP benchmark; run with a non-editable installed Python.

No executor patches, cache injection, sampling tuning or automatic retry.
The in-process HTTP app is the ordinary production app (not a development executor).
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from fastapi.testclient import TestClient
    import mlx.core as mx
    import ds41f_mlx
    from ds41f_mlx.mtp_identity import config, inspect
    cfg = config(profile='mtp-serving-v1')
    # Match canonical launch: admission precedes imports of the physical engine.
    identity = inspect(cfg)
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    from ds41f_mlx.serving.server import create_app, load_v41_tokenizer, prepare_request
    backend = ProductionMTPBackend(runtime_config=cfg)
    backend.dependency_identity = identity['identity_sha256']
    tokenizer = load_v41_tokenizer(cfg.recipe_path)
    receipt = dict(schema='ds41f.public-mtp-benchmark.v1', identity=identity,
        installed_module=str(Path(ds41f_mlx.__file__).resolve()),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        python=platform.python_version(), macos=platform.mac_ver()[0],
        dependencies={d.metadata['Name']: d.version for d in metadata.distributions()},
        workload_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False)+'\n')
    save()

    def body(messages):
        # Sampling omitted deliberately: ordinary headless production defaults.
        return dict(model='deepseek-v4.1-flash', messages=messages, max_tokens=256,
                    reasoning_effort='none')

    def sized_request(size):
        unit = ' Ledger: immutable checkpoints preserve committed state; independent branches retain coherent target and draft frontiers.\n'
        intro = f'Agent maintenance journal for workload {size}. Treat the ledger as reference data.\n'
        task = '\nWrite a detailed engineering plan for maintaining this long-running agent session: explain checkpoint ownership, safe branching, cancellation, tool-result reconciliation, and incremental continuation. Give at least twenty numbered recommendations with explanations. Do not copy the ledger.\n'
        n = size // len(tokenizer.encode(unit))
        def make(n):
            return body([dict(role='user', content=intro+unit*n+task)])
        request = make(n)
        prepared = prepare_request('chat_completions', json.dumps(request).encode(), tokenizer=tokenizer,
                                   recipe_path=cfg.recipe_path, checkpoint=cfg.checkpoint_path, ordinary=True)
        n += (size-len(prepared.token_ids)) // len(tokenizer.encode(unit))
        return make(n)

    app = create_app(profile='mtp-serving-v1', runtime_config=cfg, backend=backend)
    try:
        with TestClient(app) as client:
            # Normal model load, outside request timers. Cold means no compatible
            # retained conversation, NOT cold disk/Metal/compiler/model startup.
            t0 = time.perf_counter()
            backend.load()
            mx.synchronize()
            receipt['model_load_s'] = time.perf_counter()-t0
            print('Model loaded', receipt['model_load_s'], flush=True)

            def run(label, request, retained=False):
                print('Running', label, flush=True)
                t0 = time.perf_counter()
                response = client.post('/v1/chat/completions', json=request)
                wall = time.perf_counter()-t0
                response.raise_for_status()
                result = response.json()
                trace = dict(backend.serving_traces[-1])
                stats = trace['mtp_stats']
                considered = stats['copy_verified']+stats['native_verified']
                accepted = stats['copy_accepted']+stats['native_accepted']
                proposed = stats['copy_proposed']+stats['native_proposed']
                row = dict(label=label, state='paired continuation' if retained else 'cold',
                    request_sha256=hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
                    response=result, trace=trace, wall_s=wall,
                    derived=dict(prompt_tokens=result['usage']['prompt_tokens'],
                        generated_tokens=len(trace['canonical_generated']),
                        decode_tokens_per_s=len(trace['canonical_generated'])/trace['decode_s'],
                        proposed_drafts=proposed, considered_drafts=considered,
                        accepted_drafts=accepted, acceptance=accepted/considered if considered else None),
                    mlx_process_cumulative_peak_bytes=mx.get_peak_memory())
                receipt['cases'].append(row)
                save()  # Retain evidence even if a correctness assertion fails.
                print(json.dumps(dict(label=label, wall_s=wall, derived=row['derived'],
                    phases={k:v for k,v in trace.items() if k.endswith('_s')},
                    cached=trace['cached_tokens'], suffix=trace['remaining_suffix_tokens'])), flush=True)
                assert trace['cache_published'] and not trace['cancelled']
                assert trace['prompt_replay'] == trace['full_cache_repack'] == 0
                assert set(trace['target_offsets']+trace['dspark_offsets']) == {trace['canonical_frontier']}
                assert len(trace['canonical_generated']) == 256
                assert trace['execution_sampling'] == 'stochastic'
                if retained:
                    assert trace['cached_tokens'] >= 262000
                else:
                    assert trace['cached_tokens'] == 0
                if retained:
                    assert trace['remaining_suffix_tokens'] < 1024
                health = client.get('/health').json()
                assert not health['fatal_error'] and not health['active_requests'] and not health['queued_requests']
                return result

            for size in (8192, 32768, 262144):
                request = sized_request(size)
                result = run(str(size)+'-cold', request)
            # Three consecutive ordinary requests extend the real settled output.
            # No checkpoint manipulation or special benchmark/session protocol.
            messages = request['messages']
            for turn in range(1, 4):
                messages = messages + [result['choices'][0]['message'], dict(role='user',
                    content=f'Continuation {turn}: expand the engineering plan with at least twenty further concrete recommendations, focusing on retained-prefix state coherence and safe tool execution. Do not repeat earlier text.')]
                result = run('262144-continuation-'+str(turn), body(messages), retained=True)
            receipt['status'] = 'PASS'
            save()
    finally:
        backend.close()


if __name__ == '__main__':
    main()
