#!/usr/bin/env python3
"""Finite #4363 singleton applicability probe; process-local, NOT runtime code.
Uses fetched upstream suffix proposer, existing target verify/rollback/semantic
horizon. No installed files changed. Candidate is deliberately unqualified.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--max-tokens', type=int, default=192)
a = p.parse_args()
import mlx.core as mx
from fastapi.testclient import TestClient
from ds41f_mlx.mtp_identity import config, inspect
cfg = config(profile='mtp-serving-v1')
print('Admitting unchanged baseline', flush=True)
identity = inspect(cfg)
from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
from ds41f_mlx.serving.server import create_app
from omlx.patches.mlx_lm_mtp import batch_generator as bg
proposer_path = ROOT / 'artifacts/context-copy/upstream-context-copy.py'
assert hashlib.sha256(proposer_path.read_bytes()).hexdigest() == 'e86e0a19aa3fb2e10459138dd73b00feeb34cea2c71aa25167a9319ce2f18328'
spec = importlib.util.spec_from_file_location('evaluated_context_copy', proposer_path)
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)
original_draft = bg._dspark_next_drafts
original_verify = bg._run_verify_cycle_chain
mode = 'baseline'
record = None

def draft(g, s, hidden, committed, prev):
    start = time.perf_counter()
    copied = []
    if mode == 'copy':
        if not hasattr(s, '_evaluation_copier'):
            s._evaluation_copier = cc.ContextCopy()
        ids = committed.tolist()
        s._evaluation_copier.extend(g.tokens[0], ids)
        room = g.max_tokens[0] - g._num_tokens[0] - len(ids) - 1
        copied = s._evaluation_copier.propose(min(room, s.depth))
        if len(copied) != s.depth:
            copied = []
        record['lookup_s'] += time.perf_counter() - start
        record['lookup_calls'] += 1
        record['index_tokens_max'] = max(record['index_tokens_max'], len(s._evaluation_copier._ids))
    s._evaluation_copy = bool(copied)
    if not copied:
        return original_draft(g, s, hidden, committed, prev)
    # Exactly upstream depth=0 branch: append confirmed taps, no decoder.
    host = bg._dspark_host(g.model)
    host.dspark_append_context(hidden, s.mtp_cache)
    s.hist_offset += int(committed.shape[0])
    s.drafts = mx.array(copied, dtype=mx.uint32)
    s.draft_lps = []
    ids = mx.array(copied, dtype=mx.int32)[:, None]
    s.draft_accept_lps = bg.SparseDraftQ(ids, mx.zeros(ids.shape, dtype=mx.float32), host.args.vocab_size)
    mx.async_eval(s.drafts)
    record['copied_generated'] += len(copied)

class SkipCopyObservation:
    def __init__(self, controller):
        self.controller = controller
    def __getattr__(self, name):
        return getattr(self.controller, name)
    def observe(self, *args, **kwargs):
        pass

def verify(g, s, **kw):
    copied = getattr(s, '_evaluation_copy', False)
    copier = getattr(s, '_evaluation_copier', None)
    k = int(s.drafts.shape[0])
    before = s.stats.accepts
    controller = s.controller
    if copied and controller is not None:
        s.controller = SkipCopyObservation(controller)
    try:
        result = original_verify(g, s, **kw)
    finally:
        s.controller = controller
    # Singleton non-deferred verify settles before return.
    assert not kw.get('defer_commit')
    accepted = s.stats.accepts - before
    kind = 'copied' if copied else 'native'
    record[kind + '_verified'] += k
    record[kind + '_accepted'] += accepted
    record['cycles'] += 1
    if copied:
        copier.observe(accepted, k)
    return result

bg._dspark_next_drafts = draft
bg._run_verify_cycle_chain = verify
backend = ProductionMTPBackend(runtime_config=cfg)
backend.dependency_identity = identity['identity_sha256']
app = create_app(profile='mtp-serving-v1', runtime_config=cfg, backend=backend)
receipt = dict(identity=identity['identity_sha256'], qualification=False, cases=[])
source = (ROOT / 'tools/bench_template_reuse.py').read_text()
# Actual local agent rewrite, not an invented exact-repeat token stream.
# A local edit early in a real file followed by a large unchanged region.
request = dict(model='deepseek-v4.1-flash', temperature=0, max_tokens=a.max_tokens,
               messages=[dict(role='user', content='Edit this Python file: rename the prefix variable to kernel_prefix and update its uses. Return the complete updated file, preserving all other code exactly. No explanation.\n\n```python\n' + source + '\n```')])
ordinary = dict(model='deepseek-v4.1-flash', temperature=0, max_tokens=a.max_tokens,
                messages=[dict(role='user', content='Explain three practical tradeoffs in designing a local code editing agent. Include concrete failure modes and mitigations. Do not quote this prompt.')])

def paired_digests():
    import numpy as np
    def digest(x):
        if x is None:
            return None
        if isinstance(x, mx.array):
            raw = x.view(mx.uint8)
            mx.eval(raw)
            data = np.asarray(raw).tobytes()
        else:
            data = np.asarray(x).tobytes()
        return hashlib.sha256(data).hexdigest()
    rows = []
    for c in backend.scheduler.checkpoints._payloads.values():
        rows.append(dict(tokens_sha256=hashlib.sha256(json.dumps(c.tokens).encode()).hexdigest(),
                         length=len(c.tokens), target=[[digest(x) for x in t.cache] for t in c.target],
                         rings=[dict(offset=r.offset, keys=digest(r.keys)) for r in c.rings]))
    return sorted(rows, key=lambda x: (x['length'], x['tokens_sha256']))

def save():
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(receipt, indent=2)+'\n')
try:
    with TestClient(app) as client:
        for name, body in [('rewrite', request), ('ordinary', ordinary)]:
            # Cold prefill is not compared to a warm paired hit.
            mode = 'baseline'
            record = dict(lookup_s=0., lookup_calls=0, index_tokens_max=0, copied_generated=0,
                          copied_verified=0, copied_accepted=0, native_verified=0, native_accepted=0, cycles=0)
            print('Warm prompt', name, flush=True)
            response = client.post('/v1/chat/completions', json={**body, 'max_tokens': 16})
            response.raise_for_status()
            for selected in ('baseline', 'copy', 'baseline', 'copy'):
                mode = selected
                record = dict(lookup_s=0., lookup_calls=0, index_tokens_max=0, copied_generated=0,
                              copied_verified=0, copied_accepted=0, native_verified=0, native_accepted=0, cycles=0)
                print('Running', name, mode, flush=True)
                t = time.perf_counter()
                response = client.post('/v1/chat/completions', json=body)
                response.raise_for_status()
                wall = time.perf_counter()-t
                result = response.json()
                trace = dict(backend.serving_traces[-1])
                row = dict(workload=name, mode=mode, wall_s=wall, response=result, trace=trace, stats=dict(record), state=paired_digests())
                receipt['cases'].append(row)
                save()
                assert trace['prompt_replay'] == trace['full_cache_repack'] == 0
                assert set(trace['target_offsets'] + trace['dspark_offsets']) == {trace['canonical_frontier']}
                print(json.dumps(dict(workload=name, mode=mode, wall_s=wall, decode_s=trace['decode_s'], stats=record)), flush=True)
finally:
    backend.close()
    bg._dspark_next_drafts = original_draft
    bg._run_verify_cycle_chain = original_verify
