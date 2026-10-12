#!/usr/bin/env python3
"""Finite singleton context-copy closure on an isolated admitted candidate.
No proposal/target overrides. Only inspection and a cancellation timing gate.
Run installed candidate python -I -B, with explicit DS41F_CHECKPOINT.
"""
import argparse
import base64
import gc
import hashlib
import json
import threading
import time
import weakref
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
ROOT = Path(__file__).resolve().parents[1]
from ds41f_mlx.mtp_identity import config, inspect
cfg = config(port=18073, profile='mtp-serving-v1')
identity = inspect(cfg)
print('Admitted', identity['identity_sha256'], flush=True)
import mlx.core as mx
import httpx
import uvicorn
from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
from ds41f_mlx.serving.server import create_app
from ds41f_mlx.runtime import mtp_lifecycle as life
from omlx.patches.mlx_lm_mtp import batch_generator as bg, context_copy as cc
backend = ProductionMTPBackend(runtime_config=cfg)
backend.dependency_identity = identity['identity_sha256']
app = create_app(profile='mtp-serving-v1', runtime_config=cfg, backend=backend)
server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=18073, log_level='warning'))
thread = threading.Thread(target=server.run)
receipt = dict(identity=identity['identity_sha256'], status='RUNNING', cases=[])
current = None
cancel_gate = threading.Event()
continue_gate = threading.Event()
arm_cancel = False
in_copy_verify = False
backbone = bg._call_backbone_impl
verify = bg._run_verify_cycle_chain
quiet = life.canonical_quiesce_native_singleton


def checked_backbone(*args, **kw):
    global arm_cancel
    result = backbone(*args, **kw)
    if in_copy_verify and arm_cancel:
        arm_cancel = False
        current['cancellation_gate'] = 'target forward returned; acceptance/rollback pending'
        cancel_gate.set()
        assert continue_gate.wait(30), 'cancellation gate timeout'
    return result


def checked_verify(g, s, **kw):
    global in_copy_verify
    copied = s.copy_drafts
    used = int(s.drafts.shape[0])
    start = g.prompt_cache[0].size()
    accepted = s.stats.accepts
    if copied:
        assert used == s.depth
        assert len(s.context_copy.ids) == s.hist_offset == start
        assert all(c.offset == start for c in s.mtp_cache)
        assert s.context_copy.image_token_id not in s.drafts.tolist()
    in_copy_verify = copied
    try:
        result = verify(g, s, **kw)
    finally:
        in_copy_verify = False
    assert not kw.get('defer_commit')
    m = s.stats.accepts - accepted
    assert s.controller.cur == min(s.controller.max_depth, m + 1)
    assert all(c.size() == start + m + 1 for c in g.prompt_cache)
    assert all(c.offset == start + m + 1 for c in s.mtp_cache)
    if copied:
        from copy import copy
        matcher = copy(g._matchers[0])
        backend_stop = any(matcher.advance(int(e[0])) for e in s.queue)
        current['copy_cycles'].append(dict(depth=used, accepted=m,
            terminal=s.next_main is None, backend_stop=backend_stop,
            consumed_frontier=start+m+1))
    return result


def checked_quiet(**kw):
    s = kw['mtp_state']
    copier = getattr(s, 'context_copy', None)
    if copier is not None:
        current['index_bytes'] = copier.resident_bytes()
        current['index_tokens'] = len(copier.ids)
        current['index_ref'] = weakref.ref(copier)
    result = quiet(**kw)
    assert s.context_copy is None and s.copy_drafts is None
    assert all(r.offset == len(result.canonical_tokens) for r in result.dspark_context.caches)
    return result


bg._run_verify_cycle_chain = checked_verify
bg._call_backbone_impl = checked_backbone
life.canonical_quiesce_native_singleton = checked_quiet
source = (ROOT/'tools/bench_template_reuse.py').read_text()
rewrite = dict(model=cfg.model_id, temperature=0, max_tokens=384, messages=[dict(role='user',
    content='Edit this Python file: rename the prefix variable to kernel_prefix and update its uses. Return the complete updated file, preserving all other code exactly. No explanation.\n\n```python\n'+source+'\n```')])
ordinary = dict(model=cfg.model_id, temperature=0, max_tokens=384, messages=[dict(role='user',
    content='Explain three practical tradeoffs in designing a local code editing agent. Include concrete failure modes and mitigations. Do not quote this prompt.')])


def save():
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(receipt, indent=2)+'\n')


def idle(client):
    for _ in range(18000):
        h = client.get('/health').json()
        if not h['active_requests'] and not h['queued_requests']:
            assert not h['fatal_error'], h
            return
        time.sleep(.05)
    raise AssertionError('settlement timeout')


def states():
    import numpy as np
    def digest(x):
        if x is None:
            return None
        if isinstance(x, mx.array):
            x = x.view(mx.uint8)
            mx.eval(x)
        return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
    return sorted([dict(tokens=list(c.tokens), images=c.images,
        target=[[digest(x) for x in t.cache] for t in c.target],
        rings=[dict(offset=r.offset, keys=digest(r.keys)) for r in c.rings])
        for c in backend.scheduler.checkpoints._payloads.values()], key=lambda x: (len(x['tokens']), x['tokens']))


def begin(label):
    global current
    current = dict(label=label, copy_cycles=[], index_bytes=0, index_tokens=0)


def finish(client, result=None, wall=None):
    idle(client)
    ref = current.pop('index_ref', None)
    gc.collect()
    assert ref is None or ref() is None, 'request index retained beyond settlement'
    trace = dict(backend.serving_traces[-1])
    assert trace['prompt_replay'] == trace['full_cache_repack'] == 0
    assert set(trace['target_offsets']+trace['dspark_offsets']) == {trace['canonical_frontier']}
    assert trace['cache_published'] and trace['queue_empty'] and trace['prediction_retired']
    assert trace['generation_policy'] is not None
    row = dict(current, response=result, wall_s=wall, trace=trace, state=states())
    receipt['cases'].append(row)
    save()
    print(current['label'], 'decode', trace['decode_s'], 'stats', trace['mtp_stats'], 'index', current['index_bytes'], flush=True)
    return row


def post(client, label, body):
    begin(label)
    t = time.perf_counter()
    r = client.post('/v1/chat/completions', json=body)
    if r.is_error:
        receipt['failure'] = dict(label=label, status=r.status_code, body=r.text,
            health=client.get('/health').json(), traces=backend.serving_traces)
        receipt['status']='FAIL'
        save()
        print(json.dumps(receipt['failure'],default=str),flush=True)
    r.raise_for_status()
    return finish(client, r.json(), time.perf_counter()-t)


try:
    thread.start()
    while not server.started and thread.is_alive():
        time.sleep(.1)
    with httpx.Client(base_url='http://127.0.0.1:18073', timeout=1800) as client:
        cc.ENABLED = True
        sampled = {k:v for k,v in rewrite.items() if k != 'temperature'} | dict(seed=4363)
        s1 = post(client, 'sampled_1', sampled)
        s2 = post(client, 'sampled_2', sampled)
        assert s1['trace']['execution_sampling'] == s2['trace']['execution_sampling'] == 'stochastic'
        assert s1['trace']['mtp_stats']['copy_verified'] > 0
        assert s1['response']['choices'] == s2['response']['choices']
        assert s1['state'] == s2['state']
        # Actual production EOS after a copied textual suffix, no injected stop
        # matcher or target logits. Known following prompt delimiters may be
        # wrong drafts: target must terminate inside the speculative window.
        paragraph = ('The ledger records immutable committed checkpoints and continuation branches. '
                     'A local agent reads the file, applies a focused edit, and verifies the resulting code. '
                     'Cancellation preserves the settled prefix while discarding speculative work.')
        terminal_body = dict(model=cfg.model_id, temperature=0, max_tokens=192,
            messages=[dict(role='user', content='Repeat exactly the following paragraph, with no introduction, quotation marks, or explanation:\n\n'+paragraph)])
        cc.ENABLED = False
        tb = post(client, 'terminal_baseline', terminal_body)
        cc.ENABLED = True
        tc = post(client, 'terminal_copy', terminal_body)
        assert tc['response']['choices'] == tb['response']['choices']
        assert tc['trace']['canonical_generated'] == tb['trace']['canonical_generated']
        assert tc['state'] == tb['state']
        assert tc['response']['choices'][0]['finish_reason'] == 'stop'
        assert any(c['backend_stop'] and c['accepted'] < c['depth']-1 for c in tc['copy_cycles']), 'EOS not inside copied window with a following draft'
        # The already-qualified bounded corn image, then text-only rewrite.
        image_path = cfg.checkpoint_path/'inference/examples/images/corn.jpeg'
        image = dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,'+base64.b64encode(image_path.read_bytes()).decode()))
        vm = [dict(role='user', content=[image, dict(type='text', text='Identify this food in two short sentences.')])]
        vs = post(client, 'vision_seed', dict(model=cfg.model_id, temperature=0, max_tokens=96, messages=vm))
        assert vs['trace']['vision_encoded_count'] == 1
        vc_body = dict(rewrite, messages=vm+[vs['response']['choices'][0]['message'], rewrite['messages'][0]])
        vc = post(client, 'vision_text_copy', vc_body)
        assert vc['trace']['vision_encoded_count'] == 0 and vc['trace']['cached_tokens'] > 0
        assert vc['trace']['mtp_stats']['copy_verified'] > 0
        assert any(c['images'] for c in vc['state'])
        vr = post(client, 'vision_text_reuse', vc_body)
        assert vr['trace']['vision_encoded_count'] == 0
        assert vr['response']['choices'] == vc['response']['choices'] and vr['state'] == vc['state']
        # Disconnect after a copied target forward, before acceptance/rollback. Native
        # verification must settle before canonical cancellation; no GPU rewind.
        begin('copy_cancel')
        cancel_gate.clear(); continue_gate.clear(); arm_cancel = True
        with client.stream('POST', '/v1/chat/completions', json=dict(rewrite, stream=True)) as stream:
            stream.raise_for_status()
            assert cancel_gate.wait(60), 'copy cancellation did not activate'
        time.sleep(.2)
        continue_gate.set()
        cr = finish(client)
        assert cr['trace']['cancelled'] and cr['trace']['mtp_stats']['copy_verified'] > 0
        assert cr['trace']['quiescence']['discarded_future_token'] is not None
        partial = cr['trace']['response']['choices'][0]['message']
        follow = dict(rewrite, max_tokens=96,
            messages=rewrite['messages']+[partial, dict(role='user',content='Continue the code from where you stopped. Do not repeat the earlier code.')])
        recovery = post(client, 'copy_cancel_continuation', follow)
        assert recovery['trace']['cached_tokens'] >= 1108
        # Only after all missing edges close: same two representative workloads,
        # two timed repeats per mode. No compiler/cache campaign.
        for name, body in [('rewrite', rewrite), ('ordinary', ordinary)]:
            cc.ENABLED = False
            post(client, name+'_warm', {**body, 'max_tokens':16})
            first = None
            for mode in ('baseline', 'copy', 'baseline', 'copy'):
                cc.ENABLED = mode == 'copy'
                row = post(client, name+'_'+mode, body)
                if first is None:
                    first = row
                assert row['response']['choices'] == first['response']['choices']
                assert row['trace']['canonical_generated'] == first['trace']['canonical_generated']
                assert row['state'] == first['state']
        receipt['status'] = 'PASS'
        save()
finally:
    continue_gate.set()
    server.should_exit = True
    if thread.is_alive():
        thread.join(timeout=1800)
    backend.close()
    bg._run_verify_cycle_chain = verify
    bg._call_backbone_impl = backbone
    life.canonical_quiesce_native_singleton = quiet
