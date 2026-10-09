"""Offline M50R native-candidate prefix oracle; never a generation producer.

The oracle implements no BatchGenerator, sampler, journal or child scheduler and
imports no archived executable. The existing public candidate is observed, then
retired before oracle work. Native verification/rollback is the subject. State is
selected independently on CPU from a DIFFERENT native numerical forward with
an adversarial unconsumed suffix. All comparisons materialize AFTER native calls.
The numerical authority remains qualified oMLX/MLX at the SAME physical width;
this is not a claim of an independent implementation of all model arithmetic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback


def prefix_indices(start, length, end, retained_start=0):
    return [i for i in range(length) if retained_start <= start+i < end]


def ring_append_reference(old, before, new, capacity):
    """CPU absolute-position slot oracle, not candidate rotate/concat logic."""
    import numpy as np
    count = new.shape[2]
    end = before + count
    width = min(end, capacity)
    out = np.empty((*new.shape[:2], width, new.shape[-1]), dtype=new.dtype)
    for absolute in range(max(0, end-capacity), end):
        destination = absolute % capacity if width == capacity else absolute
        if absolute < before:
            source = absolute % capacity if old.shape[2] == capacity else absolute
            out[:, :, destination] = old[:, :, source]
        else:
            out[:, :, destination] = new[:, :, absolute-before]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=Path('artifacts/m50r/prefix-oracle.json'))
    ap.add_argument('--frontiers', default='127,128,255,256')
    ap.add_argument('--baseline', type=Path, default=Path('artifacts/m50r/performance.json'),
                    help='source-matched rate receipt; oracle arithmetic/selectors are unchanged')
    args = ap.parse_args()
    from ds41f_mlx.mtp_identity import config, inspect
    cfg = config(); identity = inspect(cfg)
    import numpy as np
    import mlx.core as mx
    import omlx.scheduler
    from mlx_lm.generate import generation_stream
    from omlx.patches.deepseek_v41.loading import load
    from omlx.patches.deepseek_v41.language import Indexer, rope
    from omlx.patches.deepseek_v41.quantization import quantize_activation
    from ds41f_mlx.prefill_fp8_mlx import DeferredPrefillAppend
    from ds41f_mlx.runtime.mtp_resources import MTPWiredLimitLease
    out = dict(schema='ds41f.m50r.prefix-oracle.v1', status='RUNNING',
               identity_sha256=identity['identity_sha256'], cases=[],
               tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               independence='CPU absolute-position prefix and ring selection; separate same-width numerical forward with different unconsumed suffix; no candidate rollback used by oracle',
               scope='native primitive qualification plus deferred observation of the existing public candidate; no M51R integration',
               archive_imports=False, oracle_live_scheduler=False)
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2)+'\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.tool.py').write_bytes(Path(__file__).read_bytes())
    save()

    def host(x):
        if x is None:
            return None
        if isinstance(x, mx.array):
            mx.eval(x)
            return np.asarray(x.view(mx.uint8)).copy()
        return np.asarray(x).view(np.uint8).copy()

    def evidence(x):
        if x is None:
            return None
        return dict(shape=list(x.shape), bytes=x.nbytes, sha256=hashlib.sha256(x.tobytes()).hexdigest())

    def clone(cache):
        result = []
        for c in cache:
            n = type(c)(c.compress_ratio)
            n.cache = list(c.cache)
            n.left_padding, n.lengths = c.left_padding, c.lengths
            result.append(n)
        return result

    def state_host(cache):
        return [[host(x) for x in [*c.cache, c.left_padding, c.lengths]] for c in cache]

    def equal(a, b):
        return (a is None and b is None) or (a is not None and b is not None and
                    a.shape == b.shape and np.array_equal(a, b))

    def compare(actual, expected):
        mismatch = []
        for layer, (aa, ee) in enumerate(zip(actual, expected)):
            for slot, (a,e) in enumerate(zip(aa,ee)):
                if not equal(a,e):
                    mismatch.append(dict(layer=layer, slot=slot, actual=evidence(a), expected=evidence(e),
                        different_bytes=None if a is None or e is None or a.shape != e.shape else int(np.count_nonzero(a != e))))
        return mismatch

    def raw_slice(x, axis, indices):
        return None if x is None else np.take(x, indices, axis=axis)

    def selected_reference(base, oracle, publications, ids, before, count):
        # Expected is constructed WITHOUT native mtp_partial_rollback, stashes,
        # candidate snapshots, target replay or archived selection implementation.
        end = before + count
        full = state_host(oracle)
        base_host = state_host(base)
        for i, c in enumerate(oracle):
            fields = full[i]
            fields[0] = np.array([end], dtype=np.int32).view(np.uint8)
            window = host(publications[i]['window'])
            old_start = before - min(before, lm._config.window_size)
            fields[1] = raw_slice(window, 1, prefix_indices(old_start, window.shape[1], end,
                                                 max(0, end-lm._config.window_size)))
            ratio = c.compress_ratio
            if ratio:
                for slot in (2,3):
                    value = fields[slot]
                    if value is not None:
                        # A group is committed iff its last absolute token < end.
                        keep = [g for g in range(value.shape[1]) if (g+1)*ratio <= end]
                        fields[slot] = raw_slice(value, 1, keep)
                if ratio > 1:
                    for slot, value in zip((4,5), publications[i]['compressor']):
                        value = host(value)
                        first = before - before % ratio
                        last_complete = (end // ratio)*ratio
                        fields[slot] = raw_slice(value, 1, prefix_indices(first, value.shape[1], end, last_complete))
            if i == 0 and lm._hasher is not None:
                old = np.asarray(base[0][6], dtype=np.int64)
                compressed = lm._hasher.token_map[np.asarray(ids[:count], dtype=np.int64)]
                joined = np.concatenate([old.reshape(1,-1), compressed.reshape(1,-1)], axis=1)
                lookback = lm._config.engram_max_ngram_size-1
                fields[6] = joined[:, -lookback:].astype(np.int64).view(np.uint8)
            for slot, value in ((7,base[i].left_padding),(8,base[i].lengths)):
                fields[slot] = None if value is None else (np.asarray(value)-count).astype(np.int32).view(np.uint8)
        return full

    index_code = Indexer.__call__.__code__
    def numerical(cache, ids, *, native_verify):
        indices = {}
        def profile(frame, event, arg):
            if event == 'return' and frame.f_code is index_code:
                layer = frame.f_locals['self']._layer
                indices[layer] = dict(indices=arg, candidates=frame.f_locals['shared'].get('candidates'))
        previous = sys.getprofile(); sys.setprofile(profile)
        try:
            if native_verify:
                logits, taps = lm(mx.array([ids], mx.int64), cache=cache,
                                  n_confirmed=1, return_hidden=True)
                publications = cache[0]._mtp_draft_stash[3]
            else:
                publications = [{} for _ in cache]
                logits, taps = lm._forward(mx.array([ids], mx.int64), cache=cache,
                                           return_dspark_hidden=True, mtp_verify_states=publications)
        finally:
            sys.setprofile(previous)
        mx.eval(logits, taps)
        return logits, taps, publications, indices

    model = None
    try:
        model, _ = load(cfg.checkpoint_path, preserve_mtp=True, engram_ssd_offload=True)
        lm = model.language_model; lm.configure_mtp(True,3)
        baseline = json.loads(args.baseline.read_text())
        assert baseline['identity']['identity_sha256'] == identity['identity_sha256']
        prompt = baseline['rows'][1]['canonical_token_ids'][:242]
        frontier_list = [int(s) for s in args.frontiers.split(',')]
        with mx.stream(generation_stream), MTPWiredLimitLease(mx, generation_stream):
            for before in frontier_list:
                cache = lm.make_cache(); rings = lm.make_mtp_cache()
                ids = (prompt * (before//len(prompt)+1))[:before]
                taps_by_layer = {i:[] for i in lm._config.dspark_target_layer_ids}
                original = {i:lm.layers[i] for i in taps_by_layer}
                class Tap:
                    def __init__(self,i,layer): self.i,self.layer=i,layer
                    def __getattr__(self,k): return getattr(self.layer,k)
                    def __call__(self,h,*a,**kw):
                        value=mx.mean(h,axis=-2)
                        taps_by_layer[self.i].append(value if value.ndim==3 else value[None])
                        return self.layer(h,*a,**kw)
                try:
                    for i, layer in original.items(): lm.layers[i]=Tap(i,layer)
                    app = DeferredPrefillAppend.create(lm,cache,ids,committed_frontier=0,mx=mx)
                    app.execute_all()
                finally:
                    for i, layer in original.items(): lm.layers[i]=layer
                hidden = mx.concatenate([mx.concatenate(taps_by_layer[i],axis=1) for i in taps_by_layer],axis=-1)
                lm.dspark_append_context(hidden,rings,start_offset=0)
                mx.eval([c.state for c in cache], [r.keys for r in rings])
                assert all(c.size()==before for c in cache)
                base_state=state_host(cache)
                subject_ids=[11932,20370,20370,20370]
                for accepted in range(4):
                    count=accepted+1
                    actual=clone(cache)
                    al,at,_,ai=numerical(actual,subject_ids,native_verify=True)
                    # Canary changes ONLY unconsumed inputs, preserving width,
                    # precision and the qualified numerical kernel geometry.
                    canary=subject_ids[:count]+[11,97,128][:4-count]
                    oracle=clone(cache)
                    ol,ot,pubs,oi=numerical(oracle,canary,native_verify=False)
                    expected=selected_reference(cache,oracle,pubs,subject_ids,before,count)
                    assert lm.mtp_partial_rollback(actual,accepted,3)
                    mismatch=compare(state_host(actual),expected)
                    tap_ok=equal(host(at[:,:count]),host(ot[:,:count]))
                    logit_ok=equal(host(al[:,:count]),host(ol[:,:count]))
                    index_ok={str(k)+'.'+name:equal(
                        None if ai[k][name] is None else host(ai[k][name][:,:count]),
                        None if oi[k][name] is None else host(oi[k][name][:,:count]))
                        for k in ai for name in ('indices','candidates')}
                    actual_rings=[]; expected_rings=[]
                    for stage, old_ring in zip(lm.mtp,rings):
                        new_ring=type(old_ring)(old_ring.max_size)
                        new_ring.keys, new_ring.offset=old_ring.keys,old_ring.offset
                        actual_rings.append(new_ring)
                    lm.dspark_append_context(at[:,:count],actual_rings,start_offset=before)
                    first=lm.mtp[0]
                    projected=first.main_norm(first.main_proj(ot[:,:count]))
                    positions=mx.arange(before,before+count)
                    for stage,old_ring in zip(lm.mtp,rings):
                        new=quantize_activation(rope(stage.attn.kv_norm(stage.attn.wkv(projected)),positions,lm._config,False))[:,None]
                        expected_rings.append(ring_append_reference(host(old_ring.keys),before,host(new),old_ring.max_size))
                    ring_ok=[equal(host(a.keys),e) and a.offset==before+count
                             for a,e in zip(actual_rings,expected_rings)]
                    row=dict(before=before,accepted=accepted,consumed=count,verify_width=4,
                             input_ids=subject_ids,oracle_ids=canary,
                             state_mismatches=mismatch,state_content_exact=not mismatch,
                             tap_content_exact=tap_ok,logit_content_exact=logit_ok,
                             index_publications_exact=index_ok,ring_content_exact=ring_ok,
                             metadata_exact=all(c.meta_state==b.meta_state for c,b in zip(actual,cache)),
                             expected_slot_digests=[[evidence(x) for x in layer] for layer in expected])
                    out['cases'].append(row);save()
                    print(json.dumps({k:v for k,v in row.items() if k not in ('expected_slot_digests','state_mismatches')}),flush=True)
                assert not compare(state_host(cache),base_state), 'qualification modified predecessor'
            # Observe actual public candidate first/full and final/partial cycles.
            # References only while live: no read/eval or oracle execution until
            # the request owner and its sole scheduler have been retired.
            import asyncio, socket
            from dataclasses import replace
            import httpx, uvicorn
            from omlx.patches.mlx_lm_mtp import batch_generator as bg_module
            from omlx.patches.deepseek_v41.mtp import DSparkMixin
            from ds41f_mlx.serving.mtp_public import LocalMTPBackend
            from ds41f_mlx.serving.server import create_app
            from ds41f_mlx.serving.local_h11 import LocalH11Protocol
            subjects=[]; pending={}; cycle_number=[0]
            codes=dict(backbone=bg_module._call_backbone_captured.__code__,
                       cycle=bg_module._run_verify_cycle_chain.__code__,
                       ring=DSparkMixin.dspark_append_context.__code__)
            def live_profile(frame,event,arg):
                if frame.f_code is codes['backbone'] and event=='call':
                    inputs=frame.f_locals['inputs']
                    if inputs.shape[-1]==4:
                        cycle_number[0]+=1
                        if cycle_number[0] in (1,10):
                            pending.update(number=cycle_number[0],inputs=inputs,
                                base=clone(frame.f_locals['cache']),indices={})
                elif frame.f_code is codes['backbone'] and event=='return' and pending:
                    pending['logits'],pending['taps']=arg[:2]
                elif frame.f_code is index_code and event=='return' and pending:
                    pending['indices'][frame.f_locals['self']._layer]=dict(
                        indices=arg,candidates=frame.f_locals['shared'].get('candidates'))
                elif frame.f_code is codes['ring'] and pending:
                    rings_now=frame.f_locals['cache']
                    if event=='call':
                        pending['ring_before']=[(r.keys,r.offset,r.max_size) for r in rings_now]
                        pending['ring_taps']=frame.f_locals['main_hidden']
                    elif event=='return':
                        pending['ring_after']=[(r.keys,r.offset) for r in rings_now]
                elif frame.f_code is codes['cycle'] and event=='return' and pending:
                    pending['accepted']=frame.f_locals['m']
                    pending['actual']=clone(frame.f_locals['gen_batch'].prompt_cache)
                    subjects.append(dict(pending));pending.clear()
            async def live_request():
                backend=LocalMTPBackend(runtime_config=cfg)
                backend._model=model  # same resident weights, sequential owner
                lm._p7_enable_overlap=True  # exactly the candidate loader setting
                backend.dependency_identity=identity['identity_sha256']
                sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen(8)
                actual_cfg=replace(cfg,port=sock.getsockname()[1])
                server=uvicorn.Server(uvicorn.Config(create_app(backend=backend,runtime_config=actual_cfg,profile='mtp-singleton-v1'),
                    lifespan='off',log_level='warning',workers=1,proxy_headers=False,ws='none',loop='asyncio',
                    limit_concurrency=8,backlog=8,timeout_keep_alive=5,h11_max_incomplete_event_size=16384,http=LocalH11Protocol))
                task=asyncio.create_task(server.serve(sockets=[sock]))
                try:
                    while not server.started: await asyncio.sleep(.01)
                    await backend._call(lambda:sys.setprofile(live_profile))
                    async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{actual_cfg.port}',timeout=600) as client:
                        response=await client.post('/v1/sessions',content=b'{}',headers={'Content-Type':'application/json'})
                        response.raise_for_status();sid=response.json()['id'];rec=backend.sessions[sid]
                        response=await client.post(f'/v1/sessions/{sid}/chat/completions',content=bytes.fromhex(baseline['rows'][1]['request_hex']),
                            headers={'Content-Type':'application/json','X-DS41F-Request-Sequence':'1'})
                        response.raise_for_status()
                        assert rec.last_turn['canonical_generated']==baseline['rows'][1]['trace']['canonical_generated']
                        response=await client.delete(f'/v1/sessions/{sid}',headers={'Content-Type':'application/json'})
                        response.raise_for_status();assert sid not in backend.sessions
                finally:
                    await backend._call(lambda:sys.setprofile(None))
                    server.should_exit=True;await task;sock.close();backend.close()
            asyncio.run(live_request())
            assert len(subjects)==2 and [s['accepted'] for s in subjects]==[3,2]
            out['live_cases']=[]
            for s in subjects:
                before=s['base'][0].size(); count=s['accepted']+1
                ids=np.asarray(s['inputs']).reshape(-1).tolist()
                canary=ids[:count]+[11,97,128][:4-count]
                oracle=clone(s['base'])
                ol,ot,pubs,oi=numerical(oracle,canary,native_verify=False)
                expected=selected_reference(s['base'],oracle,pubs,ids,before,count)
                differences=compare(state_host(s['actual']),expected)
                taps_ok=equal(host(s['taps'][:,:count]),host(ot[:,:count]))
                logits_ok=equal(host(s['logits'][:,:count]),host(ol[:,:count]))
                publications_ok={str(k)+'.'+name:equal(
                    None if s['indices'][k][name] is None else host(s['indices'][k][name][:,:count]),
                    None if oi[k][name] is None else host(oi[k][name][:,:count]))
                    for k in oi for name in ('indices','candidates')}
                projected=lm.mtp[0].main_norm(lm.mtp[0].main_proj(ot[:,:count]))
                rings_ok=[]
                for stage,(old,offset,capacity),(actual_keys,actual_offset) in zip(lm.mtp,s['ring_before'],s['ring_after']):
                    new=quantize_activation(rope(stage.attn.kv_norm(stage.attn.wkv(projected)),
                        mx.arange(offset,offset+count),lm._config,False))[:,None]
                    expected_ring=ring_append_reference(host(old),offset,host(new),capacity)
                    rings_ok.append(equal(host(actual_keys),expected_ring) and actual_offset==offset+count)
                row=dict(cycle=s['number'],before=before,accepted=s['accepted'],input_ids=ids,
                    oracle_ids=canary,owner_retired_before_oracle=True,state_content_exact=not differences,
                    state_mismatches=differences,tap_content_exact=taps_ok,logit_content_exact=logits_ok,
                    ring_append_input_matches_same_forward_taps=equal(host(s['ring_taps']),host(s['taps'][:,:count])),
                    index_publications_exact=publications_ok,ring_content_exact=rings_ok)
                out['live_cases'].append(row);save()
            out['status']='PASS' if all(r['state_content_exact'] and r['tap_content_exact'] and
                r['logit_content_exact'] and all(r['index_publications_exact'].values()) and
                all(r['ring_content_exact']) for r in [*out['cases'],*out['live_cases']]) else 'BLOCK'
    except BaseException:
        out['status']='ERROR';out['error']=traceback.format_exc();raise
    finally:
        if model is not None: model.close()
        save()


if __name__=='__main__': main()
