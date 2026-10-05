#!/usr/bin/env python3
"""Real first-party OFF long-session qualification; no alternate executor.

Run initial and --restore in separate processes. Artifacts contain measurements,
not executable shadow caches. Model/checkpoint configuration stays operator-owned.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def displaced(name):
    return (name.startswith('omlx.patches.deepseek_v41.') and name.rsplit('.', 1)[-1] in
            set('activation cache config convert engram head hyper_connection kernels language loading model mtp packed_attention processing quantization routing sharding storage vision'.split())
            or name == 'omlx.patches.deepseek_v4.switch_layers')


def digest_ids(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--artifact-root', type=Path, required=True)
    p.add_argument('--context', type=int, default=200000)
    p.add_argument('--fixture-revision', help='Git corpus revision for exact-workload requalification only')
    p.add_argument('--turns', type=int, default=16)
    p.add_argument('--decode-tokens', type=int, default=256)
    p.add_argument('--restore', type=Path, help='initial-phase JSON (fresh process required)')
    p.add_argument('--compare', type=Path, help='same-workload pre-policy receipt, evidence only')
    p.add_argument('--restore-probe-only', action='store_true',
                   help='fresh exact restore/probe and close without growing an already full frontier')
    a = p.parse_args()
    if a.restore_probe_only and not a.restore:
        p.error('--restore-probe-only requires --restore')
    class NoDonor(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if displaced(fullname):
                raise AssertionError('displaced model import: '+fullname)
    sys.meta_path.insert(0, NoDonor())
    from ds41f_mlx.config import load_runtime_config
    cfg = load_runtime_config(); cfg.apply_import_paths()
    from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend, RecipePreparedRequest
    from ds41f_mlx.runtime.dwarfstar_prefill import DwarfStarMLXPrefillSession
    from ds41f_mlx.runtime.continuation_session import M8LiveContinuationSession, M8ContinuationError
    from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
    from ds41f_mlx.runtime.kv_persistence import save_m8_idle_state, restore_m8_idle_state
    import mlx.core as mx
    import numpy as np
    import psutil
    backend = DeepSeekRecipeRuntimeBackend(runtime_config=cfg)
    result = dict(schema='ds41f.standard-off.long-session.v1', status='RUNNING',
                  git_head=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                  command=sys.argv, phase='restore' if a.restore else 'initial', turns=[], resources=[],
                  runtime_sources={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                                   for f in sorted((ROOT/'ds41f_mlx').rglob('*.py'))},
                  tooling_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter(); stop = threading.Event(); session = None
    def memory():
        vm_stat=subprocess.check_output(['vm_stat'], text=True)
        return dict(vm_stat=vm_stat,elapsed_s=time.perf_counter()-start, rss_bytes=psutil.Process().memory_info().rss,
                    mlx_active_bytes=mx.get_active_memory(), mlx_cache_bytes=mx.get_cache_memory(),
                    mlx_peak_bytes=mx.get_peak_memory(), swap_used_bytes=psutil.swap_memory().used,
                    system_available_bytes=psutil.virtual_memory().available,
                    thread_count=psutil.Process().num_threads(), fd_count=psutil.Process().num_fds(),
                    disk_io={name: counters._asdict() for name, counters in
                             (psutil.disk_io_counters(perdisk=True) or {}).items()})
    def save():
        tmp = a.output.with_suffix('.tmp'); tmp.write_text(json.dumps(result, indent=2)+'\n'); tmp.replace(a.output)
    def progress(stage):
        print(json.dumps(dict(event='heartbeat', stage=stage, elapsed_s=time.perf_counter()-start)), flush=True)
    def sample():
        while not stop.wait(10):
            snapshot=memory()
            result['resources'].append(snapshot)
            with a.output.with_suffix('.resources.jsonl').open('a') as log:
                log.write(json.dumps(snapshot)+'\n')
            progress('resource-sample')
    result['start_memory']=memory()
    result['device']=mx.device_info()
    result['physical_memory_bytes']=psutil.virtual_memory().total
    observer = threading.Thread(target=sample, daemon=True); observer.start()
    def slots(cache):
        return [dict(layer=i, slot=j, shape=list(x.shape), dtype=str(x.dtype),
                     sha256=hashlib.sha256(np.asarray(x.view(mx.uint8)).tobytes()).hexdigest())
                for i,c in enumerate(cache) for j,x in enumerate(c.cache)]
    def idle_check(expected_list):
        assert session.state == 'idle' and session.live_cache is expected_list
        assert len(expected_list) == 40 and all(c.size() == session.frontier for c in expected_list)
        assert session.total_prompt_replay_count == session.total_full_cache_repack_count == 0
        assert not any(displaced(n) for n in sys.modules)
        assert all(type(c).__module__ == 'ds41f_mlx.model_execution.cache' for c in expected_list)
    def turn(suffix, count, cancel=False):
        before = list(session.token_history); live = session.live_cache
        t = time.perf_counter()
        session.begin_turn_from_recipe_tokens(before+suffix, max_tokens=count)
        bootstrap = time.perf_counter()-t
        gen = session.generation; reports=[]; t=time.perf_counter()
        for _ in range(7 if cancel else count):
            r = session.next_token()
            if r is None: break
            reports.append(r.to_json())
            if r.finish_reason: break
        decode_s=time.perf_counter()-t
        t=time.perf_counter()
        if cancel: session.cancel_turn('qualification_cancel')
        else: session.ensure_idle()
        settle_s=time.perf_counter()-t
        idle_check(live)
        assert session.token_history == before+suffix+[r['token'] for r in reports]
        assert gen._cache is None and gen._pending is None and gen._old_wired_limit is None
        row=dict(frontier_before=len(before), frontier_after=session.frontier, suffix_tokens=len(suffix),
                 suffix_sha256=digest_ids(suffix), bootstrap_s=bootstrap, decode_s=decode_s,
                 decode_tok_s=len(reports)/decode_s, settle_s=settle_s, cancelled=cancel,
                 reports=reports, memory=memory(), same_list=True, replay=0, repack=0)
        result['turns'].append(row); save(); progress('turn-complete')
        return row
    try:
        progress('load'); t=time.perf_counter(); backend.load()
        result['load_s']=time.perf_counter()-t
        model=backend._model; lm=model.language_model; tokenizer=backend._runtime.processor.tokenizer
        assert type(lm).__module__ == 'ds41f_mlx.model_execution.language'
        result['admission']=backend._runtime.admission.describe()
        result['configured_max_seq_len']=lm._config.max_seq_len
        result['loaded_memory']=memory()
        dc=OMLXDecodeConfig(preserve_mtp=False, omlx_path=cfg.omlx_path)
        probe_suffix=tokenizer.encode('\nOperational checkpoint: summarize the invariants of the committed session.\n')
        if a.restore:
            source=json.loads(a.restore.read_text()); assert source['status']=='PASS'
            t=time.perf_counter()
            cache, ids, _=restore_m8_idle_state(artifact_path=Path(source['artifact']), model=model,
                checkpoint=cfg.checkpoint_path, omlx_path=cfg.omlx_path)
            result['restore_s']=time.perf_counter()-t
            assert slots(cache)==source['persisted_slots'] and digest_ids(ids)==source['persisted_history_sha256']
            session=M8LiveContinuationSession.from_live_cache(model=model, live_cache=cache, token_history=ids, config=dc, mx=mx)
            row=turn(probe_suffix, 32)
            assert [r['token'] for r in row['reports']]==source['probe_tokens']
            assert slots(session.live_cache)==source['probe_slots']
            result['fresh_process_restore_exact']=True
            if a.restore_probe_only:
                # The last live branch already consumed the remaining capacity.
                # Reproduce it exactly; do not manufacture a post-ceiling suffix.
                for key in ('persisted_slots', 'persisted_history_sha256', 'persisted_frontier',
                            'artifact', 'artifact_bytes', 'probe_tokens', 'probe_slots'):
                    result[key]=source[key]
                result.update(save_s=0.0, save_performed=False, restored_artifact_only=True,
                              final_frontier=session.frontier, diagnostics=session.diagnostics(), status='PASS')
                return
        else:
            # Exercise the actual recipe serving infer path as well as its long-lived runtime seams.
            options=types.SimpleNamespace(temperature=0.0, top_p=0.0, max_tokens=16)
            req=RecipePreparedRequest('chat_completions', types.SimpleNamespace(model=None, stream=True, inference_options=options),
                                      None, tokenizer.encode('Explain why committed cache state matters.'), [])
            async def serve():
                async for _ in backend.infer(req): pass
            asyncio.run(serve()); trace=backend.last_trace.to_json()
            assert trace['cleanup_called'] and trace['same_live_cache_handoff']
            assert trace['prompt_replay_count']==trace['full_cache_repack_count']==0
            result['serving_trace']=trace
            # Mixed real repository documents/code and uniquely numbered operational journal entries.
            corpus_files=['docs/architecture.md','docs/session-state.md','docs/engram.md',
                          'ds41f_mlx/runtime/continuation_session.py','docs/operations.md']
            corpus='\n'.join(subprocess.check_output(['git','show',f'{a.fixture_revision}:{f}'], text=True)
                             if a.fixture_revision else (ROOT/f).read_text() for f in corpus_files)
            result['fixture_corpus_revision']=a.fixture_revision
            text='Review this runtime maintenance journal and retain its operational constraints.\n'
            sections=max(80, a.context//len(tokenizer.encode(corpus))+2)
            text+='\n'.join(f'Journal section {i}:\n{corpus}\nEnd section {i}.\n' for i in range(sections))
            ids=tokenizer.encode(text)[:a.context]; assert len(ids)==a.context
            result['fixture']=dict(kind='repository text and numbered maintenance journal', count=len(ids), sha256=digest_ids(ids))
            result['before_prefill_memory']=memory(); save()
            progress('prefill'); pre=DwarfStarMLXPrefillSession(model, omlx_path=cfg.omlx_path).prefill(ids[:-1])
            result['after_prefill_memory']=memory(); progress('prefill-complete'); save()
            result['prefill']=pre.to_json(); result['prefill']['tok_s']=(len(ids)-1)/pre.seconds
            live=pre.live_result.live_cache; t=time.perf_counter()
            session=M8LiveContinuationSession.from_prefill_result(model=model, live_result=pre.live_result,
                terminal_prompt_token=ids[-1], config=dc, max_tokens=a.decode_tokens)
            bootstrap=time.perf_counter()-t; gen=session.generation; t=time.perf_counter()
            reps=[r.to_json() for r in gen.generate(a.decode_tokens)]; seconds=time.perf_counter()-t
            session.ensure_idle(); idle_check(live)
            assert session.token_history==ids+[r['token'] for r in reps]
            result['initial_decode']=dict(bootstrap_s=bootstrap, decode_s=seconds, decode_tok_s=len(reps)/seconds, reports=reps, memory=memory())
            assert pre.live_result.handoff_count==1
            result['p5']=dict(handoff_count=1, same_list=True, exported=False, replay=0, repack=0)
            del pre
        for i in range(a.turns):
            # Alternate tiny follow-ups and realistic tool/document result-sized appends.
            size=[1,17,257,2049][i%4]
            text=('\nTool result %d: inspect the committed frontier, SSD Engram lookup lifecycle and persistence manifest. '%i)*(size+1)
            suffix=tokenizer.encode(text)[:size]; assert len(suffix)==size
            # Invalid exact-prefix admission must leave the sole authority untouched.
            live=session.live_cache; before=session.frontier
            try: session.begin_turn_from_recipe_tokens([0]+session.token_history[1:]+suffix)
            except M8ContinuationError: pass
            else: raise AssertionError('non-prefix admitted')
            idle_check(live); assert session.frontier==before
            turn(suffix, a.decode_tokens, cancel=i in (3,11))
        live=session.live_cache; result['persisted_slots']=slots(live)
        result['persisted_history_sha256']=digest_ids(session.token_history)
        result['persisted_frontier']=session.frontier
        t=time.perf_counter(); artifact=save_m8_idle_state(artifact_root=a.artifact_root, model=model,
            live_cache=live, all_tokens=session.token_history, checkpoint=cfg.checkpoint_path, omlx_path=cfg.omlx_path)
        result['save_s']=time.perf_counter()-t; result['artifact']=str(artifact.path)
        result['artifact_bytes']=sum(f.stat().st_size for f in artifact.path.rglob('*') if f.is_file())
        idle_check(live)
        row=turn(probe_suffix,32)
        result['probe_tokens']=[r['token'] for r in row['reports']]; result['probe_slots']=slots(session.live_cache)
        result['final_frontier']=session.frontier
        result['diagnostics']=session.diagnostics()
        if a.compare:
            previous=json.loads(a.compare.read_text()); assert previous['status']=='PASS'
            keys=['persisted_frontier','persisted_history_sha256','persisted_slots','probe_tokens','probe_slots','final_frontier']
            if not a.restore: keys.append('fixture')
            for key in keys:
                assert result[key]==previous[key], key
            for old,new in zip(previous['turns'],result['turns'],strict=True):
                assert old['suffix_sha256']==new['suffix_sha256']
                assert [r['token'] for r in old['reports']]==[r['token'] for r in new['reports']]
            if not a.restore:
                assert [r['token'] for r in previous['initial_decode']['reports']]==[r['token'] for r in result['initial_decode']['reports']]
            result['pre_policy_comparison']='exact tokens, history, persisted/probe all 280 slots'
        result['status']='PASS'
    except BaseException as exc:
        result.update(status='FAIL', error=repr(exc)); raise
    finally:
        if session is not None: session.close()
        backend.close(); stop.set(); observer.join()
        result['closed_memory']=memory(); result['wall_s']=time.perf_counter()-start
        result['backend_active_sessions']=backend.active_generation_sessions
        current=mx.set_cache_limit(0); mx.set_cache_limit(current)
        result['retired_allocator_cache_limit_bytes']=current
        wired=mx.set_wired_limit(0); mx.set_wired_limit(wired)
        result['retired_wired_limit_bytes']=wired
        if result['status']=='PASS':
            assert current==result['admission']['previous_allocator_cache_limit_bytes']
            assert wired==result['admission']['previous_wired_limit_bytes']
        save()


if __name__=='__main__': main()
