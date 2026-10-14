#!/usr/bin/env python3
"""Reproduce only the first saved 200K OFF 2K suffix, including its warm-up.

Use installed Python -I for installed controls. A historical native resource
control may use explicit preserved dependency origins and the current first-party
core; it is not installed-native publication acceptance. No fresh prefix, ladder,
cache repair or observer synchronization. Preceding decoded tokens must equal the
historical receipt; record/assert the GLM profile, not just the serving profile.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--expected',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expect-glm-profile', choices=['native', 'portable'])
    a=p.parse_args()
    source=json.loads(a.source.read_text()); expected=json.loads(a.expected.read_text())
    result=dict(status='RUNNING', turns=[], events=[])
    def save(): a.output.write_text(json.dumps(result,indent=2,default=str)+'\n')
    from ds41f_mlx.config import load_runtime_config
    cfg=load_runtime_config();cfg.apply_import_paths()
    from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend
    backend=DeepSeekRecipeRuntimeBackend(runtime_config=cfg)
    async def run():
        t=time.perf_counter();backend.load();result['load_s']=time.perf_counter()-t
        result['admission']=backend._runtime.admission.describe()
        if a.expect_glm_profile:
            assert result['admission']['native_profile']==a.expect_glm_profile
        save()
        import mlx.core as mx
        from mlx_lm.generate import generation_stream
        from ds41f_mlx.runtime.kv_persistence import restore_m8_idle_state
        from ds41f_mlx.runtime.continuation_session import M8LiveContinuationSession
        from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
        from ds41f_mlx.prefill_fp8_mlx import DeferredPrefillAppend
        active=[False]
        original=DeferredPrefillAppend.execute_all
        def observed(app):
            t=time.perf_counter()
            try: return original(app)
            finally:
                if active[0]:result['events'].append(dict(name='execute_all',C=app.C,T=app.T,seconds=time.perf_counter()-t))
        DeferredPrefillAppend.execute_all=observed
        def core():
            from contextlib import nullcontext
            with nullcontext():  # Historical collector used the main thread/default stream.
                cache,ids,_=restore_m8_idle_state(artifact_path=Path(source['artifact']),model=backend._model,
                    checkpoint=cfg.checkpoint_path,omlx_path=cfg.omlx_path)
                assert hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==source['persisted_history_sha256']
                import numpy as np
                slots=[dict(layer=i,slot=j,shape=list(x.shape),dtype=str(x.dtype),
                    sha256=hashlib.sha256(np.asarray(x.view(mx.uint8)).tobytes()).hexdigest())
                    for i,c in enumerate(cache) for j,x in enumerate(c.cache)]
                assert slots==source['persisted_slots'], 'historical saved slots differ'
                result['saved_slots_exact']=True
                session=M8LiveContinuationSession.from_live_cache(model=backend._model,live_cache=cache,token_history=ids,
                    config=OMLXDecodeConfig(preserve_mtp=False,omlx_path=cfg.omlx_path),mx=mx)
                tok=backend._runtime.processor.tokenizer
                def turn(label,suffix,count,want=None):
                    active[0]=True
                    before=session.frontier;t=time.perf_counter()
                    session.begin_turn_from_suffix(suffix,max_tokens=count)
                    bootstrap=time.perf_counter()-t
                    reports=[session.next_token() for _ in range(count)]
                    session.ensure_idle();active[0]=False
                    tokens=[r.token for r in reports if r]
                    row=dict(label=label,frontier=before,suffix_tokens=len(suffix),bootstrap_s=bootstrap,
                        record=session.turn_records[-1].to_json(),tokens=tokens,
                        active_memory=mx.get_active_memory(),cache_memory=mx.get_cache_memory())
                    result['turns'].append(row);save()
                    if want is not None: assert tokens==want[:count],(label,tokens,want[:count])
                probe=tok.encode('\nOperational checkpoint: summarize the invariants of the committed session.\n')
                assert hashlib.sha256(json.dumps(probe,separators=(',',':')).encode()).hexdigest()==expected['turns'][0]['suffix_sha256']
                turn('restore-probe',probe,32,source['probe_tokens'])
                probe_slots=[dict(layer=i,slot=j,shape=list(x.shape),dtype=str(x.dtype),
                    sha256=hashlib.sha256(np.asarray(x.view(mx.uint8)).tobytes()).hexdigest())
                    for i,c in enumerate(session.live_cache) for j,x in enumerate(c.cache)]
                assert probe_slots==source['probe_slots'], 'historical post-probe slots differ'
                result['probe_slots_exact']=True;save()
                for i,size in enumerate([1,17,257,2049]):
                    suffix=tok.encode(('\nTool result %d: inspect the committed frontier, SSD Engram lookup lifecycle and persistence manifest. '%i)*(size+1))[:size]
                    old=expected['turns'][i+1]  # first receipt row is the restore probe
                    assert session.frontier==old['frontier_before'],(i,session.frontier,old['frontier_before'])
                    assert hashlib.sha256(json.dumps(suffix,separators=(',',':')).encode()).hexdigest()==old['suffix_sha256']
                    count=len(old['reports']) if i<3 else 2
                    turn(f'historical-turn-{i}',suffix,count,[r['token'] for r in old['reports']])
                # One same-shape warm continuation separates a new-shape first
                # use from persistent long-frontier suffix cost.
                turn('warm-same-suffix',suffix,2)
                result['diagnostics']=session.diagnostics()
                result['admission']=backend._runtime.admission.describe()
                session.close()
        core()
        backend.close();result['status']='PASS';save()
    try:asyncio.run(run())
    except BaseException as exc:
        result['status']='FAIL';result['error']=repr(exc);save();raise

if __name__=='__main__':main()
