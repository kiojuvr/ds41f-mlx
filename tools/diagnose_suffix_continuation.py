#!/usr/bin/env python3
"""Bounded installed-runtime suffix phase observer (no extra evaluation barriers).

Run MTP then OFF, sequentially, with each installed Python -I. The MTP run
writes the tokenizer-derived core fixture shared by OFF. This is a token-level
continuation probe, not a reconstructed HTTP/OpenCode conversation acceptance.
"""
import argparse
import asyncio
import functools
import hashlib
import inspect
import json
from pathlib import Path
import time
import types


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=['mtp-serving-v1', 'standard-off'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--historical-receipt', type=Path, help='OFF only: one 2049-token control on saved historical 200K state')
    a = p.parse_args()
    result = dict(profile=a.profile, status='RUNNING', events=[], turns=[])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        a.output.write_text(json.dumps(result, indent=2, default=str)+'\n')
    active = [False]
    stack = []
    def wrap(host, name):
        old = getattr(host, name)
        @functools.wraps(old)
        def call(*args, **kwargs):
            if not active[0]:
                return old(*args, **kwargs)
            start = time.perf_counter()
            parent = list(stack)
            stack.append(name)
            row = dict(name=name, parent=parent)
            if name == 'execute_all':
                app = args[0]
                row.update(C=app.C, T=app.T, segments=[dict(start=s.start,end=s.end,mode=s.mode.value) for s in app.plan.segments])
            if name in ('eval', 'synchronize'):
                frame = inspect.currentframe().f_back
                row['caller'] = f'{frame.f_code.co_filename}:{frame.f_lineno}:{frame.f_code.co_name}'
            try:
                return old(*args, **kwargs)
            finally:
                stack.pop()
                row.update(start=start, seconds=time.perf_counter()-start)
                result['events'].append(row)
        setattr(host, name, call)
    if a.profile == 'mtp-serving-v1':
        from ds41f_mlx.mtp_identity import config, inspect as admission
        cfg = config(profile=a.profile)
        report = admission(cfg)
        cfg.apply_environment()
        from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
        backend = ProductionMTPBackend(runtime_config=cfg)
        backend.dependency_identity = report['identity_sha256']
        result['identity'] = report['identity_sha256']
    else:
        from ds41f_mlx.config import load_runtime_config
        cfg = load_runtime_config()
        cfg.apply_import_paths()
        from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend
        backend = DeepSeekRecipeRuntimeBackend(runtime_config=cfg)
    import mlx.core as mx
    from mlx_lm.generate import generation_stream
    import ds41f_mlx
    result['package'] = ds41f_mlx.__file__
    from ds41f_mlx.prefill_fp8_mlx import DeferredPrefillAppend
    wrap(DeferredPrefillAppend, 'execute_all')
    wrap(mx, 'eval')
    wrap(mx, 'synchronize')
    def memory():
        return dict(active=mx.get_active_memory(), cache=mx.get_cache_memory())
    async def run():
        t = time.perf_counter()
        await backend._call(backend.load)
        result['load_s'] = time.perf_counter()-t
        result['loaded_memory'] = memory()
        save()
        lm = backend._model.language_model
        if a.profile == 'mtp-serving-v1':
            from ds41f_mlx.serving.server import load_v41_tokenizer, prepare_request
            from ds41f_mlx.serving.paired_checkpoint import PairedCheckpointAuthority
            tok = load_v41_tokenizer(cfg.recipe_path)
            wrap(type(lm), 'dspark_append_context')
            wrap(PairedCheckpointAuthority, 'capture')
            wrap(PairedCheckpointAuthority, 'acquire')
            text = ''.join(f'Maintenance journal {i}: preserve committed cache state, exact token history, zero replay and zero repack.\n' for i in range(18000))
            prefix = list(tok.encode(text))[:209000]
            suffix = list(tok.encode(text))[209000:209000+1773]
            assert len(prefix) == 209000 and len(suffix) == 1773
            fixture = dict(prefix=prefix, suffix=suffix)
            a.fixture.write_text(json.dumps(fixture))
            result['fixture_sha256'] = hashlib.sha256(a.fixture.read_bytes()).hexdigest()
            def prepared(ids):
                body = dict(model='deepseek-v4.1-flash', temperature=0, max_tokens=2,
                            messages=[dict(role='user', content='Briefly describe committed state.')])
                req = prepare_request('chat_completions', json.dumps(body).encode(), tokenizer=tok,
                                      recipe_path=cfg.recipe_path, checkpoint=cfg.checkpoint_path, ordinary=True)
                req.token_ids = ids  # Explicit core fixture, not an HTTP compatibility claim.
                return req
            t = time.perf_counter()
            await backend.ordinary_response(prepared(prefix+list(tok.encode('Unrelated bootstrap.'))[:2]), tokenizer=tok)
            result['prefix_wall_s'] = time.perf_counter()-t
            result['prefix_trace'] = backend.serving_traces[-1]
            result['ready_memory'] = memory()
            save()
            active[0] = True
            t = time.perf_counter()
            response = await backend.ordinary_response(prepared(prefix+suffix), tokenizer=tok)
            active[0] = False
            result['turns'].append(dict(wall_s=time.perf_counter()-t, trace=backend.serving_traces[-1], response=json.loads(response.body)))
            assert result['turns'][0]['trace']['cached_tokens'] == len(prefix)
            await backend._call(backend.scheduler.close_serving)
            backend.close()
        else:
            fixture = json.loads(a.fixture.read_text())
            result['fixture_sha256'] = hashlib.sha256(a.fixture.read_bytes()).hexdigest()
            prefix, suffix = fixture['prefix'], fixture['suffix']
            from ds41f_mlx.runtime.dwarfstar_prefill import DwarfStarMLXPrefillSession
            from ds41f_mlx.runtime.continuation_session import M8LiveContinuationSession
            from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
            def core():
                with mx.stream(generation_stream):
                    t = time.perf_counter()
                    prefill = DwarfStarMLXPrefillSession(backend._model, omlx_path=cfg.omlx_path).prefill(prefix[:-1])
                    result['prefix_wall_s'] = time.perf_counter()-t
                    session = M8LiveContinuationSession.from_prefill_result(model=backend._model,
                        live_result=prefill.live_result, terminal_prompt_token=prefix[-1], max_tokens=2,
                        config=OMLXDecodeConfig(preserve_mtp=False, omlx_path=cfg.omlx_path))
                    session.ensure_idle()  # Retire the producer through P5; no output consumed.
                    assert session.token_history == prefix
                    result['ready_memory'] = memory()
                    save()
                    active[0] = True
                    t = time.perf_counter()
                    session.begin_turn_from_suffix(suffix, max_tokens=2)
                    bootstrap = time.perf_counter()-t
                    t = time.perf_counter()
                    reports = [session.next_token() for _ in range(2)]
                    decode = time.perf_counter()-t
                    t = time.perf_counter()
                    session.ensure_idle()
                    active[0] = False
                    result['turns'].append(dict(bootstrap_s=bootstrap, decode_s=decode,
                        settle_s=time.perf_counter()-t, record=session.turn_records[-1].to_json(),
                        reports=[r.to_json() if r else None for r in reports], diagnostics=session.diagnostics()))
                    session.close()
                    if a.historical_receipt:
                        from ds41f_mlx.runtime.kv_persistence import restore_m8_idle_state
                        source = json.loads(a.historical_receipt.read_text())
                        cache, ids, _ = restore_m8_idle_state(artifact_path=Path(source['artifact']),
                            model=backend._model, checkpoint=cfg.checkpoint_path, omlx_path=cfg.omlx_path)
                        digest = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
                        assert digest == source['persisted_history_sha256']
                        historical = M8LiveContinuationSession.from_live_cache(model=backend._model,
                            live_cache=cache, token_history=ids, mx=mx,
                            config=OMLXDecodeConfig(preserve_mtp=False, omlx_path=cfg.omlx_path))
                        tok = backend._runtime.processor.tokenizer
                        text = ('\nTool result 3: inspect the committed frontier, SSD Engram lookup lifecycle and persistence manifest. ')*2050
                        old_suffix = list(tok.encode(text))[:2049]
                        active[0] = True
                        t = time.perf_counter()
                        historical.begin_turn_from_suffix(old_suffix, max_tokens=2)
                        bootstrap = time.perf_counter()-t
                        for _ in range(2):
                            historical.next_token()
                        historical.ensure_idle()
                        active[0] = False
                        result['historical_control'] = dict(frontier=len(ids), suffix_tokens=len(old_suffix),
                            suffix_sha256=hashlib.sha256(json.dumps(old_suffix, separators=(',', ':')).encode()).hexdigest(),
                            bootstrap_s=bootstrap, record=historical.turn_records[-1].to_json(), diagnostics=historical.diagnostics())
                        historical.close()
            await backend._call(core)
            backend.close()
        result['final_memory'] = memory()
        result['status'] = 'PASS'
        save()
    try:
        asyncio.run(run())
    except BaseException as exc:
        result['status'] = 'FAIL'
        result['error'] = repr(exc)
        save()
        raise

if __name__ == '__main__':
    main()
