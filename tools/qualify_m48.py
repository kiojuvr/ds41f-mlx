"""Bounded matched model-execution migration check, not a production selector.

Oracle runs in its own process using the frozen M47 lifetime/admission modules.
Owned runs reject imports/calls into displaced numerical modules, including load.
Both exercise the actual serving backend and fresh dense/P5/target/idle paths.
"""
import argparse
import asyncio
from collections import Counter
import hashlib
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import types

BASELINE = 'bdd9e9c6107b5465f85cbaca7afadfaa73d7273c'
NUMERICAL = set('activation cache config convert engram head hyper_connection kernels language loading model mtp packed_attention processing quantization routing sharding storage vision'.split())


def displaced(name):
    return (name.startswith('omlx.patches.deepseek_v41.') and name.rsplit('.', 1)[-1] in NUMERICAL
            or name == 'omlx.patches.deepseek_v4.switch_layers')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--contexts', type=int, nargs='+', default=[4096, 32768])
    parser.add_argument('--tokens', type=int, default=64)
    args = parser.parse_args()
    counts = Counter()
    scratch = tempfile.TemporaryDirectory(prefix='ds41f-m48-')
    if args.oracle:
        # Evidence only: frozen M47 source, never an OFF production switch.
        for leaf in ('resource_admission', 'omlx_core', 'kv_persistence'):
            name = 'ds41f_mlx.runtime.' + leaf
            path = 'ds41f_mlx/runtime/' + leaf + '.py'
            source = subprocess.check_output(['git', 'show', BASELINE + ':' + path])
            module = types.ModuleType(name)
            module.__file__ = str(Path(path).resolve())
            sys.modules[name] = module
            exec(compile(source, module.__file__, 'exec'), module.__dict__)
            if leaf == 'resource_admission':
                pin = Path(scratch.name)/'m47-pin.json'
                pin.write_bytes(subprocess.check_output(['git', 'show', BASELINE + ':ds41f_mlx/runtime/admitted_resources.json']))
                module.PIN = pin
    else:
        class NoDonor(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if displaced(fullname):
                    raise AssertionError('displaced model import: ' + fullname)
        sys.meta_path.insert(0, NoDonor())

    def observe(frame, event, arg):
        if event != 'call': return
        name = frame.f_globals.get('__name__', '')
        if not args.oracle and displaced(name):
            raise AssertionError('displaced numerical execution: ' + name)
        if name.startswith('ds41f_mlx.model_execution.'):
            counts[name + ':' + frame.f_code.co_qualname] += 1

    from ds41f_mlx.config import load_runtime_config
    cfg = load_runtime_config(); cfg.apply_import_paths()
    from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend, RecipePreparedRequest
    from ds41f_mlx.runtime.dwarfstar_prefill import DwarfStarMLXPrefillSession
    from ds41f_mlx.prefill_fp8_mlx import handoff_to_generation
    from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
    from ds41f_mlx.runtime.continuation_session import M8LiveContinuationSession
    from ds41f_mlx.runtime.kv_persistence import save_m8_idle_state, restore_m8_idle_state
    import mlx.core as mx
    import numpy as np
    backend = DeepSeekRecipeRuntimeBackend(runtime_config=cfg)
    result = dict(schema='ds41f.m48.model-execution.v1', status='RUNNING', oracle=args.oracle,
                  baseline=BASELINE, contexts=args.contexts, tokens=args.tokens, runs=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save(): args.output.write_text(json.dumps(result, indent=2)+'\n')
    def inventory(cache):
        return [dict(layer=i, slot=j, shape=list(x.shape), dtype=str(x.dtype),
                     sha256=hashlib.sha256(np.asarray(x.view(mx.uint8)).tobytes()).hexdigest())
                for i,c in enumerate(cache) for j,x in enumerate(c.cache)]
    try:
        # Observe checkpoint construction too. Thread profile covers real infer.
        sys.setprofile(observe); threading.setprofile(observe)
        backend.load()
        model = backend._model; lm = model.language_model
        tokenizer = backend._runtime.processor.tokenizer
        result['admission'] = backend._runtime.admission.describe()
        result['model_class'] = type(lm).__module__ + '.' + type(lm).__name__
        if not args.oracle:
            assert type(lm).__module__ == 'ds41f_mlx.model_execution.language'
            assert not any(displaced(n) for n in sys.modules)
        options = types.SimpleNamespace(temperature=0.0, top_p=0.0, max_tokens=16)
        request = RecipePreparedRequest('chat_completions',
            types.SimpleNamespace(model=None, stream=True, inference_options=options), None,
            tokenizer.encode('Explain why a computer uses a cache.'), [])
        async def serve():
            async for _ in backend.infer(request): pass
        asyncio.run(serve())
        trace = backend.last_trace.to_json()
        assert trace['cleanup_called'] and not backend.active_generation_sessions
        assert trace['same_live_cache_handoff'] and trace['handoff_count'] == 1
        assert trace['prompt_replay_count'] == trace['full_cache_repack_count'] == 0
        result['serving_trace'] = trace
        decode_cfg = OMLXDecodeConfig(preserve_mtp=False, omlx_path=cfg.omlx_path)
        for n in args.contexts:
            ids = tokenizer.encode('Explain the purpose of a computer cache. ' * (n//4+100))[:n]
            assert len(ids) == n
            pre = DwarfStarMLXPrefillSession(model, omlx_path=cfg.omlx_path).prefill(ids[:-1])
            live = pre.live_result.live_cache
            generation = handoff_to_generation(pre.live_result, model, terminal_prompt_token=ids[-1],
                config=decode_cfg, max_tokens=args.tokens)
            tokens = [r.token for r in generation.generate(args.tokens)]
            cache, history = generation.extract_final_state()
            assert cache is live and history == ids+tokens
            assert all(c.size() == len(history) for c in cache)
            assert pre.live_result.handoff_count == 1 and generation.prompt_replay_count == 0
            assert pre.live_result._setup.block_runner.full_cache_repack_count == 0
            row = dict(context=n, tokens=tokens, slots=inventory(cache),
                       timing=generation.timing_summary(), same_live_list=True,
                       cache_classes=sorted({type(c).__module__ for c in cache}))
            generation.close()
            artifact = save_m8_idle_state(artifact_root=Path(scratch.name)/str(n), model=model,
                live_cache=cache, all_tokens=history, checkpoint=cfg.checkpoint_path, omlx_path=cfg.omlx_path)
            restored, restored_ids, _ = restore_m8_idle_state(artifact_path=artifact.path, model=model,
                checkpoint=cfg.checkpoint_path, omlx_path=cfg.omlx_path)
            assert inventory(restored) == row['slots'] and restored_ids == history
            continuation = M8LiveContinuationSession.from_live_cache(model=model, live_cache=restored,
                token_history=history, config=decode_cfg, mx=mx)
            suffix = tokenizer.encode(' Summarize that briefly.')
            continuation.begin_turn_from_suffix(suffix, max_tokens=8)
            continued = [r.token for r in continuation.generation.generate(8)]
            continuation.ensure_idle()
            assert continuation.live_cache is restored
            assert continuation.token_history == history+suffix+continued
            assert continuation.total_prompt_replay_count == continuation.total_full_cache_repack_count == 0
            row.update(continued_tokens=continued, continued_slots=inventory(restored),
                       persistence_exact=True, idle_append_same_list=True)
            continuation.close()
            result['runs'].append(row); save()
        if args.compare:
            oracle = json.loads(args.compare.read_bytes())
            assert oracle['status'] == 'PASS' and oracle['oracle']
            assert oracle['serving_trace']['generated_tokens'] == trace['generated_tokens']
            for a,b in zip(oracle['runs'], result['runs'], strict=True):
                for key in ('context','tokens','slots','continued_tokens','continued_slots'):
                    assert a[key] == b[key], key
            result['comparison'] = 'exact serving tokens, fresh tokens and all 40x7 slots, restored/idle-append tokens and all 40x7 slots'
        result['owned_calls'] = dict(sorted(counts.items()))
        if not args.oracle:
            for module in ('loading','language','quantization','engram','storage','kernels','hyper_connection','head','switch_layers'):
                assert any(k.startswith('ds41f_mlx.model_execution.'+module+':') for k in counts), module
            result['displaced_imports_and_calls_forbidden'] = True
        result['status'] = 'PASS'
    except BaseException as exc:
        result.update(status='FAIL', error=repr(exc)); raise
    finally:
        sys.setprofile(None); threading.setprofile(None)
        backend.close(); scratch.cleanup(); save()


if __name__ == '__main__': main()
