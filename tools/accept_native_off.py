"""Bounded installed OFF publication acceptance; no long-context campaign.

Run with deployment Python -I. Uses production backend/P5/P6 owners, forwards
unchanged native calls, and excludes admission smoke from selected-path proof.
"""
import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from ds41f_mlx.execution_identity import inspect_payload
    authority, compatibility = inspect_payload()
    assert authority['profile'] == 'standard-off'
    assert compatibility['glm_profile'] == 'native'
    import ds41f_mlx
    manifest = Path(ds41f_mlx.__file__).parent/'execution_manifest.json'
    from omlx.custom_kernels.glm_moe_dsa import fast
    originals = {name: getattr(fast._ext, name) for name in fast.NATIVE_SYMBOLS}
    counts = Counter(); examples = {}; stage = 'prefill_p5'
    def observe(name, original):
        def call(*a, **kw):
            frames = inspect.stack(context=0)
            callers = [f for f in frames[1:] if any(part in f.filename for part in
                ('/ds41f_mlx/model_execution/', '/ds41f_mlx/prefill_fp8_mlx/', '/ds41f_mlx/runtime/'))
                and not f.filename.endswith('resource_admission.py')]
            value = original(*a, **kw)
            if callers:
                counts[name] += 1
                shapes = [list(x.shape) for x in a if hasattr(x, 'shape')]
                key = stage+':'+name
                if any('p6' in f.filename for f in callers): key += ':p6'
                if key not in examples:
                    examples[key] = dict(shapes=shapes, callers=[dict(file=f.filename,
                        function=f.function, line=f.lineno) for f in callers],
                        layer_indices=[f.frame.f_locals.get('layer_index', f.frame.f_locals.get('i'))
                                       for f in callers])
            del frames, callers
            return value
        return call
    from ds41f_mlx.config import load_runtime_config
    from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend
    from ds41f_mlx.runtime.dwarfstar_prefill import DwarfStarMLXPrefillSession
    from ds41f_mlx.runtime.continuation_session import M8LiveContinuationSession
    from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
    import mlx.core as mx
    cfg = load_runtime_config(); cfg.apply_import_paths()
    backend = DeepSeekRecipeRuntimeBackend(runtime_config=cfg)
    session = None
    receipt = dict(status='RUNNING', manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                   bundle=authority['glm'], compatibility=compatibility)
    try:
        backend.load()
        admission = backend._runtime.admission
        receipt['admission'] = admission.describe()
        model = backend._model
        tokenizer = backend._runtime.processor.tokenizer
        for name, original in originals.items(): setattr(fast._ext, name, observe(name, original))
        ids = tokenizer.encode('Explain committed cache ownership and frontier alignment. ' * 80)[:256]
        assert len(ids) == 256
        pre = DwarfStarMLXPrefillSession(model, omlx_path=cfg.omlx_path).prefill(ids[:-1])
        live = pre.live_result.live_cache
        session = M8LiveContinuationSession.from_prefill_result(model=model, live_result=pre.live_result,
            terminal_prompt_token=ids[-1], config=OMLXDecodeConfig(preserve_mtp=False, omlx_path=cfg.omlx_path),
            max_tokens=8)
        def settle(before, suffix):
            gen = session.generation
            reports = [r.to_json() for r in gen.generate(8)]
            session.ensure_idle()
            assert session.state == 'idle' and session.live_cache is live
            assert len(live) == 40 and all(c.size() == session.frontier for c in live)
            assert all(type(c).__module__ == 'ds41f_mlx.model_execution.cache' for c in live)
            assert session.token_history == before+suffix+[r['token'] for r in reports]
            assert session.total_prompt_replay_count == session.total_full_cache_repack_count == 0
            assert gen._cache is None and gen._pending is None and gen._old_wired_limit is None
            return dict(frontier=session.frontier, same_list=True, replay=0, repack=0, reports=reports,
                        diagnostics=session.diagnostics())
        receipt['p5'] = settle([], ids)
        assert pre.live_result.handoff_count == 1
        receipt['p5']['handoff_count'] = 1
        before = list(session.token_history)
        suffix = tokenizer.encode('\nContinue explaining the exact committed frontier and resource lifetime. ' * 4)[:17]
        assert len(suffix) == 17
        prior = dict(counts)
        stage = 'idle_suffix_p6'
        session.begin_turn_from_recipe_tokens(before+suffix, max_tokens=8)
        receipt['p6'] = settle(before, suffix)
        receipt['p6']['suffix_tokens'] = len(suffix)
        receipt['p6']['native_call_delta'] = {k: v-prior.get(k, 0) for k,v in counts.items()}
        assert receipt['p6']['native_call_delta'].get('deepseek_v41_packed_attention', 0) > 0
        suffix_attention = examples['idle_suffix_p6:deepseek_v41_packed_attention:p6']
        # P6 appends all suffix rows except its held-out terminal P5 token.
        assert suffix_attention['shapes'][0][2] == len(suffix)-1
        assert any(v > 0 and ('expert' in k or 'moe' in k) for k,v in counts.items())
        receipt['successful_model_native_calls'] = dict(counts)
        receipt['selected_path_examples'] = examples
        receipt['status'] = 'PASS'
    except BaseException as exc:
        receipt.update(status='FAIL', error=repr(exc)); raise
    finally:
        for name, original in originals.items(): setattr(fast._ext, name, original)
        if session is not None: session.close()
        backend.close()
        cache = mx.set_cache_limit(0); mx.set_cache_limit(cache)
        wired = mx.set_wired_limit(0); mx.set_wired_limit(wired)
        receipt['retirement'] = dict(active_sessions=backend.active_generation_sessions,
            allocator_cache_limit_bytes=cache, wired_limit_bytes=wired)
        if receipt['status'] == 'PASS':
            assert backend.active_generation_sessions == 0
            assert cache == receipt['admission']['previous_allocator_cache_limit_bytes']
            assert wired == receipt['admission']['previous_wired_limit_bytes']
        args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')


if __name__ == '__main__': main()
