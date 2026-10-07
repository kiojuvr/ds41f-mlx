"""Real-checkpoint promotion blocker probe; never a runtime selector.

No admission/loader/transaction guard is bypassed. Unsupported validation is
observation-only on idle state, not an attempted speculative mutation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from ds41f_mlx.config import load_runtime_config
    cfg = load_runtime_config()
    cfg.apply_import_paths()
    import mlx.core as mx
    import numpy as np
    from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend
    from ds41f_mlx.runtime.dwarfstar_prefill import DwarfStarMLXPrefillSession
    from ds41f_mlx.prefill_fp8_mlx import handoff_to_generation
    from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
    from ds41f_mlx.runtime.target_generation import TargetGenerationSession
    from ds41f_mlx.model_execution.loading import load

    result = dict(schema='ds41f.m49.ownership-blocker.v1', status='RUNNING',
                  base=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  checkpoint=str(cfg.checkpoint_path), probes=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    backend = DeepSeekRecipeRuntimeBackend(runtime_config=cfg)
    generation = None

    def reject(name, fn):
        try:
            fn()
        except (ValueError, RuntimeError) as exc:
            result['probes'].append(dict(name=name, rejected=True, error=str(exc)))
        else:
            raise AssertionError('unexpected support: ' + name)

    def digest(cache):
        mx.eval(*(x for c in cache for x in c.cache if x is not None))
        return [dict(layer=i, slot=j, shape=list(x.shape), dtype=str(x.dtype),
                     sha256=hashlib.sha256(np.asarray(x.view(mx.uint8)).tobytes()).hexdigest())
                for i, c in enumerate(cache) for j, x in enumerate(c.cache)]

    try:
        reject('first_party_mtp_load', lambda: load(cfg.checkpoint_path, preserve_mtp=True))
        start = perf_counter()
        backend.load()
        result['load_s'] = perf_counter() - start
        model = backend._model
        lm = model.language_model
        result['admission'] = backend._runtime.admission.describe()
        result['model_class'] = type(lm).__module__ + '.' + type(lm).__name__
        result['native_mtp_hooks'] = {name: callable(getattr(lm, name, None)) for name in (
            'configure_mtp', 'make_mtp_cache', 'dspark_append_context',
            'mtp_validate_committed_context', 'mtp_take_committed_context',
            'mtp_install_committed_context')}
        assert not any(result['native_mtp_hooks'].values())
        ids = backend._runtime.tokenizer.encode('Explain why a computer uses a cache. ' * 1500)[:4096]
        assert len(ids) == 4096
        pre = DwarfStarMLXPrefillSession(model, omlx_path=cfg.omlx_path).prefill(ids[:-1])
        live = pre.live_result.live_cache
        generation = handoff_to_generation(pre.live_result, model, terminal_prompt_token=ids[-1],
            config=OMLXDecodeConfig(preserve_mtp=False, omlx_path=cfg.omlx_path), max_tokens=32)
        tokens = [r.token for r in generation.generate(32)]
        cache, history = generation.extract_final_state()
        assert cache is live and history == ids + tokens
        assert all(c.size() == len(history) for c in cache)
        before = digest(cache)
        reject('owned_generation_mtp_configuration', lambda: TargetGenerationSession(
            model, cache, np.asarray(history), config=OMLXDecodeConfig(preserve_mtp=True, speculation_enabled=True)))
        tx = generation.target_forward
        reject('owned_four_row_verification', lambda: tx.validate(mx.array([[1, 2, 3, 4]]), cache, len(history)))
        cache[0]._mtp_verify_state = {}
        try:
            reject('owned_verification_state', lambda: tx.validate(mx.array([1]), cache, len(history)))
        finally:
            del cache[0]._mtp_verify_state
        assert before == digest(cache)
        assert all(c.size() == len(history) for c in cache)
        result['off_control'] = dict(context=len(ids), generated=tokens, frontier=len(history),
            same_live_list=True, all_40_slots_unchanged_by_rejected_probes=True,
            prompt_replay=generation.prompt_replay_count,
            full_cache_repack=pre.live_result._setup.block_runner.full_cache_repack_count,
            handoff_count=pre.live_result.handoff_count, timing=generation.timing_summary(), slots=before)
        assert generation.prompt_replay_count == pre.live_result._setup.block_runner.full_cache_repack_count == 0
        generation.close()
        generation = None
        resources = backend._runtime.admission
        backend.close()
        reject('retired_target_resource', resources.assert_active)
        result.update(status='BLOCKED', decision='NO_PROMOTION',
            explanation='Current guarded MTP cannot enter the canonical owned target execution contract. No guard bypass or fallback attempted.')
    except BaseException as exc:
        result.update(status='PROBE_FAILED', error=repr(exc))
        raise
    finally:
        if generation is not None:
            generation.close()
        backend.close()
        args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
