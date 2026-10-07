"""M50 stopped-boundary evidence on the admitted real checkpoint.

This is NOT speculative qualification. Execute only canonical OFF tokens; retain
hashes, never executable snapshots. No guard bypass, rollback, replay or repack.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    from ds41f_mlx.config import load_runtime_config
    cfg = load_runtime_config()
    cfg.apply_import_paths()
    import mlx.core as mx
    import numpy as np
    from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend
    from ds41f_mlx.runtime.dwarfstar_prefill import DwarfStarMLXPrefillSession
    from ds41f_mlx.prefill_fp8_mlx import handoff_to_generation
    from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig

    result = dict(schema='ds41f.m50.stopped-state-boundary.v1', status='RUNNING',
                  base=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  checkpoint=str(cfg.checkpoint_path), transitions=[], guards=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    backend = DeepSeekRecipeRuntimeBackend(runtime_config=cfg)
    generation = None

    def fingerprint(x):
        mx.eval(x)
        return dict(shape=list(x.shape), dtype=str(x.dtype),
                    sha256=hashlib.sha256(np.asarray(x.view(mx.uint8)).tobytes()).hexdigest())

    def snapshot(cache):
        mx.eval(*(x for item in cache for x in (*item.cache, item.lengths, item.left_padding)
                  if x is not None))
        return [dict(layer=i, frontier=item.size(), ratio=item.compress_ratio,
                     slots=[fingerprint(x) for x in item.cache],
                     lengths=None if item.lengths is None else fingerprint(item.lengths),
                     left_padding=None if item.left_padding is None else fingerprint(item.left_padding))
                for i, item in enumerate(cache)]

    try:
        backend.load()
        model = backend._model
        c = model.language_model._config
        result['admission'] = backend._runtime.admission.describe()
        result['layout'] = dict(window=c.window_size, ratios=list(c.compress_ratios),
                                kv_sources=list(c.kv_source_layers),
                                index_sources=list(c.index_source_layers))
        # Advance across 4096: actual qualified window eviction and compressor
        # group completion, not an invented physical ring implementation.
        ids = backend._runtime.tokenizer.encode('Explain why a computer uses a cache. ' * 1500)[:4095]
        assert len(ids) == 4095
        pre = DwarfStarMLXPrefillSession(model, omlx_path=cfg.omlx_path).prefill(ids[:-1])
        live = pre.live_result.live_cache
        identities = tuple(live)
        generation = handoff_to_generation(pre.live_result, model, terminal_prompt_token=ids[-1],
            config=OMLXDecodeConfig(preserve_mtp=False, omlx_path=cfg.omlx_path), max_tokens=16)
        tx = generation.target_forward
        initial = snapshot(live)
        for name, token in [('several_flat_rows', mx.array([1, 2, 3, 4])),
                            ('several_sequence_rows', mx.array([[1, 2, 3, 4]]))]:
            try:
                tx.validate(token, live, len(ids))
            except (RuntimeError, ValueError) as exc:
                result['guards'].append(dict(name=name, rejected=True, error=str(exc)))
            else:
                raise AssertionError('unsupported speculative shape admitted')
        tx.validate(mx.array([1]), live, len(ids))
        assert snapshot(live) == initial
        result['initial'] = initial
        for _ in range(8):
            before = snapshot(live)
            # Passive hashes of the retained window and compressed prefixes.
            window_suffix = [fingerprint(item[1][:, 1:]) for item in live]
            compressed = [(item[2].shape[1], item[3].shape[1]) for item in live]
            report = generation.next_token()
            assert report is not None
            after = snapshot(live)
            assert all(row['frontier'] == report.frontier_after for row in after)
            tx.producer.validate_completion(live, c, report.frontier_after)
            assert all(a is b for a, b in zip(live, identities))
            assert generation.current_token_history() == ids + generation.generated_tokens
            layers = []
            for i, item in enumerate(live):
                assert fingerprint(item[1][:, :-1]) == window_suffix[i]
                for slot, n in zip((2, 3), compressed[i]):
                    assert fingerprint(item[slot][:, :n]) == before[i]['slots'][slot]
                r = item.compress_ratio
                completed = r > 1 and report.frontier_after % r == 0
                if completed:
                    assert before[i]['slots'][4]['shape'][1] == r - 1
                    assert after[i]['slots'][4]['shape'][1] == 0
                    assert after[i]['slots'][5]['shape'][1] == 0
                layers.append(dict(layer=i, window_oldest_evicted=True,
                    compressed_prefix_unchanged=True, compressor_remainder_consumed=completed,
                    changed_slots=[j for j in range(7) if before[i]['slots'][j] != after[i]['slots'][j]]))
            result['transitions'].append(dict(report=report.to_json(), layers=layers, state=after))
        assert any(layer['compressor_remainder_consumed']
                   for step in result['transitions'] for layer in step['layers'])
        final = snapshot(live)
        generation.cancel()
        cache, history = generation.extract_final_state()
        assert cache is live and snapshot(cache) == final
        assert history == ids + generation.generated_tokens
        assert all(not getattr(item, '_p6_append_pending', False) for item in cache)
        assert generation._pending is None
        result['off_control'] = dict(single_row_transactions=8, all_40_layers_coherent=True,
            all_280_slots_observed=True, exact_list_idle_return=True, cancelled_at_idle=True,
            committed_frontier=len(history), accepted_history=history,
            prompt_replay=generation.prompt_replay_count,
            full_cache_repack=pre.live_result._setup.block_runner.full_cache_repack_count,
            handoff_count=pre.live_result.handoff_count)
        assert result['off_control']['prompt_replay'] == result['off_control']['full_cache_repack'] == 0
        generation.close()
        generation = None
        resources = backend._runtime.admission
        prefetch = getattr(model.language_model, '_engram_prefetch', None)
        result['prefetch_drained'] = prefetch is None or (
            getattr(prefetch, '_pending', None) is None and getattr(prefetch, '_prefetched', None) is None)
        assert result['prefetch_drained']
        backend.close()
        try:
            resources.assert_active()
        except RuntimeError as exc:
            result['resource_retired'] = str(exc)
        else:
            raise AssertionError('retired admission reusable')
        result.update(status='BLOCKED', decision='NO',
            missing_primitive='M46 bounded prefix undo/publication journal under parent target lease',
            speculative_qualification='NOT RUN: no admitted speculative transaction exists')
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
