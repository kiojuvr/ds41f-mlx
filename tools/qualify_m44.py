"""Matched real-checkpoint execution/state ownership qualification.

Independent fresh P7 prefills, never cloned/replayed/repacked to admit decode.
The legacy engine is an explicit comparison oracle, not a production selector.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

from ds41f_mlx.config import RuntimeConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--contexts', type=int, nargs='+', default=[4096, 32768])
    p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--engine', choices=['both', 'owned'], default='both')
    args = p.parse_args()
    cfg = RuntimeConfig(); cfg.apply_import_paths()
    import mlx.core as mx
    import omlx.scheduler
    from omlx.patches.deepseek_v41.loading import load
    from ds41f_mlx.runtime.dwarfstar_prefill import DwarfStarMLXPrefillSession
    from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
    from ds41f_mlx.runtime.omlx_generation import OMLXGenerationSession
    from ds41f_mlx.runtime.target_generation import TargetGenerationSession
    from ds41f_mlx.prefill_fp8_mlx import handoff_to_generation, DeferredPrefillAppend
    from ds41f_mlx.prefill_fp8_mlx.p6_append import P6AppendError
    result = dict(schema='ds41f.m44.execution-qualification.v1', status='RUNNING',
                  checkpoint=str(cfg.checkpoint_path), engine_scope=args.engine,
                  checkpoint_config_sha256=hashlib.sha256((cfg.checkpoint_path/'config.json').read_bytes()).hexdigest(),
                  execution_source_sha256=hashlib.sha256(Path(sys.modules[TargetGenerationSession.__module__].__file__).read_bytes()).hexdigest(),
                  runs=[], comparisons=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save(): args.output.write_text(json.dumps(result, indent=2)+'\n')
    model = None
    try:
        t0 = perf_counter()
        model, processor = load(cfg.checkpoint_path, preserve_mtp=False, engram_ssd_offload=True)
        result['load_s'] = perf_counter()-t0
        decode_cfg = OMLXDecodeConfig(omlx_path=cfg.omlx_path, preserve_mtp=False)
        forward_calls = []
        lm = model.language_model
        original_forward = type(lm)._forward
        def traced_forward(obj, ids, cache=None, *a, **kw):
            if obj is lm:
                forward_calls.append(dict(shape=list(ids.shape), cache_list=id(cache)))
            return original_forward(obj, ids, cache, *a, **kw)
        type(lm)._forward = traced_forward
        for n in args.contexts:
            ids = processor.tokenizer.encode('Explain the purpose of a computer cache. ' * (n//4+100))[:n]
            assert len(ids) == n
            reference = None
            engines = [('legacy', OMLXGenerationSession), ('owned', TargetGenerationSession)]
            if args.engine == 'owned': engines = engines[1:]
            for label, factory in engines:
                mx.reset_peak_memory()
                pre = DwarfStarMLXPrefillSession(model, omlx_path=cfg.omlx_path).prefill(ids[:-1])
                live = pre.live_result.live_cache
                forward_calls.clear()
                t0 = perf_counter()
                session = handoff_to_generation(pre.live_result, model, terminal_prompt_token=ids[-1],
                    config=decode_cfg, max_tokens=args.tokens, session_factory=factory)
                bootstrap = perf_counter()-t0
                try:
                    tokens = [r.token for r in session.generate(args.tokens)]
                    cache, history = session.extract_final_state()
                    assert history == ids+tokens
                    assert session.cache_offsets(cache) == (len(history),)*40
                    assert session.prompt_replay_count == 0
                    assert len(forward_calls) == len(tokens)+1
                    assert all(c['shape'] == [1, 1] for c in forward_calls)
                    assert pre.live_result.handoff_count == 1
                    assert pre.live_result._setup.block_runner.full_cache_repack_count == 0
                    commit = pre.live_result._setup.p6_commit_authority
                    try:
                        DeferredPrefillAppend.continue_from_commit(lm, commit, history+[ids[-1]], mx=mx)
                    except P6AppendError as exc:
                        assert 'reserved/transferred' in str(exc)
                    else:
                        raise AssertionError('stale prefill certificate remained executable')
                    if label == 'owned':
                        assert cache is live
                        assert not hasattr(session, '_bg')
                        assert all(c['cache_list'] == id(live) for c in forward_calls)
                    row = dict(engine=label, context=n, tokens=tokens, bootstrap_s=bootstrap,
                        prefill_s=pre.seconds, timing=session.timing_summary(),
                        metadata=session.metadata().to_json(), final_frontier=len(history),
                        same_list_through_idle=cache is live, target_forward_count=len(forward_calls),
                        all_target_inputs_single_token=True, stale_prefill_certificate_rejected=True,
                        peak_memory_bytes=mx.get_peak_memory())
                    result['runs'].append(row)
                    if reference is None:
                        # Passive final arrays only; never used for execution.
                        reference = (tokens, cache, row)
                    else:
                        expected, old_cache, old = reference
                        assert tokens == expected
                        mismatches = []
                        for layer, (a, b) in enumerate(zip(old_cache, cache)):
                            for slot in range(7):
                                x, y = a[slot], b[slot]
                                if x is None or y is None:
                                    assert x is y
                                    continue
                                assert x.shape == y.shape and x.dtype == y.dtype
                                if not bool(mx.all(x == y).item()):
                                    mismatches.append([layer, slot])
                        assert not mismatches, mismatches
                        result['comparisons'].append(dict(context=n, token_identity=True,
                            exact_all_40_by_7_cache_slots=True,
                            owned_legacy_median_ratio=row['timing']['median_tok_s']/old['timing']['median_tok_s']))
                    save()
                finally:
                    session.close()
            reference = None
        result['status'] = 'PASS'
    except BaseException as exc:
        result.update(status='FAIL', error=repr(exc)); raise
    finally:
        save()
        if model is not None:
            if 'original_forward' in locals(): type(lm)._forward = original_forward
            model.close()


if __name__ == '__main__': main()
