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

from ds41f_mlx.config import load_runtime_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--contexts', type=int, nargs='+', default=[4096, 32768])
    p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--engine', choices=['both', 'owned'], default='both')
    args = p.parse_args()
    cfg = load_runtime_config(); cfg.apply_import_paths()
    import mlx.core as mx
    import omlx.scheduler
    from omlx.patches.deepseek_v41.loading import load
    from ds41f_mlx.runtime.dwarfstar_prefill import DwarfStarMLXPrefillSession
    from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
    from ds41f_mlx.runtime.target_generation import TargetGenerationSession
    from ds41f_mlx.runtime.target_forward import TargetForwardTransaction

    from tools.m47_m46_control import TargetForwardTransaction as M45Control
    class M45Session(TargetGenerationSession):
        # Only the subordinate boundary differs; generation remains M44/M45.
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.target_forward = M45Control(self.language_model, self.mx)
    from ds41f_mlx.prefill_fp8_mlx import handoff_to_generation, DeferredPrefillAppend
    from ds41f_mlx.prefill_fp8_mlx.p6_append import P6AppendError
    result = dict(schema='ds41f.m47.resource-admission-qualification.v1', status='RUNNING',
                  checkpoint=str(cfg.checkpoint_path), engine_scope=args.engine,
                  checkpoint_config_sha256=hashlib.sha256((cfg.checkpoint_path/'config.json').read_bytes()).hexdigest(),
                  execution_source_sha256=hashlib.sha256(Path(sys.modules[TargetGenerationSession.__module__].__file__).read_bytes()).hexdigest(),
                  forward_source_sha256=hashlib.sha256(Path(sys.modules[TargetForwardTransaction.__module__].__file__).read_bytes()).hexdigest(),
                  qualification_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  state_producer_sha256=hashlib.sha256(Path('ds41f_mlx/runtime/state_production.py').read_bytes()).hexdigest(),
                  m46_control_sha256=hashlib.sha256(Path('tools/m47_m46_control.py').read_bytes()).hexdigest(),
                  admission_source_sha256=hashlib.sha256(Path('ds41f_mlx/runtime/resource_admission.py').read_bytes()).hexdigest(),
                  resource_pin_sha256=hashlib.sha256(Path('ds41f_mlx/runtime/admitted_resources.json').read_bytes()).hexdigest(),
                  runs=[], comparisons=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save(): args.output.write_text(json.dumps(result, indent=2)+'\n')
    model = None
    try:
        t0 = perf_counter()
        from ds41f_mlx.runtime.omlx_core import OmlxRuntime, OmlxRuntimeConfig
        runtime = OmlxRuntime(OmlxRuntimeConfig(checkpoint_path=cfg.checkpoint_path,
            omlx_path=cfg.omlx_path, recipe_path=cfg.recipe_path, preserve_mtp=False))
        model, processor = runtime.load_model()
        result['admission'] = runtime.admission.describe()
        result['load_s'] = perf_counter()-t0
        result['admission_s'] = runtime.admission_seconds
        result['binding_s'] = runtime.binding_seconds
        from ds41f_mlx.runtime import resource_admission
        fingerprint = resource_admission.fingerprint
        def forbidden_hash(*a):
            raise AssertionError('resource file hashing after startup')
        resource_admission.fingerprint = forbidden_hash
        import statistics
        setup_checks = []
        for _ in range(32):
            start = perf_counter(); runtime.admission.validate_binding(model.language_model)
            setup_checks.append(perf_counter()-start)
        start = perf_counter()
        for _ in range(100000): runtime.admission.assert_active()
        result['overhead'] = dict(session_setup_median_s=statistics.median(setup_checks),
            token_guard_s=(perf_counter()-start)/100000, no_resource_file_hashing_after_startup=True)
        result['negative_admission'] = []
        decode_cfg = OMLXDecodeConfig(omlx_path=cfg.omlx_path, preserve_mtp=False)
        forward_calls = []
        from omlx.patches.deepseek_v41.cache import DeepseekV41Cache
        original_extract = DeepseekV41Cache.__dict__['extract']
        original_merge = DeepseekV41Cache.__dict__['merge']
        def forbidden_cache_operation(*a, **kw):
            raise AssertionError('owned target attempted row extraction/merge')
        lm = model.language_model
        original_forward = type(lm)._forward
        def traced_forward(obj, ids, cache=None, *a, **kw):
            if obj is lm:
                assert label != 'owned', 'external target forward on owned path'
                forward_calls.append(dict(shape=list(ids.shape), cache_list=id(cache)))
            return original_forward(obj, ids, cache, *a, **kw)
        type(lm)._forward = traced_forward
        owned_forward = TargetForwardTransaction.forward
        def traced_owned(obj, token, cache, frontier):
            forward_calls.append(dict(shape=[1, 1], cache_list=id(cache)))
            assert all(c.size() == frontier for c in cache)
            result = owned_forward(obj, token, cache, frontier)
            assert all(c.size() == frontier+1 for c in cache)
            return result
        TargetForwardTransaction.forward = traced_owned
        control_forward = M45Control.forward
        def traced_control(obj, token, cache, frontier):
            forward_calls.append(dict(shape=[1, 1], cache_list=id(cache)))
            assert all(c.size() == frontier for c in cache)
            logits = control_forward(obj, token, cache, frontier)
            assert all(c.size() == frontier+1 for c in cache)
            return logits
        M45Control.forward = traced_control
        from omlx.patches.deepseek_v41.language import Block, Attention, Compressor, Indexer
        state_calls = {cls: cls.__call__ for cls in (Block, Attention, Compressor, Indexer)}
        for cls, call in state_calls.items():
            def guarded(obj, *a, _call=call, **kw):
                assert not in_decode or label != 'owned', 'external state producer on owned path'
                return _call(obj, *a, **kw)
            cls.__call__ = guarded
        in_decode = False
        for n in args.contexts:
            ids = processor.tokenizer.encode('Explain the purpose of a computer cache. ' * (n//4+100))[:n]
            assert len(ids) == n
            reference = None
            engines = [('m46-control', M45Session), ('owned', TargetGenerationSession)]
            if args.engine == 'owned': engines = engines[1:]
            for label, factory in engines:
                mx.reset_peak_memory()
                pre = DwarfStarMLXPrefillSession(model, omlx_path=cfg.omlx_path).prefill(ids[:-1])
                live = pre.live_result.live_cache
                if label == 'owned' and not result['negative_admission']:
                    # Fault real loaded resources while holding the sole committed
                    # P7 list. No alternate cache and no numerical execution.
                    from ds41f_mlx.runtime.resource_admission import ResourceAdmissionError
                    before = (tuple(id(x) for x in live), tuple(c.size() for c in live),
                              tuple(id(v) for c in live for v in c.cache))
                    embed = lm.layers[lm._config.engram_layer_ids[0]].engram.embed
                    other = lm.layers[lm._config.engram_layer_ids[1]].engram.embed
                    faults = [('config', lm._config, 'norm_eps', 0.5),
                              ('numerical_module', lm.layers[0].attn, 'kv_norm', lm.layers[1].attn.kv_norm),
                              ('engram_key', embed, '_weight_key', 'wrong'),
                              ('engram_descriptor', embed, '_weights', other._weights)]
                    for name, owner, attribute, bad in faults:
                        original = getattr(owner, attribute)
                        try:
                            setattr(owner, attribute, bad)
                            try: TargetForwardTransaction(lm, mx)
                            except ResourceAdmissionError: pass
                            else: raise AssertionError('bad resource admitted: '+name)
                        finally: setattr(owner, attribute, original)
                        after = (tuple(id(x) for x in live), tuple(c.size() for c in live),
                                 tuple(id(v) for c in live for v in c.cache))
                        assert before == after
                        assert not any(getattr(c, '_p6_append_pending', False) for c in live)
                        runtime.admission.validate_binding(lm)
                        result['negative_admission'].append(dict(resource=name, rejected=True,
                            before_authoritative_mutation=True))
                forward_calls.clear()
                if label == 'owned':
                    DeepseekV41Cache.extract = forbidden_cache_operation
                    DeepseekV41Cache.merge = classmethod(forbidden_cache_operation)
                in_decode = True
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
                        assert str(exc) in (
                            'prior P6 commit does not own the same live cache',
                            'prior P6 commit has been reserved/transferred for P5')
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
                        owned_no_external_forward_or_extract_merge=label == 'owned',
                        all_40_frontiers_checked_each_forward=label == 'owned',
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
                            owned_m46_median_ratio=row['timing']['median_tok_s']/old['timing']['median_tok_s']))
                    save()
                finally:
                    session.close()
                    in_decode = False
                    DeepseekV41Cache.extract = original_extract
                    DeepseekV41Cache.merge = original_merge
            reference = None
        # After all comparisons, prove an unadmitted cache type cannot reach
        # owned numerical execution and retirement revokes existing executors.
        from ds41f_mlx.runtime.resource_admission import ResourceAdmissionError
        frontier = len(history)
        guard = TargetForwardTransaction(lm, mx)
        class UnadmittedCache(DeepseekV41Cache): pass
        original_class = type(cache[0]); cache[0].__class__ = UnadmittedCache
        before = tuple(c.size() for c in cache)
        try:
            guard.execute(mx.array([ids[-1]], mx.int64), cache, frontier,
                          lambda p: mx.argmax(p, axis=-1), mx.new_thread_local_stream(mx.default_device()))
        except ValueError as exc:
            assert 'unadmitted packed-cache' in str(exc)
        else: raise AssertionError('unadmitted cache mutated')
        assert tuple(c.size() for c in cache) == before and all(c._p6_append_invalid for c in cache)
        cache[0].__class__ = original_class
        runtime.admission.retire()
        try: TargetForwardTransaction(lm, mx)
        except ResourceAdmissionError: pass
        else: raise AssertionError('retired resources remained admissible')
        result['negative_admission'].extend([
            dict(resource='packed_cache_type', rejected=True, before_authoritative_mutation=True, whole_lease_burn=True),
            dict(resource='retired_model_resources', rejected=True, before_authoritative_mutation=True)])
        result['status'] = 'PASS'
    except BaseException as exc:
        result.update(status='FAIL', error=repr(exc)); raise
    finally:
        save()
        if model is not None:
            if 'original_forward' in locals(): type(lm)._forward = original_forward
            if 'owned_forward' in locals(): TargetForwardTransaction.forward = owned_forward
            if 'control_forward' in locals(): M45Control.forward = control_forward
            if 'state_calls' in locals():
                for cls, call in state_calls.items(): cls.__call__ = call
            if 'fingerprint' in locals(): resource_admission.fingerprint = fingerprint
            runtime.close()


if __name__ == '__main__': main()
