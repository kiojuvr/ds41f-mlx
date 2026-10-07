"""Real checkpoint journal qualification, isolated from generation scheduling.

Independent OFF oracles are destroyed before the tentative executor is created.
The qualification driver supplies consumed-input counts; no native scheduler or
sampling acceptance protocol is installed in the production generation loop.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from ds41f_mlx.config import load_runtime_config
    cfg = load_runtime_config(); cfg.apply_import_paths()
    import mlx.core as mx
    import numpy as np
    from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend
    from ds41f_mlx.runtime.dwarfstar_prefill import DwarfStarMLXPrefillSession
    from ds41f_mlx.prefill_fp8_mlx import handoff_to_generation
    from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
    from ds41f_mlx.runtime.target_generation import TargetGenerationSession

    result = dict(schema='ds41f.m51.journal-qualification.v1', status='RUNNING',
                  base=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  cycles=[], cancellation=[], faults=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    backend = DeepSeekRecipeRuntimeBackend(runtime_config=cfg)
    started = time.monotonic()
    def save():
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    def note(msg):
        print(f'{time.monotonic() - started:.1f}s {msg}', flush=True)
        save()
    def fingerprint(cache):
        mx.eval(*(v for x in cache for v in (*x.cache, x.lengths, x.left_padding) if v is not None))
        return [[None if v is None else dict(shape=list(v.shape), dtype=str(v.dtype),
                 sha256=hashlib.sha256(np.asarray(v.view(mx.uint8)).tobytes()).hexdigest())
                 for v in (*x.cache, x.lengths, x.left_padding)] for x in cache]
    def retired(journal, cache):
        assert journal.phase == 'retired' and not journal.objects
        assert not journal.initial and not journal.projections and not journal.evicted
        assert not journal.histories and not journal.logits
        assert all(not x._p6_append_pending and not hasattr(x, '_accepted_prefix_journal') for x in cache)
    try:
        note('loading admitted checkpoint')
        backend.load()
        result['admission'] = backend._runtime.admission.describe()
        model = backend._model
        c = model.language_model._config
        result['layout'] = dict(window=c.window_size, kv_sources=list(c.kv_source_layers),
                                index_sources=list(c.index_source_layers), ratios=list(c.compress_ratios))
        ids = backend._runtime.tokenizer.encode('Explain why a computer uses a cache. ' * 1500)[:4095]
        assert len(ids) == 4095
        stream = mx.new_thread_local_stream(mx.default_device())
        counters = []
        def fresh():
            pre = DwarfStarMLXPrefillSession(model, omlx_path=cfg.omlx_path).prefill(ids[:-1])
            gen = handoff_to_generation(pre.live_result, model, terminal_prompt_token=ids[-1],
                config=OMLXDecodeConfig(preserve_mtp=False, omlx_path=cfg.omlx_path), max_tokens=64)
            original = gen._cache
            tx = gen.target_forward
            cache, history = gen.extract_final_state('qualification_lease_transfer')
            assert cache is original and history == ids
            counters.append(dict(replay=gen.prompt_replay_count,
                repack=pre.live_result._setup.block_runner.full_cache_repack_count,
                handoffs=pre.live_result.handoff_count))
            gen.close()
            return tx, cache
        spans = [8, 4, 1, 0, 6]
        tokens = [11 + j for j in range(8)]
        # Independent sequential canonical OFF executor, then destroy its list.
        tx, control = fresh()
        expected = [fingerprint(control)]
        all_prefixes = {0: expected[0]}
        f = len(ids)
        for cycle, k in enumerate(spans):
            for j in range(k):
                tx.execute(mx.array([tokens[j]], mx.int32), control, f,
                           lambda p: mx.argmax(p, axis=-1), stream)
                f += 1
                if cycle == 0:
                    all_prefixes[j + 1] = fingerprint(control)
            expected.append(fingerprint(control))
        del control, tx
        gc.collect(); mx.synchronize(stream)
        note('independent OFF oracles complete')
        tx, cache = fresh()
        identities = tuple(cache)
        f = len(ids)
        for cycle, k in enumerate(spans):
            journal = tx.begin_prefix_journal(cache, f, 8, stream)
            for token in tokens:
                journal.advance(mx.array([token], mx.int32))
            journal.complete()
            payload = journal.payload_bytes
            before = f
            owner_frontier = [f]
            def publish(end):
                assert owner_frontier[0] == before
                assert all(x._p6_append_pending and x.size() == end for x in cache)
                owner_frontier[0] = end
            f = journal.settle(k, publish=publish)
            assert owner_frontier[0] == f
            assert fingerprint(cache) == expected[cycle + 1]
            retired(journal, cache)
            assert all(a is b for a, b in zip(cache, identities))
            result['cycles'].append(dict(accepted=k, verified=8, frontier_before=before,
                frontier_after=f, exact_all_280_state_and_metadata_off_match=True,
                journal_bytes=payload, journal_retired=True, exact_objects=True,
                settlement_packed_byte_copy=journal.settlement_packed_copy_bytes,
                compression_boundary=True, chronological_window_eviction=True))
            note(f'cycle {cycle} accepted {k} PASS')
        # OFF continuation + exact-list idle return after successful settlement.
        history = ids + [t for k in spans for t in tokens[:k]]
        gen = TargetGenerationSession.from_prefilled_cache(model, cache, history,
                OMLXDecodeConfig(preserve_mtp=False, omlx_path=cfg.omlx_path), max_tokens=1)
        gen.start(19)
        final, committed = gen.extract_final_state()
        assert final is cache and committed == history + [19]
        gen.close()
        result['idle_exact_list_return'] = True
        del final, gen, cache, identities, tx
        gc.collect(); mx.synchronize(stream)
        # Arbitrary prefix checks not already covered by the repeated chain.
        for k in [1, 2, 3, 4, 5, 6, 7]:
            tx, cache = fresh()
            journal = tx.begin_prefix_journal(cache, len(ids), 8, stream)
            for token in tokens:
                journal.advance(mx.array([token], mx.int32))
            journal.complete(); journal.settle(k, publish=lambda end: None)
            assert fingerprint(cache) == all_prefixes[k]
            retired(journal, cache)
            result.setdefault('arbitrary_prefixes', []).append(dict(accepted=k, off_match=True))
            del cache, tx, journal
            gc.collect(); mx.synchronize(stream)
            note(f'arbitrary prefix {k} PASS')
        for after in (False, True):
            tx, cache = fresh()
            journal = tx.begin_prefix_journal(cache, len(ids), 8, stream)
            for token in tokens:
                journal.advance(mx.array([token], mx.int32))
            if after:
                journal.complete()
            journal.cancel()
            assert fingerprint(cache) == expected[0]
            retired(journal, cache)
            result['cancellation'].append(dict(after_materialization=after, off_match=True,
                                               journal_retired=True))
            del cache, tx, journal
            gc.collect(); mx.synchronize(stream)
            note(f'cancel after materialization={after} PASS')
        for phase, layer, slot in [
            ('materialize-before', -1, -1), ('materialize-after', -1, -1),
            ('materialize-state', 19, -1), ('prepare', 19, -1),
            ('publish', 0, 0), ('publish', 0, 1), ('publish', 2, 2),
            ('publish', 2, 3), ('publish', 2, 4), ('publish', 2, 5),
            ('publish', 0, 6), ('publish', 39, 'lengths'),
            ('publish', 39, 'left_padding'), ('publication-boundary', -1, -1),
            ('owner-publication', -1, -1),
        ]:
            tx, cache = fresh()
            aliases = tuple(cache)
            def fail(p, i, s):
                assert all(x._p6_append_pending for x in aliases)
                if (p, i, s) == (phase, layer, slot):
                    raise RuntimeError('M51 injected barrier fault')
            journal = tx.begin_prefix_journal(cache, len(ids), 8, stream, fault=fail)
            for token in tokens:
                journal.advance(mx.array([token], mx.int32))
            try:
                journal.complete()
                journal.settle(3, publish=lambda end: fail('owner-publication', -1, -1))
            except RuntimeError as exc:
                assert str(exc) == 'M51 injected barrier fault'
            else:
                raise AssertionError('fault not reached')
            assert journal.phase == 'burned' and not journal.objects and not journal.initial
            assert all(x._p6_append_failed and x._p6_append_invalid for x in aliases)
            try:
                tx.validate(mx.array([1]), cache, len(ids))
            except RuntimeError:
                pass
            else:
                raise AssertionError('burned alias admitted')
            result['faults'].append(dict(phase=phase, layer=layer, slot=slot,
                                         all_aliases_burned=True, journal_retired=True))
            del cache, aliases, tx, journal
            gc.collect(); mx.synchronize(stream)
            note(f'fault {phase}/{layer}/{slot} PASS')
        # Matched M50 OFF control: frozen orchestration only, same admitted
        # first-party model/math/storage. Not a donor scheduler or live shadow.
        import types
        import statistics
        frozen_producer = types.ModuleType('m50_state_production')
        exec(compile(subprocess.check_output(['git', 'show',
            'bdd08e7:ds41f_mlx/runtime/state_production.py'], text=True),
            '<m50-state-production>', 'exec'), frozen_producer.__dict__)
        frozen_target = types.ModuleType('m50_target_forward')
        exec(compile(subprocess.check_output(['git', 'show',
            'bdd08e7:ds41f_mlx/runtime/target_forward.py'], text=True),
            '<m50-target-forward>', 'exec'), frozen_target.__dict__)
        frozen_target.DecodeStateProducer = frozen_producer.DecodeStateProducer
        perf, outputs = {}, {}
        for name in ('m50', 'm51'):
            tx, cache = fresh()
            if name == 'm50':
                tx = frozen_target.TargetForwardTransaction(model.language_model, mx)
            latencies = []
            for j in range(32):
                begin = time.perf_counter()
                tx.execute(mx.array([tokens[j % 8]], mx.int32), cache, len(ids) + j,
                           lambda p: mx.argmax(p, axis=-1), stream)
                latencies.append(time.perf_counter() - begin)
            perf[name] = dict(median_latency_s=statistics.median(latencies), latencies_s=latencies)
            outputs[name] = fingerprint(cache)
            del tx, cache
            gc.collect(); mx.synchronize(stream)
        assert outputs['m50'] == outputs['m51']
        perf['all_state_byte_match'] = True
        perf['latency_ratio_m51_over_m50'] = perf['m51']['median_latency_s'] / perf['m50']['median_latency_s']
        assert perf['latency_ratio_m51_over_m50'] < 1.5
        result['off_regression'] = perf
        note('matched M50 OFF state/performance regression PASS')
        assert all(x['replay'] == x['repack'] == 0 and x['handoffs'] == 1 for x in counters)
        result['setup_counters'] = counters
        prefetch = getattr(model.language_model, '_engram_prefetch', None)
        result['ssd_prefetch_drained'] = prefetch is None or (
            prefetch._pending is None and all(getattr(layer.engram.embed, '_prefetched', None) is None
                for layer in model.language_model.layers if 'engram' in layer))
        assert result['ssd_prefetch_drained']
        resources = backend._runtime.admission
        backend.close()
        try:
            resources.assert_active()
        except RuntimeError:
            result['model_resources_retired'] = True
        else:
            raise AssertionError('resource retirement failed')
        result.update(status='PASS', decision='YES_PRIMITIVE_ONLY',
                      native_oracle='NOT RUN; independent canonical OFF used',
                      elapsed_s=time.monotonic() - started)
        note('checkpoint qualification PASS; see separate physical prefix/storage audit')
    except BaseException as exc:
        result.update(status='FAILED', error=repr(exc))
        save()
        raise
    finally:
        backend.close()
        save()


if __name__ == '__main__':
    main()
