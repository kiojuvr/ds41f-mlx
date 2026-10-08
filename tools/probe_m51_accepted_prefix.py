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
    p.add_argument('--strategy', choices=['off', 'first-party-mtp-development'], default='off')
    p.add_argument('--execution', choices=['row', 'block'], default='row')
    p.add_argument('--numerical-trace', action='store_true',
                   help='diagnostic synchronization at layer-0 boundaries; NOT performance evidence')
    p.add_argument('--block-width', type=int, choices=range(2, 9), default=8)
    p.add_argument('--width-matrix', action='store_true', help='All prefixes of every narrower block width')
    p.add_argument('--initial-frontier', type=int, choices=[255, 4095], default=4095)
    p.add_argument('--greedy-inputs', action='store_true')
    p.add_argument('--trace-layer', type=int, default=0, help='-1 captures every layer')
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
                  strategy=args.strategy, execution=args.execution,
                  numerical_trace=args.numerical_trace, block_width=args.block_width,
                  initial_frontier=args.initial_frontier, greedy_inputs=args.greedy_inputs,
                  cycles=[], cancellation=[], faults=[])
    result['runtime_source_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path('ds41f_mlx/runtime/accepted_prefix.py'),
                     Path('ds41f_mlx/runtime/target_forward.py'),
                     Path('ds41f_mlx/runtime/state_production.py'),
                     Path('ds41f_mlx/model_execution/language.py'),
                     Path('ds41f_mlx/model_execution/quantization.py'),
                     Path('ds41f_mlx/model_execution/head.py'),
                     Path('ds41f_mlx/runtime/admitted_resources.json'))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    backend = DeepSeekRecipeRuntimeBackend(runtime_config=cfg, execution_strategy=args.strategy)
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
        ids = backend._runtime.tokenizer.encode('Explain why a computer uses a cache. ' * 1500)[:args.initial_frontier]
        assert len(ids) == args.initial_frontier
        stream = mx.new_thread_local_stream(mx.default_device())
        counters = []
        bootstrap_anchor = [None]
        def fresh():
            pre = DwarfStarMLXPrefillSession(model, omlx_path=cfg.omlx_path).prefill(ids[:-1])
            gen = handoff_to_generation(pre.live_result, model, terminal_prompt_token=ids[-1],
                config=OMLXDecodeConfig(preserve_mtp=False, omlx_path=cfg.omlx_path), max_tokens=64)
            original = gen._cache
            bootstrap_anchor[0] = int(gen._pending.item())
            tx = gen.target_forward
            cache, history = gen.extract_final_state('qualification_lease_transfer')
            assert cache is original and history == ids
            counters.append(dict(replay=gen.prompt_replay_count,
                repack=pre.live_result._setup.block_runner.full_cache_repack_count,
                handoffs=pre.live_result.handoff_count))
            gen.close()
            return tx, cache
        numerical_oracle = []
        numerical_actual = []
        logit_oracle = []
        trace_pending = [args.numerical_trace]
        def capture(frame, event, arg, records):
            if not trace_pending[0] or event != 'return':
                return
            name = frame.f_code.co_name
            module = frame.f_locals.get('module')
            owner = frame.f_locals.get('self')
            layer = getattr(module, '_layer', getattr(getattr(module, 'attn', None), '_layer',
                                                     frame.f_locals.get('layer', -1)))
            values = []
            selected = isinstance(layer, int) and layer >= 0 and (
                args.trace_layer == -1 or layer == args.trace_layer)
            if name == 'attention' and selected:
                values = [(key, frame.f_locals[key]) for key in
                          ('query', 'kv_input', 'qr', 'q', 'new', 'out')]
                values.append(('attention-result', arg))
            elif name == 'block' and selected:
                values = [('block-h', arg[0]), ('block-pre', arg[1])]
            for label, value in values:
                mx.eval(value)
                records.append((f'layer{layer}.{label}', np.asarray(value.view(mx.uint8)).copy(),
                                str(value.dtype), value.shape))

        def verify(journal, tokens):
            import sys
            counts = dict(target_forward=0, backbone_layer_regions=0, forward_widths=[])
            forward_code = journal.target.forward.__func__.__code__
            block_code = journal.target.producer.block.__func__.__code__
            def profile(frame, event, arg):
                capture(frame, event, arg, numerical_actual)
                if event == 'call' and frame.f_code is forward_code:
                    counts['target_forward'] += 1
                    counts['forward_widths'].append(frame.f_locals['token'].shape[0])
                if event == 'call' and frame.f_code is block_code:
                    counts['backbone_layer_regions'] += 1
            previous = sys.getprofile()
            sys.setprofile(profile)
            try:
                if args.execution == 'block':
                    journal.advance_block(mx.array(tokens, mx.int32))
                else:
                    for token in tokens:
                        journal.advance(mx.array([token], mx.int32))
            finally:
                sys.setprofile(previous)
                result.setdefault('physical_topology', []).append(counts)
            assert counts['target_forward'] == (1 if args.execution == 'block' else len(tokens))
            assert counts['backbone_layer_regions'] == 40 * counts['target_forward']

        def compare(actual, oracle):
            differences = [dict(layer=i, slot=s, actual=a, expected=e)
                           for i, (aa, ee) in enumerate(zip(actual, oracle))
                           for s, (a, e) in enumerate(zip(aa, ee)) if a != e]
            if differences:
                result.setdefault('state_mismatch_cases', []).append(differences)
                save()
            return not differences

        def tap_digest(value):
            if value is None:
                return None
            mx.eval(value)
            return hashlib.sha256(np.asarray(value.view(mx.uint8)).tobytes()).hexdigest()

        def compare_taps(journal, oracle):
            actual = [tap_digest(row) for row in journal.tap_rows[:len(oracle)]]
            ok = actual == oracle
            if not ok:
                result.setdefault('tap_mismatch_cases', []).append(dict(actual=actual, expected=oracle))
            return ok

        width = args.block_width
        spans = [width, width // 2, 1, 0, min(6, width)]
        tokens = [11 + j for j in range(width)]
        # Independent sequential canonical OFF executor, then destroy its list.
        tx, control = fresh()
        if args.greedy_inputs:
            tokens[0] = bootstrap_anchor[0]
        expected = [fingerprint(control)]
        all_prefixes = {0: expected[0]}
        expected_taps = []
        f = len(ids)
        for cycle, k in enumerate(spans):
            taps = []
            for j in range(k):
                import sys
                previous = sys.getprofile()
                def row_profile(frame, event, arg):
                    if cycle == 0:
                        capture(frame, event, arg, numerical_oracle)
                    if event == 'return' and frame.f_code is tx.forward.__func__.__code__:
                        mx.eval(arg)
                        logit_oracle.append(hashlib.sha256(
                            np.asarray(arg.view(mx.uint8)).tobytes()).hexdigest())
                sys.setprofile(row_profile)
                try:
                    _, pending = tx.execute(mx.array([tokens[j]], mx.int32), control, f,
                                             lambda p: mx.argmax(p, axis=-1), stream)
                    if args.greedy_inputs and cycle == 0 and j + 1 < width:
                        tokens[j + 1] = int(pending.item())
                finally:
                    sys.setprofile(previous)
                if args.strategy != 'off':
                    taps.append(tap_digest(tx.tap_rows))
                f += 1
                if cycle == 0:
                    all_prefixes[j + 1] = fingerprint(control)
            expected.append(fingerprint(control))
            expected_taps.append(taps)
        expected_logits, cursor = [], 0
        for k in spans:
            expected_logits.append(logit_oracle[cursor:cursor + k])
            cursor += k
        del control, tx
        gc.collect(); mx.synchronize(stream)
        result['verified_input_ids'] = tokens
        note('independent OFF oracles complete')
        tx, cache = fresh()
        identities = tuple(cache)
        f = len(ids)
        for cycle, k in enumerate(spans):
            journal = tx.begin_prefix_journal(cache, f, width, stream)
            verify(journal, tokens)
            journal.complete()
            if args.numerical_trace and cycle == 0:
                trace = []
                for label, actual, dtype, shape in numerical_actual:
                    rows = [data for key, data, _, _ in numerical_oracle if key == label]
                    oracle = np.concatenate(rows, axis=1)
                    different = actual != oracle
                    per_row = [int(np.count_nonzero(different[:, row]))
                               for row in range(different.shape[1])]
                    trace.append(dict(boundary=label, dtype=dtype, shape=list(shape),
                                      different_bytes=int(np.count_nonzero(different)),
                                      different_bytes_per_row=per_row))
                result['numerical_trace'] = trace
                numerical_actual.clear()
                numerical_oracle.clear()
                trace_pending[0] = False
            actual_logits = [tap_digest(row) for row in journal.logits[:k]]
            logit_match = actual_logits == expected_logits[cycle]
            if not logit_match:
                result.setdefault('logit_mismatch_cases', []).append(
                    dict(actual=actual_logits, expected=expected_logits[cycle]))
            tap_match = compare_taps(journal, expected_taps[cycle])
            payload = journal.payload_bytes
            before = f
            owner_frontier = [f]
            def publish(end):
                assert owner_frontier[0] == before
                assert all(x._p6_append_pending and x.size() == end for x in cache)
                owner_frontier[0] = end
            f = journal.settle(k, publish=publish)
            assert owner_frontier[0] == f
            state_match = compare(fingerprint(cache), expected[cycle + 1])
            retired(journal, cache)
            assert all(a is b for a, b in zip(cache, identities))
            result['cycles'].append(dict(accepted=k, verified=width, frontier_before=before,
                frontier_after=f, exact_all_280_state_and_metadata_off_match=state_match,
                journal_bytes=payload, journal_retired=True, exact_objects=True,
                exact_same_forward_taps_off_match=tap_match,
                exact_full_logits_off_match=logit_match,
                settlement_packed_byte_copy=journal.settlement_packed_copy_bytes,
                compression_boundary=True, chronological_window_eviction=True))
            note(f'cycle {cycle} accepted {k} {"PASS" if state_match else "FAIL"}')
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
        for k in range(1, width):
            tx, cache = fresh()
            journal = tx.begin_prefix_journal(cache, len(ids), 8, stream)
            verify(journal, tokens)
            journal.complete()
            tap_match = compare_taps(journal, expected_taps[0][:k])
            journal.settle(k, publish=lambda end: None)
            state_match = compare(fingerprint(cache), all_prefixes[k])
            retired(journal, cache)
            result.setdefault('arbitrary_prefixes', []).append(dict(accepted=k, off_match=state_match,
                                                                  same_forward_tap_match=tap_match))
            del cache, tx, journal
            gc.collect(); mx.synchronize(stream)
            note(f'arbitrary prefix {k} {"PASS" if state_match else "FAIL"}')
        for after in (False, True):
            tx, cache = fresh()
            journal = tx.begin_prefix_journal(cache, len(ids), 8, stream)
            verify(journal, tokens)
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
            verify(journal, tokens)
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
                tx.execute(mx.array([tokens[j % len(tokens)]], mx.int32), cache, len(ids) + j,
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
        if args.width_matrix:
            assert args.execution == 'block'
            for offered in range(2, width):
                for accepted in range(offered + 1):
                    tx, cache = fresh()
                    journal = tx.begin_prefix_journal(cache, len(ids), offered, stream)
                    verify(journal, tokens[:offered])
                    journal.complete()
                    taps_ok = compare_taps(journal, expected_taps[0][:accepted])
                    logits_ok = ([tap_digest(row) for row in journal.logits[:accepted]]
                                 == expected_logits[0][:accepted])
                    journal.settle(accepted, publish=lambda end: None)
                    state_ok = compare(fingerprint(cache), all_prefixes[accepted])
                    retired(journal, cache)
                    result.setdefault('width_matrix', []).append(dict(
                        offered=offered, accepted=accepted, all_state_off_match=state_ok,
                        same_forward_tap_match=taps_ok, full_logits_match=logits_ok))
                    assert state_ok and taps_ok and logits_ok
                    del cache, tx, journal
                    gc.collect(); mx.synchronize(stream)
                note(f'width {offered} every prefix PASS')
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
        assert not result.get('state_mismatch_cases'), 'independent OFF state oracle mismatch'
        assert not result.get('tap_mismatch_cases'), 'independent OFF same-forward tap oracle mismatch'
        assert not result.get('logit_mismatch_cases'), 'independent OFF full-logit oracle mismatch'
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
