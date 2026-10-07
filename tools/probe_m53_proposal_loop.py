"""Opt-in real checkpoint DSpark -> M52 -> M51 qualification; no native scheduler."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import traceback


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--matrix', action='store_true')
    args = p.parse_args()
    from ds41f_mlx.config import load_runtime_config
    cfg = load_runtime_config()
    cfg.apply_import_paths()
    import mlx.core as mx
    from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend
    from ds41f_mlx.runtime.dwarfstar_prefill import DwarfStarMLXPrefillSession
    from ds41f_mlx.prefill_fp8_mlx import handoff_to_generation
    from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
    from ds41f_mlx.runtime.dspark_proposal import DSparkProposalProducer
    backend = DeepSeekRecipeRuntimeBackend(runtime_config=cfg)
    out = dict(decision='NOT PASS', phase='load', cycles=[])
    start = time.perf_counter()
    gen = child = None
    old_profile = sys.getprofile()
    target_calls = [0]
    proposal_times, verify_times, transport_times = [], [], []
    def note(phase):
        out['phase'] = phase
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2) + '\n')
        print(f'{time.perf_counter()-start:.2f}s {phase}', flush=True)
    def guard(frame, event, arg):
        if event == 'call':
            name = frame.f_code.co_filename.replace('\\', '/')
            if name.endswith('/ds41f_mlx/model_execution/language.py') and frame.f_code.co_name == '_forward':
                raise RuntimeError('diagnostic target reexecution forbidden')
            if name.endswith('/ds41f_mlx/runtime/target_forward.py') and frame.f_code.co_name == 'forward':
                target_calls[0] += 1
            if '/omlx/patches/mlx_lm_mtp/' in name or '/omlx/patches/deepseek_v41/' in name and name.endswith(('mtp.py', 'dspark.py', 'language.py')):
                raise RuntimeError('donor execution forbidden: ' + name)
    try:
        note('load')
        backend.load()
        model = backend._model
        admission = backend._runtime.admission
        note('admit child')
        child = admission.admit_proposal_child(model.language_model)
        out['identity'] = dict(parent=child.identity, source_keys=len(child.source_keys), shards=child.shards)
        note('canonical prefill')
        sys.setprofile(guard)
        ids = backend._runtime.tokenizer.encode('Explain why a computer uses a cache. ' * 100)[:255]
        pre = DwarfStarMLXPrefillSession(model, omlx_path=cfg.omlx_path).prefill(ids[:-1])
        seed = pre.live_result.dspark_committed_context
        out['seed'] = dict(start=seed.start, end=seed.end, shape=list(seed.rows.shape))
        gen = handoff_to_generation(pre.live_result, model, terminal_prompt_token=ids[-1],
            config=OMLXDecodeConfig(preserve_mtp=False, omlx_path=cfg.omlx_path), max_tokens=64)
        note('seed rings')
        producer = DSparkProposalProducer(gen, gen._prefill_tap_receipt)
        out['seed_retired'] = seed.retired
        assert not child.receipts
        note('first real proposal')
        t0 = time.perf_counter()
        while not gen._stopped:
            phase = time.perf_counter()
            drafts = producer.propose()
            proposal_times.append(time.perf_counter() - phase)
            phase = time.perf_counter()
            r = gen.speculative_cycle(drafts)
            verify_times.append(time.perf_counter() - phase)
            phase = time.perf_counter()
            if gen._stopped:
                producer.retire()
            else:
                receipt = gen.take_tap_receipt()
                producer.advance(receipt)
                assert receipt.retired
                assert producer.frontier == gen.token_frontier
            transport_times.append(time.perf_counter() - phase)
            assert set(gen.active_cache_offsets()) == {len(gen.current_token_history())}
            out['cycles'].append(dict(proposals=drafts, accepted=r['proposal_acceptance_count'],
                                      consumed=r['consumed_positions'], frontier=gen.token_frontier,
                                      ring_offsets=[None if ring.keys is None else ring.offset for ring in producer.rings]))
        elapsed = time.perf_counter() - t0
        out['decode'] = dict(tokens=len(gen.generated_tokens), seconds=elapsed,
                             tok_s=len(gen.generated_tokens)/elapsed)
        out['acceptance'] = dict(proposed=sum(len(r['proposals']) for r in out['cycles']),
                                accepted=sum(r['accepted'] for r in out['cycles']))
        out['phase_seconds'] = dict(proposal_math=sum(proposal_times),
            target_verify_and_settlement=sum(verify_times), hidden_transport_and_ring=sum(transport_times))
        assert target_calls[0] == 1 + sum(len(r['proposals']) + 1 for r in out['cycles'])
        out['target_forward_calls'] = target_calls[0]
        out['hidden_target_reexecution'] = 0
        out['coherent'] = True
        out['terminal'] = dict(reason=gen.stop_reason, frontier=gen.token_frontier,
            ring_retired=not producer.active, receipts=len(child.receipts))
        out['replay'] = gen.prompt_replay_count
        out['repack'] = pre.live_result._setup.block_runner.full_cache_repack_count
        out['decision'] = 'LOOP RUN; extended qualification required'
        note('loop complete')
        if args.matrix:
            import hashlib
            import numpy as np
            def snapshot(s):
                cache = s._cache or s._final_cache
                return dict(history=s.current_token_history(), frontier=s.token_frontier,
                    pending=None if s._pending is None else int(s._pending.item()),
                    slots=[[hashlib.sha256(np.asarray(v.view(mx.uint8)).tobytes()).hexdigest()
                            if v is not None else None for v in c.cache] for c in cache])
            def new_session(sampler=None, count=32):
                result = DwarfStarMLXPrefillSession(model, omlx_path=cfg.omlx_path).prefill(ids[:-1])
                s = handoff_to_generation(result.live_result, model, terminal_prompt_token=ids[-1],
                    config=OMLXDecodeConfig(preserve_mtp=False, omlx_path=cfg.omlx_path),
                    max_tokens=count, sampler=sampler)
                d = DSparkProposalProducer(s, s._prefill_tap_receipt)
                assert s.prompt_replay_count == 0
                assert result.live_result._setup.block_runner.full_cache_repack_count == 0
                return s, d
            gen.close()
            out['matrix'] = {}
            # Paired sessions are strictly sequential, never concurrent lifetimes.
            for mode in ('greedy', 'stochastic'):
                def fresh_sampler():
                    key = [mx.random.key(53)]
                    def sampler(lp):
                        key[0], draw = mx.random.split(key[0])
                        return mx.random.categorical(lp / 0.7, key=draw).astype(mx.uint32)
                    return sampler
                gen, producer = new_session(fresh_sampler() if mode == 'stochastic' else None)
                rows = []
                while not gen._stopped:
                    r = producer.cycle()
                    rows.append((r['proposal_acceptance_count'], r['consumed_positions']))
                expected = snapshot(gen)
                gen.close()
                gen, producer = new_session(fresh_sampler() if mode == 'stochastic' else None)
                gen.disable_proposals()
                gen.generate(32)
                actual = snapshot(gen)
                assert actual == expected
                out['matrix'][mode + '_off_parity'] = dict(exact=True, cycles=rows)
                gen.close()
            # Rejection controls perturb real producer outputs, not replace
            # the producer with a scripted scheduler. Canonical sampler unchanged.
            rejection_rows = []
            for expected_accept in (0, 2):
                gen, producer = new_session(count=16)
                raw = producer.propose()
                drafts = list(raw)
                drafts[expected_accept] = (drafts[expected_accept] + 1) % child.config.vocab_size
                r = gen.speculative_cycle(drafts)
                assert r['proposal_acceptance_count'] == expected_accept
                receipt = gen.take_tap_receipt()
                assert receipt.rows.shape[1] == expected_accept + 1
                producer.advance(receipt)
                assert producer.frontier == gen.token_frontier and not child.receipts
                assert gen.current_token_history()[-(expected_accept + 1):] == [r['confirmed_anchor']] + list(raw[:expected_accept])
                expected = snapshot(gen)
                rejection_rows.append(dict(raw=raw, verified=drafts, accepted=expected_accept,
                    consumed=r['consumed_positions'], receipt_retired=receipt.retired, off_exact=True))
                gen.close()
                gen, producer = new_session(count=16)
                gen.disable_proposals()
                gen.generate(expected_accept + 1)
                assert snapshot(gen) == expected
                gen.close()
            out['matrix']['first_and_partial_reject'] = rejection_rows
            gen, producer = new_session(count=16)
            low = []
            for _ in range(5):
                raw = producer.propose()
                drafts = list(raw)
                drafts[0] = (drafts[0] + 1) % child.config.vocab_size
                r = gen.speculative_cycle(drafts)
                assert r['proposal_acceptance_count'] == 0 and r['consumed_positions'] == 1
                producer.advance(gen.take_tap_receipt())
                low.append(dict(proposals=raw, accepted=0, consumed=1, frontier=producer.frontier))
            out['matrix']['low_acceptance'] = low
            gen.close()
            gen, producer = new_session(count=16)
            before = snapshot(gen)
            try:
                producer.propose(_fault=lambda phase: (_ for _ in ()).throw(RuntimeError('proposal before')))
            except RuntimeError:
                pass
            else:
                raise AssertionError('proposal failure missing')
            assert snapshot(gen) == before and not producer.active
            assert all(r.keys is None for r in producer.rings)
            gen.disable_proposals()
            gen.next_token()
            out['matrix']['proposal_failure_before_verify'] = dict(exact=True, explicit_idle_disable=True)
            gen.close()
            gen, producer = new_session(count=16)
            before = snapshot(gen)
            armed = [False]
            r = producer.cycle(cancelled=lambda: armed[0], _fault=lambda phase:
                armed.__setitem__(0, True) if phase == 'materialized' else None)
            assert r['consumed_positions'] == 0 and snapshot(gen) == before
            assert producer.frontier == gen.token_frontier and not child.receipts
            out['matrix']['cancel'] = dict(exact=True, consumed=0)
            producer.cycle()
            # Proposal math failure before verify cannot mutate target truth.
            before = snapshot(gen)
            # Advance failure is deliberately after canonical publication.
            drafts = producer.propose()
            r = gen.speculative_cycle(drafts)
            committed = snapshot(gen)
            receipt = gen.take_tap_receipt()
            try:
                producer.advance(receipt, _fault=lambda phase: (_ for _ in ()).throw(
                    RuntimeError('proposal ring fault')))
            except Exception:
                pass
            else:
                raise AssertionError('proposal fault missing')
            assert not producer.active and receipt.retired and snapshot(gen) == committed
            gen.disable_proposals()
            gen.next_token()
            assert set(gen.active_cache_offsets()) == {len(gen.current_token_history())}
            out['matrix']['proposal_failure_after_publication'] = dict(coherent=True,
                receipt_retired=True, ring_retired=True, explicit_idle_disable=True)
            gen.close()
            # Repeated idle continuation destroys old rings; warm only through
            # legitimate new canonical forwards, never reconstructed history.
            from ds41f_mlx.runtime.target_generation import TargetGenerationSession
            continuations = []
            gen, producer = new_session(count=192)
            for epoch in range(2):
                producer.cycle()
                pending = int(gen._pending.item())
                cache, history = gen.extract_final_state()
                assert not producer.active and all(r.keys is None for r in producer.rings)
                gen.close()
                gen = TargetGenerationSession.from_prefilled_cache(model, cache, history,
                    OMLXDecodeConfig(preserve_mtp=False, omlx_path=cfg.omlx_path), max_tokens=192)
                gen.start(pending)
                producer = DSparkProposalProducer(gen)
                for _ in range(child.config.window_size - 1):
                    gen.next_token()
                    producer.advance(gen.take_tap_receipt())
                assert all(r.keys.shape[2] == child.config.window_size for r in producer.rings)
                r = producer.cycle()
                assert gen._cache is cache
                continuations.append(dict(epoch=epoch, frontier=gen.token_frontier,
                    warmup=child.config.window_size, consumed=r['consumed_positions']))
            out['matrix']['continuations'] = continuations
            gen.close()
            gen, producer = new_session()
            try:
                producer.cycle(_fault=lambda phase: (_ for _ in ()).throw(RuntimeError('target fault'))
                    if phase == 'tentative' else None)
            except RuntimeError:
                pass
            else:
                raise AssertionError('target fault missing')
            assert gen._failed and not producer.active and not child.receipts
            assert all(c._p6_append_invalid for c in gen._cache)
            out['matrix']['target_failure'] = dict(all_aliases_burned=True, derived_retired=True)
            gen.close()
            gen, producer = new_session(count=16)
            before = snapshot(gen)
            child.retire()
            assert not producer.active and not child.receipts and child.parameters is None
            assert child.model is None
            try:
                child.assert_active()
            except RuntimeError:
                pass
            else:
                raise AssertionError('retired proposal capability executable')
            assert snapshot(gen) == before
            gen.disable_proposals()
            gen.next_token()
            assert set(gen.active_cache_offsets()) == {len(gen.current_token_history())}
            out['matrix']['proposal_resource_revoke'] = dict(unchanged_target=True,
                read_handles_released=True, permission_revoked=True, explicit_idle_disable=True)
            gen.close()
            out['decision'] = 'MATRIX RUN; review all qualification gates'
            note('matrix complete')
    except BaseException as exc:
        out['error'] = f'{type(exc).__name__}: {exc}'
        out['traceback'] = traceback.format_exc()
        note('failed at ' + out['phase'])
        raise
    finally:
        sys.setprofile(old_profile)
        if gen is not None:
            gen.close()
        backend.close()
        if child is not None:
            out['retirement'] = dict(active=child.active, receipts=len(child.receipts),
                                     producers=len(child.producers), parameters_released=child.parameters is None,
                                     shared_target_handles_released=child.model is None)
        gc.collect()
        out['elapsed_s'] = time.perf_counter()-start
        args.output.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
