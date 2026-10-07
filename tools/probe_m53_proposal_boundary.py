"""Real-checkpoint M53 blocker probe, NOT DSpark decode qualification.

No donor model/scheduler is loaded or used by this probe. The standard admitted
model is examined at the same-forward tap and model-lifetime resource seams.
Empty M52 cycles are explicitly controls, never actual DSpark proposals.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from ds41f_mlx.config import load_runtime_config
    cfg = load_runtime_config()
    cfg.apply_import_paths()
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend
    from ds41f_mlx.runtime.dwarfstar_prefill import DwarfStarMLXPrefillSession
    from ds41f_mlx.prefill_fp8_mlx import handoff_to_generation
    from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
    from ds41f_mlx.runtime.target_generation import TargetGenerationSession
    from ds41f_mlx.model_execution import loading

    out = dict(schema='ds41f.m53.proposal-boundary.v1', status='RUNNING',
               decision='NO / M53 NOT PASS', actual_dspark_proposals=0,
               qualification='boundary probe only; not end-to-end DSpark qualification')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()

    def note(text):
        args.output.write_text(json.dumps(out, indent=2) + '\n')
        print(f'{time.perf_counter()-begin:.1f}s {text}', flush=True)

    def digest(value):
        return hashlib.sha256(np.asarray(value.view(mx.uint8)).tobytes()).hexdigest()

    def snapshot(gen):
        cache = gen._cache if gen._cache is not None else gen._final_cache
        mx.eval(*[v for c in cache for v in (*c.cache, c.lengths, c.left_padding)
                  if v is not None])
        return dict(history=gen.current_token_history(), frontier=gen.token_frontier,
                    pending=None if gen._pending is None else int(gen._pending.item()),
                    slots=[[(list(v.shape), str(v.dtype), digest(v)) if v is not None
                            else None for v in (*c.cache, c.lengths, c.left_padding)]
                           for c in cache])

    def coherent(gen):
        assert set(gen.active_cache_offsets()) == {len(gen.current_token_history())}
        assert gen.token_frontier == len(gen.current_token_history())

    backend = DeepSeekRecipeRuntimeBackend(runtime_config=cfg)
    sessions = []
    old_profile = sys.getprofile()
    forbidden_calls = []

    def guard(frame, event, arg):
        # Observe real execution, not sys.modules membership/import side effects.
        # Imports have already completed before this guard is installed.
        if event == 'call':
            path = frame.f_code.co_filename.replace('\\', '/')
            if ('omlx/patches/mlx_lm_mtp/' in path
                    or path.endswith('/omlx/patches/deepseek_v41/mtp.py')):
                forbidden_calls.append((path, frame.f_code.co_name))
                raise RuntimeError('donor MTP execution forbidden by M53 probe')

    try:
        note('loading full M47-admitted official checkpoint, standard OFF')
        backend.load()
        model = backend._model
        lm = model.language_model
        admission = backend._runtime.admission
        out['resource_identity'] = admission.identity
        out['model_type'] = f'{type(lm).__module__}.{type(lm).__name__}'
        out['admitted_dspark_modules'] = [name for name in admission.modules
                                          if 'dspark' in name or name.endswith('.mtp')]
        params = tree_flatten(lm.parameters())
        out['loaded_mtp_parameter_count'] = sum(name.startswith('mtp.') for name, _ in params)
        out['loaded_mtp_stage_count'] = len(getattr(lm, 'mtp', ()))
        out['config'] = {name: getattr(lm._config, name) for name in
                         ('preserve_mtp', 'n_mtp_layers', 'dspark_block_size',
                          'dspark_target_layer_ids', 'window_size')}
        mapping = json.loads((admission._checkpoint/'model.safetensors.index.json').read_text())['weight_map']
        mtp_keys = [key for key in mapping if key.startswith('mtp.')]
        out['checkpoint_mtp_tensor_count'] = len(mtp_keys)
        out['checkpoint_mtp_stages'] = sorted({key.split('.')[1] for key in mtp_keys})
        out['checkpoint_mtp_shards'] = sorted({mapping[key] for key in mtp_keys})
        assert mtp_keys and not out['loaded_mtp_parameter_count']
        assert not out['loaded_mtp_stage_count'] and not out['admitted_dspark_modules']
        # Do NOT select OmlxRuntime(preserve_mtp=True): that would load a donor.
        try:
            loading.load(admission._checkpoint, preserve_mtp=True)
        except ValueError as exc:
            out['first_party_preserve_mtp_rejection'] = str(exc)
        else:
            raise AssertionError('unexpected first-party MTP loader admission')
        note('official MTP tensors exist, but first-party model does not bind them')

        sys.setprofile(guard)
        ids = backend._runtime.tokenizer.encode('Explain why a computer uses a cache. ' * 100)[:255]
        pre = DwarfStarMLXPrefillSession(model, omlx_path=cfg.omlx_path).prefill(ids[:-1])
        out['prefill_dspark_context_present'] = pre.live_result.dspark_committed_context is not None
        assert not out['prefill_dspark_context_present']
        gen = handoff_to_generation(pre.live_result, model, terminal_prompt_token=ids[-1],
            config=OMLXDecodeConfig(preserve_mtp=False, omlx_path=cfg.omlx_path), max_tokens=24)
        sessions.append(gen)
        cache = gen._cache
        initial = snapshot(gen)
        journal = gen.target_forward.begin_prefix_journal(cache, gen.token_frontier, 2, gen.stream)
        row = journal.advance(gen._pending)
        journal.advance(mx.array([10], mx.uint32))
        journal.complete()
        out['journal_outputs'] = dict(row_shape=list(row.shape), row_dtype=str(row.dtype),
                                      logits_count=len(journal.logits),
                                      hidden_attribute_present=hasattr(journal, 'hidden'),
                                      dspark_attribute_present=hasattr(journal, 'dspark_hidden'))
        # Cancel rather than grant this observer target/history authority.
        journal.cancel()
        assert snapshot(gen) == initial
        assert all(not hasattr(c, '_accepted_prefix_journal') for c in cache)
        out['journal_cancel_exact_state'] = True
        out['journal_retired'] = journal.phase
        armed = [False]
        receipt = gen.speculative_cycle([], cancelled=lambda: armed[0],
            _fault=lambda phase: armed.__setitem__(0, True) if phase == 'materialized' else None)
        assert receipt['cancelled'] and receipt['consumed_positions'] == 0
        assert snapshot(gen) == initial
        out['empty_cycle_presampling_cancellation'] = dict(exact_state=True, consumed=0)
        receipt = gen.speculative_cycle([])
        assert receipt['consumed_positions'] == 1 and receipt['proposal_acceptance_count'] == 0
        coherent(gen)
        out['empty_cycle_control'] = dict(consumed=1, proposals=0, accepted=0)
        gen.generate(3)
        coherent(gen)
        # Repeated continuation exercises only the established M52 boundary.
        out['continuations'] = []
        for _ in range(2):
            pending = int(gen._pending.item())
            returned, history = gen.extract_final_state()
            assert returned is cache
            nxt = TargetGenerationSession.from_prefilled_cache(model, returned, history,
                OMLXDecodeConfig(preserve_mtp=False, omlx_path=cfg.omlx_path), max_tokens=16)
            sessions.append(nxt)
            nxt.start(pending)
            coherent(nxt)
            out['continuations'].append(dict(exact_list=True, frontier=nxt.token_frontier))
            gen.close()
            gen = nxt
            gen.generate(2 if len(out['continuations']) == 1 else 0)
        mx.synchronize(gen.stream)
        t0 = time.perf_counter()
        reports = gen.generate(16)
        elapsed = time.perf_counter()-t0
        coherent(gen)
        assert gen.stop_reason == 'length' and len(reports) == 16
        returned, history = gen.extract_final_state()
        assert returned is cache
        out['off_control_timing'] = dict(tokens=len(reports), elapsed_s=elapsed,
            tok_s=len(reports)/elapsed, instrumentation='Python donor-call profile guard enabled',
            is_mtp_throughput=False)
        out['terminal_control'] = dict(reason=gen.stop_reason, exact_list=True,
                                       history_frontier=len(history), all_40_layers_coherent=True)
        out['counters'] = dict(replay=sum(s.prompt_replay_count for s in sessions),
            full_cache_repack=pre.live_result._setup.block_runner.full_cache_repack_count,
            initial_handoffs=pre.live_result.handoff_count)
        assert out['counters']['replay'] == out['counters']['full_cache_repack'] == 0
        # Target-side control: no producer exists to test a proposal-side failure.
        failed = TargetGenerationSession.from_prefilled_cache(model, returned, history,
            OMLXDecodeConfig(preserve_mtp=False, omlx_path=cfg.omlx_path))
        sessions.append(failed)
        failed.start(10)
        try:
            failed.speculative_cycle([], _fault=lambda phase: (_ for _ in ()).throw(
                RuntimeError('M53 target control fault')) if phase == 'tentative' else None)
        except RuntimeError as exc:
            assert str(exc) == 'M53 target control fault'
        else:
            raise AssertionError('target fault not raised')
        assert all(c._p6_append_invalid and c._p6_append_failed for c in cache)
        assert all(not hasattr(c, '_accepted_prefix_journal') for c in cache)
        try:
            failed.extract_final_state()
        except RuntimeError:
            pass
        else:
            raise AssertionError('burned target published continuation')
        out['target_failure_control'] = dict(all_aliases_burned=True,
            journal_retired=True, continuation_rejected=True)
        out['native_mtp_calls'] = forbidden_calls
        out['not_qualified'] = ['actual proposal generation', 'all/partial/first reject',
            'proposal RNG isolation', 'canonical stochastic sampling with real proposals',
            'proposal advancement/reset/retirement/failure', 'MTP throughput',
            'native candidate proposal/acceptance/result comparison', 'semantic horizon']
        note('boundary controls passed; M53 end-to-end remains NOT PASS')
        out['status'] = 'BOUNDARY_CONFIRMED'
    except BaseException as exc:
        out['status'] = 'PROBE_FAIL'
        out['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        sys.setprofile(old_profile)
        try:
            for session in reversed(sessions):
                session.close()
            if backend._model is not None:
                readers = [layer.engram.embed for layer in backend._model.language_model.layers
                           if 'engram' in layer]
                admission = backend._runtime.admission
                backend.close()
                out['retirement_control'] = dict(admission_active=admission.active,
                    engram_closed=all(reader._closed for reader in readers),
                    proposal_resources_allocated=0)
            gc.collect()
        finally:
            out['elapsed_s'] = time.perf_counter()-begin
            args.output.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
