"""Real-checkpoint canonical owner qualification; OFF oracles are hash-only."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from ds41f_mlx.config import load_runtime_config
    cfg = load_runtime_config(); cfg.apply_import_paths()
    import mlx.core as mx
    import numpy as np
    from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend
    from ds41f_mlx.runtime.dwarfstar_prefill import DwarfStarMLXPrefillSession
    from ds41f_mlx.prefill_fp8_mlx import handoff_to_generation
    from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
    from ds41f_mlx.runtime.target_generation import TargetGenerationSession

    out = dict(status='RUNNING', cases=[], faults=[], cancellation=[], counters=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    begin = time.monotonic()
    def note(text):
        args.output.write_text(json.dumps(out, indent=2) + '\n')
        print(f'{time.monotonic()-begin:.1f}s {text}', flush=True)
    def digest(v):
        return hashlib.sha256(np.asarray(v.view(mx.uint8)).tobytes()).hexdigest()
    class Sampler:
        def __init__(self, mode):
            self.mode = mode
            self.rng = np.random.default_rng(521)
            self.calls = []
        def __call__(self, lp):
            mx.eval(lp)
            self.calls.append(digest(lp))
            if self.mode == 'mlx_rng':
                return mx.random.categorical(lp, axis=-1)
            if self.mode == 'rng':
                p = np.exp(np.asarray(lp, dtype=np.float64).reshape(-1))
                token = self.rng.choice(len(p), p=p/p.sum())
            elif self.mode in ('script', 'eos'):
                token = 1 if self.mode == 'eos' and len(self.calls) == 3 else 10 + len(self.calls)
            else:
                token = int(mx.argmax(lp).item())
            return mx.array([token], mx.uint32)
        def state(self):
            return dict(calls=list(self.calls), rng=self.rng.bit_generator.state,
                        mlx_rng=[digest(x) for x in mx.random.state] if self.mode == 'mlx_rng' else None)
    def snapshot(gen):
        cache = gen._cache if gen._cache is not None else gen._final_cache
        values = [v for x in cache for v in (*x.cache, x.lengths, x.left_padding) if v is not None]
        mx.eval(*values)
        return dict(cache=[[(list(v.shape), str(v.dtype), digest(v)) if v is not None else None
                           for v in (*x.cache, x.lengths, x.left_padding)] for x in cache],
                    history=gen.current_token_history(), generated=list(gen.generated_tokens),
                    frontier=gen.token_frontier, pending=None if gen._pending is None else int(gen._pending.item()),
                    stopped=gen._stopped, reason=gen.stop_reason, sampling=gen.sampler.state())
    backend = DeepSeekRecipeRuntimeBackend(runtime_config=cfg)
    try:
        note('loading checkpoint')
        backend.load()
        model = backend._model
        ids = backend._runtime.tokenizer.encode('Explain why a computer uses a cache. ' * 100)[:255]
        out['admission'] = backend._runtime.admission.describe()
        out['frontier'] = len(ids)
        def fresh(mode='greedy', max_tokens=40, stop=()):
            pre = DwarfStarMLXPrefillSession(model, omlx_path=cfg.omlx_path).prefill(ids[:-1])
            sampler = Sampler(mode)
            if mode == 'mlx_rng':
                mx.random.seed(521)
            gen = handoff_to_generation(pre.live_result, model, terminal_prompt_token=ids[-1],
                config=OMLXDecodeConfig(preserve_mtp=False, omlx_path=cfg.omlx_path, stop_token_ids=stop),
                max_tokens=max_tokens, sampler=sampler)
            out['counters'].append(dict(replay=gen.prompt_replay_count,
                repack=pre.live_result._setup.block_runner.full_cache_repack_count,
                handoffs=pre.live_result.handoff_count))
            return gen
        def dispose(gen):
            gen.close(); gc.collect(); mx.synchronize()
        for mode in ('greedy', 'script', 'rng', 'mlx_rng'):
            oracle = fresh(mode)
            expected = [snapshot(oracle)]
            tokens = []
            for _ in range(35):
                tokens.append(oracle.next_token().token)
                expected.append(snapshot(oracle))
            dispose(oracle); del oracle
            gen = fresh(mode)
            exact = gen._cache
            n = 0
            for accepted in (7, 3, 0, 0, 6):
                drafts = tokens[n+1:n+8]
                if accepted < 7:
                    drafts[accepted] = (drafts[accepted]+1) % model.language_model._config.vocab_size
                receipt = gen.speculative_cycle(drafts)
                assert receipt['proposal_acceptance_count'] == accepted
                assert receipt['consumed_positions'] == accepted+1
                n += accepted+1
                assert snapshot(gen) == expected[n]
                assert gen._cache is exact
                assert all(not hasattr(x, '_accepted_prefix_journal') for x in exact)
                out['cases'].append(dict(mode=mode, accepted=accepted, consumed=accepted+1,
                                         repeated=True, exact_off=True))
            for _ in range(2):
                gen.next_token(); n += 1
                assert snapshot(gen) == expected[n]
            returned, history = gen.extract_final_state()
            assert returned is exact and history == expected[n]['history']
            # New owner consumes the previously confirmed lookahead as its anchor.
            nxt = TargetGenerationSession.from_prefilled_cache(model, returned, history,
                OMLXDecodeConfig(preserve_mtp=False, omlx_path=cfg.omlx_path), sampler=gen.sampler)
            nxt.start(expected[n]['pending']); n += 1
            state = snapshot(nxt)
            assert state['cache'] == expected[n]['cache']
            assert state['history'] == expected[n]['history']
            assert state['pending'] == expected[n]['pending']
            assert state['sampling'] == expected[n]['sampling']
            dispose(nxt); dispose(gen)
            for accepted in range(8):
                gen = fresh(mode)
                drafts = tokens[1:8]
                if accepted < 7:
                    drafts[accepted] = (drafts[accepted]+1) % model.language_model._config.vocab_size
                receipt = gen.speculative_cycle(drafts)
                assert receipt['proposal_acceptance_count'] == accepted
                assert snapshot(gen) == expected[accepted+1]
                dispose(gen)
            note(f'{mode}: repeated and every consumed prefix 1..8 PASS')
        for max_tokens, stop in ((1, ()), (4, ()), (8, ()), (40, (11,)), (40, (14,))):
            oracle = fresh('script', max_tokens, stop)
            oracle.generate(8)
            expected_terminal = snapshot(oracle)
            dispose(oracle)
            gen = fresh('script', max_tokens, stop)
            exact = gen._cache
            receipt = gen.speculative_cycle(range(12, 19))
            assert snapshot(gen) == expected_terminal
            assert gen._final_cache is exact
            out['cases'].append(dict(terminal=stop, max_tokens=max_tokens, exact_off=True,
                                     consumed=receipt['consumed_positions']))
            dispose(gen)
        # DeepSeek V4.1 EOS id 1, reached inside an accepted draft prefix.
        oracle = fresh('eos', stop=(1,)); oracle.generate(8)
        eos_expected = snapshot(oracle); dispose(oracle)
        gen = fresh('eos', stop=(1,))
        receipt = gen.speculative_cycle([12, 1, 14, 15, 16, 17, 18])
        assert snapshot(gen) == eos_expected and receipt['consumed_positions'] == 3
        out['cases'].append(dict(eos_id=1, consumed=3, exact_off=True))
        dispose(gen)
        gen = fresh('script'); initial = snapshot(gen); dispose(gen)
        for boundary in (1, 2, 5, 10):
            gen = fresh('script'); exact = gen._cache
            calls = [0]
            def cancel():
                calls[0] += 1
                return calls[0] >= boundary
            receipt = gen.speculative_cycle(range(12, 19), cancelled=cancel)
            assert receipt['consumed_positions'] == 0
            assert snapshot(gen) == initial
            assert gen._cache is exact
            assert all(not hasattr(x, '_accepted_prefix_journal') for x in exact)
            out['cancellation'].append(dict(boundary=boundary, consumed=0, exact_off=True))
            # Coherent continuation after cancellation.
            assert gen.next_token().token == 11
            dispose(gen)
        gen = fresh('script')
        requested = [False]
        receipt = gen.speculative_cycle(range(12, 19), cancelled=lambda: requested[0],
            _fault=lambda phase: requested.__setitem__(0, True) if phase == 'acceptance' else None)
        assert receipt['consumed_positions'] == 8 and gen.stop_reason == 'cancelled'
        out['cancellation'].append(dict(boundary='after sampling', deferred=True, consumed=8))
        dispose(gen)
        for phase in ('verify-before', 'tentative', 'materialized', 'acceptance', 'frontier',
                      'settled', 'history', 'lookahead', 'terminal', 'response'):
            gen = fresh('script', max_tokens=1)
            aliases = tuple(gen._cache)
            def fault(p):
                if p == phase:
                    raise RuntimeError('M52 publication fault')
            try:
                gen.speculative_cycle(range(12, 19), _fault=fault)
            except RuntimeError as exc:
                assert str(exc) == 'M52 publication fault'
            else:
                raise AssertionError('fault not reached')
            assert gen._failed
            assert all(x._p6_append_invalid and x._p6_append_failed for x in aliases)
            assert all(not hasattr(x, '_accepted_prefix_journal') for x in aliases)
            for action in (gen.next_token, gen.current_token_history, gen.extract_final_state):
                try:
                    action()
                except RuntimeError:
                    pass
                else:
                    raise AssertionError('burned owner published continuation')
            out['faults'].append(dict(phase=phase, burned=True, retired=True))
            dispose(gen)
            note(f'fault {phase} PASS')
        assert all(c == dict(replay=0, repack=0, handoffs=1) for c in out['counters'])
        prefetch = getattr(model.language_model, '_engram_prefetch', None)
        assert prefetch is None or (prefetch._pending is None and all(
            getattr(layer.engram.embed, '_prefetched', None) is None
            for layer in model.language_model.layers if 'engram' in layer))
        resources = backend._runtime.admission
        backend.close()
        try:
            resources.assert_active()
        except RuntimeError:
            pass
        else:
            raise AssertionError('resource retirement failed')
        out.update(status='PASS', idle_exact_list=True, retirement=True, replay=0, full_cache_repack=0)
        note('M52 PASS')
    except BaseException as exc:
        out.update(status='FAIL', error=repr(exc)); note('FAILED'); raise
    finally:
        backend.close()


if __name__ == '__main__':
    main()
