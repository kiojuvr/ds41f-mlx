"""Generation contract on real MLX with reduced M51 physical-state producer."""
import pytest
from test_m44_target_generation import TinyModel, ROOT, mx
from test_m51_accepted_prefix import StateTarget, fresh, equal
from ds41f_mlx.runtime.accepted_prefix import AcceptedPrefixJournal
from ds41f_mlx.runtime.target_generation import TargetGenerationSession
from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig


class Target(StateTarget):
    def __init__(self):
        super().__init__()
        self.model._config.vocab_size = 128
        self.children = []
    def forward(self, token, cache, frontier, journal=None):
        super().forward(token, cache, frontier, journal)
        return mx.where(mx.arange(128)[None, :] == (token.reshape(1, 1)+1) % 128, 9., 0.)
    def begin_prefix_journal(self, *args):
        child = AcceptedPrefixJournal(self, *args)
        self.children.append(child)
        return child
    def execute(self, token, cache, frontier, sampler, stream):
        self.validate(token, cache, frontier)
        lp = self.forward(token, cache, frontier)
        pending = sampler(lp - mx.logsumexp(lp, axis=-1, keepdims=True))
        mx.eval(pending)
        return int(token.item()), pending


def session(max_tokens=64, stop=(), sampler=None):
    target, cache = Target(), fresh()
    for f in range(5):
        target.forward(mx.array([f]), cache, f)
    gen = TargetGenerationSession(TinyModel(), cache, list(range(5)),
        OMLXDecodeConfig(preserve_mtp=False, omlx_path=ROOT, stop_token_ids=stop),
        sampler=sampler, max_tokens=max_tokens)
    gen.target_forward, gen.language_model = target, target.model
    gen.start(10)
    return gen, target, cache


def compare(a, b):
    equal(a._cache or a._final_cache, b._cache or b._final_cache)
    assert a.current_token_history() == b.current_token_history()
    assert a.generated_tokens == b.generated_tokens
    assert a.token_frontier == b.token_frontier
    assert a._stopped == b._stopped and a.stop_reason == b.stop_reason
    assert (None if a._pending is None else a._pending.tolist()) == (
            None if b._pending is None else b._pending.tolist())


@pytest.mark.parametrize('accepted', range(8))
def test_acceptance_boundaries_off_exact(accepted):
    gen, target, cache = session()
    oracle, _, _ = session()
    drafts = list(range(12, 19))
    if accepted < 7:
        drafts[accepted] = 99
    try:
        receipt = gen.speculative_cycle(drafts)
        assert receipt['confirmed_anchor'] == 11
        assert receipt['proposal_acceptance_count'] == accepted
        assert receipt['consumed_positions'] == accepted + 1
        oracle.generate(accepted + 1)
        compare(gen, oracle)
        assert target.children[-1].phase == 'retired'
        assert not target.children[-1].logits and not target.children[-1].objects
        assert gen._cache is cache
        gen.next_token(); oracle.next_token(); compare(gen, oracle)
    finally:
        gen.close(); oracle.close()


@pytest.mark.parametrize('max_tokens,stop', [(1, ()), (4, ()), (8, ()), (64, (11,)), (64, (14,))])
def test_terminal_publication_off_exact(max_tokens, stop):
    gen, _, exact = session(max_tokens, stop)
    oracle, _, _ = session(max_tokens, stop)
    try:
        gen.speculative_cycle(range(12, 19)); oracle.generate(8)
        compare(gen, oracle)
        assert gen.extract_final_state()[0] is exact
    finally:
        gen.close(); oracle.close()


@pytest.mark.parametrize('boundary', [1, 2, 5, 10])
def test_cancel_zero_without_sampling(boundary):
    calls = []
    def sampler(lp):
        calls.append(1)
        return mx.argmax(lp, axis=-1)
    gen, target, exact = session(sampler=sampler)
    oracle, _, _ = session()
    checks = [0]
    def cancel():
        checks[0] += 1
        return checks[0] >= boundary
    try:
        receipt = gen.speculative_cycle(range(12, 19), cancelled=cancel)
        assert receipt['consumed_positions'] == 0 and len(calls) == 1
        compare(gen, oracle)
        assert gen._cache is exact
        if target.children:
            assert target.children[-1].phase == 'retired'
        gen.next_token(); oracle.next_token(); compare(gen, oracle)
    finally:
        gen.close(); oracle.close()


@pytest.mark.parametrize('phase', ['verify-before', 'tentative', 'materialized', 'acceptance',
    'frontier', 'settled', 'history', 'lookahead', 'terminal', 'response'])
def test_publication_fault_burns_even_idle_aliases(phase):
    gen, target, cache = session(max_tokens=1)
    def fail(p):
        if p == phase:
            raise RuntimeError('fault')
    try:
        with pytest.raises(RuntimeError, match='fault'):
            gen.speculative_cycle(range(12, 19), _fault=fail)
        assert gen._failed
        assert all(x._p6_append_invalid and x._p6_append_failed for x in cache)
        assert all(not hasattr(x, '_accepted_prefix_journal') for x in cache)
        assert all(j.phase in ('retired', 'burned') and not j.objects for j in target.children)
        for action in (gen.next_token, gen.current_token_history, gen.extract_final_state):
            with pytest.raises(RuntimeError):
                action()
    finally:
        gen.close()


@pytest.mark.parametrize('action', ['current_token_history', 'metadata', 'active_cache_offsets',
                                   'next_token', 'stop', 'close', 'extract_final_state'])
def test_reentrant_sampler_burns_and_cannot_observe_state(action):
    gen, _, exact = session()
    gen.sampler = lambda lp: getattr(gen, action)()
    try:
        with pytest.raises(RuntimeError):
            gen.speculative_cycle([12])
        assert all(x._p6_append_invalid for x in exact)
    finally:
        gen.close()


def test_zero_drafts_and_max_bound():
    gen, _, _ = session()
    oracle, _, _ = session()
    try:
        r = gen.speculative_cycle([])
        assert r['proposal_acceptance_count'] == 0 and r['consumed_positions'] == 1
        oracle.next_token(); compare(gen, oracle)
        r = gen.speculative_cycle(range(13, 44))
        assert r['consumed_positions'] == 32 and r['proposal_acceptance_count'] == 31
        oracle.generate(32); compare(gen, oracle)
        with pytest.raises(ValueError):
            gen.speculative_cycle(range(32))
    finally:
        gen.close(); oracle.close()


@pytest.mark.parametrize('attribute', ['_history', 'generated_tokens', 'step_reports'])
def test_mid_python_publication_burn(attribute):
    class Broken(list):
        def extend(self, values):
            super().append(values[0])
            raise RuntimeError('mid-list publication')
    gen, target, exact = session()
    setattr(gen, attribute, Broken(getattr(gen, attribute)))
    try:
        with pytest.raises(RuntimeError, match='mid-list'):
            gen.speculative_cycle(range(12, 19))
        assert target.children[-1].phase == 'retired'
        assert gen._failed and all(x._p6_append_invalid for x in exact)
        with pytest.raises(RuntimeError):
            gen.extract_final_state()
    finally:
        gen.close()


@pytest.mark.parametrize('bad', [mx.array([128]), mx.array([12.]), mx.array([12, 13])])
def test_invalid_sampler_burn(bad):
    gen, _, exact = session()
    gen.sampler = lambda lp: bad
    try:
        with pytest.raises(RuntimeError, match='canonical sampler'):
            gen.speculative_cycle([12])
        assert gen._failed and all(x._p6_append_invalid for x in exact)
    finally:
        gen.close()


def test_sampling_cancel_deferred_and_rng_calls_only_canonical():
    calls = []
    def sampler(lp):
        calls.append(lp.tolist())
        return mx.argmax(lp, axis=-1)
    gen, _, exact = session(sampler=sampler)
    requested = [False]
    try:
        r = gen.speculative_cycle([12, 99, 15], cancelled=lambda: requested[0],
            _fault=lambda p: requested.__setitem__(0, True) if p == 'acceptance' else None)
        assert r['proposal_acceptance_count'] == 1 and r['consumed_positions'] == 2
        assert len(calls) == 3  # bootstrap + exactly two canonical steps
        assert gen.stop_reason == 'cancelled' and gen._pending is None
        assert gen.extract_final_state()[0] is exact
    finally:
        gen.close()
