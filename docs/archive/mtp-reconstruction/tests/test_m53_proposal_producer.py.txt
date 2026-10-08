"""Producer lifecycle on reduced real MLX; arithmetic is a stub, labeled here."""
from types import SimpleNamespace
import numpy as np
import pytest
from test_m53_proposal_primitives import tapped_session
from ds41f_mlx.runtime.dspark_proposal import DSparkProposalProducer
from ds41f_mlx.runtime.hidden_taps import CommittedTapReceipt
from ds41f_mlx.model_execution import dspark
from test_m44_target_generation import mx


def producer_session(monkeypatch):
    gen, target, child, cache = tapped_session()
    child.producers = set()
    child.config.n_mtp_layers = 3
    child.config.dspark_block_size = 3
    child.config.dspark_noise_token_id = 0
    child.model = SimpleNamespace(embed=None, head=None)
    class Stage:
        main_proj = staticmethod(lambda x: x)
        main_norm = staticmethod(lambda x: x)
        markov_head = staticmethod(lambda ids: (mx.zeros((1, 128)), None))
        class attn:
            @staticmethod
            def append_context(x, ring, start_offset):
                ring.append(x[:, None], start_offset=start_offset)
    child.parameters = SimpleNamespace(mtp=[Stage(), Stage(), Stage()])
    def math(view, anchor, rings, width):
        tokens = anchor.reshape(1, 1, 1) + mx.arange(1, width + 1)[None, :, None]
        return mx.where(mx.arange(128)[None, None, :] == tokens, 9., 0.), None
    monkeypatch.setattr(dspark, 'proposal_forward', math)
    # This legitimate forward supplies the bootstrap receipt.
    gen.next_token()
    seed = CommittedTapReceipt(child, gen, gen.token_frontier - 5, mx.zeros((1, 4, 3)))
    producer = DSparkProposalProducer(gen, seed)
    return gen, child, producer, cache


def test_producer_rng_and_repeated_cycles(monkeypatch):
    gen, child, producer, cache = producer_session(monkeypatch)
    try:
        mx.random.seed(53)
        expected = mx.random.uniform(shape=(10,)); mx.eval(expected)
        mx.random.seed(53)
        for _ in range(3):
            r = producer.cycle()
            assert r['proposal_acceptance_count'] == 3 and r['consumed_positions'] == 4
            assert producer.frontier == gen.token_frontier
            assert {ring.offset for ring in producer.rings} == {gen.token_frontier}
            assert not child.receipts
            assert gen._cache is cache
        actual = mx.random.uniform(shape=(10,)); mx.eval(actual)
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))
    finally:
        gen.close()
    assert not producer.active and not child.producers and not child.receipts
    assert all(ring.keys is None for ring in producer.rings)


@pytest.mark.parametrize('accepted', [0, 1, 2])
def test_ring_advances_only_consumed_inputs(monkeypatch, accepted):
    gen, child, producer, _ = producer_session(monkeypatch)
    try:
        drafts = list(producer.propose())
        drafts[accepted] = 99
        r = gen.speculative_cycle(drafts)
        assert r['proposal_acceptance_count'] == accepted
        producer.advance(gen.take_tap_receipt())
        assert producer.frontier == gen.token_frontier
        assert all(99 not in ring.keys.reshape(-1).tolist() for ring in producer.rings)
        assert not child.receipts
    finally:
        gen.close()


@pytest.mark.parametrize('boundary', ['before', 'after'])
def test_proposal_failure_does_not_burn_committed_target(monkeypatch, boundary):
    gen, child, producer, cache = producer_session(monkeypatch)
    try:
        def fail(phase):
            raise RuntimeError('proposal failure')
        if boundary == 'before':
            history = gen.current_token_history()
            with pytest.raises(RuntimeError):
                producer.propose(_fault=fail)
        else:
            r = gen.speculative_cycle(producer.propose())
            history = gen.current_token_history()
            receipt = gen.take_tap_receipt()
            with pytest.raises(RuntimeError):
                producer.advance(receipt, _fault=fail)
            assert receipt.retired
        assert gen.current_token_history() == history
        assert not producer.active and not child.receipts
        assert all(not getattr(c, '_p6_append_failed', False) for c in cache)
        gen.disable_proposals()
        gen.next_token()
        assert set(gen.active_cache_offsets()) == {len(gen.current_token_history())}
    finally:
        gen.close()


def test_warm_ring_requires_full_legitimate_window(monkeypatch):
    gen, child, producer, _ = producer_session(monkeypatch)
    try:
        producer.retire()
        gen.next_token()
        producer = DSparkProposalProducer(gen)
        assert producer.rings[0].keys.shape[2] == 1
        with pytest.raises(RuntimeError, match='priming'):
            producer.propose()
        assert not producer.active
        gen.next_token()
        producer = DSparkProposalProducer(gen)
        for _ in range(child.config.window_size - 1):
            gen.next_token()
            producer.advance(gen.take_tap_receipt())
        assert all(r.keys.shape[2] == 4 for r in producer.rings)
        assert len(producer.propose()) == 3
    finally:
        gen.close()


def test_deferred_cancel_retires_derived_state(monkeypatch):
    gen, child, producer, _ = producer_session(monkeypatch)
    try:
        armed = [False]
        r = producer.cycle(cancelled=lambda: armed[0], _fault=lambda p:
            armed.__setitem__(0, True) if p == 'acceptance' else None)
        assert r['consumed_positions'] == 4 and r['cancelled']
        assert gen.stop_reason == 'cancelled' and not producer.active
        assert not child.receipts and all(r.keys is None for r in producer.rings)
        assert set(gen.active_cache_offsets()) == {len(gen.current_token_history())}
    finally:
        gen.close()


def test_cancel_preserves_ring_then_target_failure_retires(monkeypatch):
    gen, child, producer, cache = producer_session(monkeypatch)
    try:
        frontier = producer.frontier
        hashes = [ring.keys.tolist() for ring in producer.rings]
        armed = [False]
        r = producer.cycle(cancelled=lambda: armed[0], _fault=lambda p:
            armed.__setitem__(0, True) if p == 'materialized' else None)
        assert r['consumed_positions'] == 0 and producer.frontier == frontier
        assert hashes == [ring.keys.tolist() for ring in producer.rings]
        with pytest.raises(RuntimeError):
            producer.cycle(_fault=lambda p: (_ for _ in ()).throw(RuntimeError('target'))
                           if p == 'tentative' else None)
        assert not producer.active and not child.receipts
        assert all(c._p6_append_failed for c in cache)
    finally:
        gen.close()
