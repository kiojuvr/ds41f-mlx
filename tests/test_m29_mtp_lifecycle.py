from __future__ import annotations

import types

import pytest

from ds41f_mlx.runtime.mtp_lifecycle import (
    CanonicalTransportHistory,
    DSparkCommittedContext,
    MTPLifecycleError,
    canonical_quiesce_native_singleton,
    CanonicalPhysicalConnection,
    OMLXMTPGenerationSession,
)


class FakeCache:
    def __init__(self, n: int):
        self.n = n
    def size(self):
        return self.n
    def advance(self, k: int):
        self.n += k


class FakeRing:
    def __init__(self, offset: int):
        self.offset = offset


class FakeConfig:
    dspark_target_layer_ids = [37, 38, 39]
    dspark_block_size = 128


class FakeLM:
    _config = FakeConfig()
    def __call__(self, arr, cache=None, return_dspark_hidden=False):
        tokens = arr.tolist()[0]
        for c in cache:
            c.advance(len(tokens))
        return object(), object()
    def dspark_append_context(self, hidden, caches, start_offset: int):
        for c in caches:
            c.offset = start_offset + 1


class FakeArray:
    def __init__(self, tokens):
        self.tokens = tokens
    def tolist(self):
        return self.tokens


class FakeMX:
    int64 = "int64"
    def array(self, data, dtype=None):
        return FakeArray(data)


def target(n):
    return [FakeCache(n) for _ in range(40)]


def state(offset, queue):
    return types.SimpleNamespace(mtp_cache=[FakeRing(offset) for _ in range(3)], queue=list(queue), drafts=object(), next_main=object(), rollback_stash=object(), uid=7)


def test_canonical_and_transport_frontiers_are_distinct():
    h = CanonicalTransportHistory(prompt_tokens=(1, 2, 3))
    h.record_delivered(10)
    h.commit_undelivered([11, 12])
    assert h.canonical_frontier == 6
    assert h.delivered_frontier == 4
    assert h.recovery_suffix_tokens == [11, 12]


def test_m28_cache_ahead_quiescence_drains_only_existing_queue_prefix():
    h = CanonicalTransportHistory(prompt_tokens=(1, 2, 3))
    h.record_delivered(10)  # canonical frontier 4, transport frontier 4
    st = state(6, [(11, None, "accepted"), (12, None, "accepted"), (99, None, "future")])
    res = canonical_quiesce_native_singleton(language_model=FakeLM(), target_cache=target(6), mtp_state=st, history=h, mx=FakeMX())
    assert res.canonical_tokens == (1, 2, 3, 10, 11, 12)
    assert res.recovery_suffix_tokens == (11, 12)
    assert res.discarded_future_token == 99
    assert res.counters.new_verify_cycles == 0
    assert res.counters.new_proposals == 0
    assert res.counters.target_forwards == 0
    assert res.counters.history_replay == 0
    assert res.counters.full_cache_repack == 0
    assert st.queue == []
    assert [r.offset for r in st.mtp_cache] == [6, 6, 6]


def test_m28_target_behind_one_token_materializes_bounded_suffix():
    h = CanonicalTransportHistory(prompt_tokens=(1, 2, 3))
    h.record_delivered(10)
    st = state(3, [])
    res = canonical_quiesce_native_singleton(language_model=FakeLM(), target_cache=target(3), mtp_state=st, history=h, mx=FakeMX())
    assert res.canonical_tokens == (1, 2, 3, 10)
    assert res.counters.target_forwards == 1
    assert res.counters.dspark_appends == 1
    assert [r.offset for r in st.mtp_cache] == [4, 4, 4]


def test_unknown_queue_topology_fails_closed():
    h = CanonicalTransportHistory(prompt_tokens=(1, 2, 3))
    st = state(5, [(10, None, "accepted")])
    with pytest.raises(MTPLifecycleError, match="unknown queue topology"):
        canonical_quiesce_native_singleton(language_model=FakeLM(), target_cache=target(5), mtp_state=st, history=h, mx=FakeMX())


def test_connection_separates_consumed_queue_suffix_and_pending_prediction():
    # New host authority boundary: a committed accepted prefix is not an ACK,
    # and its final bonus remains an unconsumed prediction.
    h = CanonicalTransportHistory(prompt_tokens=(1, 2, 3))
    h.commit_undelivered([10])
    st = state(6, [(11, None, 'accepted'), (12, None, 'accepted'), (99, None, 'bonus')])
    st.hist_offset = 6
    connection = CanonicalPhysicalConnection(rng_draws=0)
    connection.bind(h, st.mtp_cache, st)
    assert connection.consumed_tokens == (1, 2, 3, 10, 11, 12)
    assert connection.queue_ahead == (11, 12)
    assert connection.pending_prediction == 99
    assert h.canonical_frontier == 4 and h.delivered_frontier == 3
    h.commit_undelivered([11, 12])
    connection.bind(h, st.mtp_cache, None)
    connection.retire('settled')
    assert connection.pending_prediction is None and connection.rng_draws == 0
    with pytest.raises(MTPLifecycleError, match='retired'):
        connection.bind(h, st.mtp_cache, None)


@pytest.mark.parametrize('native_uid', [7, 8])
def test_connection_publication_failure_burns_native_aliases(native_uid):
    # Exactly the introduced boundary: native next mutates successfully, then
    # metadata/history publication fails. No uncertain owner may become idle.
    from threading import RLock
    sess = object.__new__(OMLXMTPGenerationSession)
    sess._operation_lock = RLock()
    sess._operation_failed = sess._closed = sess._quiesced = False
    sess._started = True
    sess.semantic_guard = None
    sess.connection = CanonicalPhysicalConnection()
    sess.history = CanonicalTransportHistory(prompt_tokens=(1, 2, 3))
    calls = []
    sess.uid = 7
    def native_next():
        calls.append('mutate')
        return [], [types.SimpleNamespace(uid=native_uid, token=10, logprobs=object(), finish_reason=None)]
    sess._bg = types.SimpleNamespace(next=native_next, remove=lambda ids: calls.append('remove'),
                                    close=lambda: calls.append('close'))
    sess.mx = types.SimpleNamespace(synchronize=lambda stream: None)
    sess.stream = None
    def fail():
        raise RuntimeError('publication failure')
    sess._bind_connection = fail
    with pytest.raises(RuntimeError, match='publication failure' if native_uid == 7 else 'foreign native response'):
        sess.next_token(transport_delivered=False)
    if native_uid != 7:
        assert not sess.history.canonical_generated_tokens
    assert calls == ['mutate', 'remove', 'close']
    assert sess.connection.disposition == 'fault' and sess._bg is None
    assert sess.uid is sess.dspark_context is sess.model is sess.language_model is None
    with pytest.raises(MTPLifecycleError, match='failed closed'):
        sess.quiesce()
    sess.close()  # retirement remains idempotent


def test_dspark_context_validates_offsets():
    ctx = DSparkCommittedContext.from_native([FakeRing(7), FakeRing(7), FakeRing(7)], frontier=7, target_layer_ids=[37, 38, 39], window_size=128)
    assert ctx.stage_count == 3
    with pytest.raises(MTPLifecycleError):
        DSparkCommittedContext.from_native([FakeRing(7), FakeRing(8)], frontier=7, target_layer_ids=[37, 38])
