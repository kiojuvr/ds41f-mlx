"""Real MLX bounded journal/barrier tests (small state; not checkpoint evidence)."""
from types import SimpleNamespace
import gc
import numpy as np
import pytest
from test_m44_target_generation import mx
from ds41f_mlx.runtime.accepted_prefix import AcceptedPrefixJournal
from ds41f_mlx.runtime.state_production import DecodeStateProducer
from ds41f_mlx.model_execution.cache import DeepseekV41Cache


class StateTarget:
    """Explicit state-transition double with irreversible row/group replacement."""
    def __init__(self):
        self.mx = mx
        self.resources = SimpleNamespace(assert_active=lambda: None)
        c = SimpleNamespace(window_size=4, n_layers=40, compress_ratios=(2,) * 40,
                            kv_source_layers=tuple(range(40)), index_source_layers=tuple(range(40)),
                            engram_layer_ids=(0,), engram_max_ngram_size=3)
        self.model = SimpleNamespace(_config=c, _hasher=self.hash)
        self.tap_rows = None
        self.producer = DecodeStateProducer(mx, None)

    def validate(self, token, cache, frontier):
        assert len(cache) == 40
        if any(x.size() != frontier or getattr(x, '_p6_append_pending', False)
               or getattr(x, '_p6_append_invalid', False) for x in cache):
            raise RuntimeError('not canonical')

    @staticmethod
    def invalidate(cache):
        for item in cache:
            item._p6_append_failed = item._p6_append_invalid = True

    @staticmethod
    def hash(ids, history, image_mask):
        joined = np.concatenate([np.asarray(history), np.asarray(ids)], 1)
        return None, joined[:, -2:]

    def forward(self, token, cache, frontier, journal=None):
        width = token.shape[0]
        for i, item in enumerate(cache):
            new = token.reshape(1, width, 1).astype(mx.float32) + i
            if journal:
                journal.window_write(i, item[1], 4, new)
                journal.compressor_write(i, new, new + 1)
            item[1] = mx.concatenate([item[1], new], 1)[:, -4:]
            kv = mx.concatenate([item[4], new], 1)
            gate = mx.concatenate([item[5], new + 1], 1)
            cutoff = kv.shape[1] // 2 * 2
            if cutoff:
                item[2] = mx.concatenate([item[2], mx.sum(
                    kv[:, :cutoff].reshape(1, -1, 2, 1), axis=2)], 1)
                item[3] = mx.concatenate([item[3], mx.sum(
                    gate[:, :cutoff].reshape(1, -1, 2, 1), axis=2)], 1)
            item[4], item[5] = kv[:, cutoff:], gate[:, cutoff:]
            item[0] = mx.array([frontier + width], mx.int32)
            if i == 0:
                item[6] = mx.array(self.hash(token[None], item[6], None)[1], mx.int64)
            item.lengths -= width
            item.left_padding -= width
        return token.astype(mx.float32) if width == 1 else token.reshape(1, width, 1).astype(mx.float32)


def fresh():
    cache = [DeepseekV41Cache(2) for _ in range(40)]
    for i, x in enumerate(cache):
        x[0] = mx.array([0], mx.int32)
        for s in range(1, 6):
            x[s] = mx.zeros((1, 0, 1), mx.float32)
        x[6] = mx.zeros((1, 2 if i == 0 else 0), mx.int64)
        x.lengths = mx.array([100], mx.int32)
        x.left_padding = mx.array([0], mx.int32)
    return cache


def equal(a, b):
    for x, y in zip(a, b):
        for p, q in zip((*x.cache, x.lengths, x.left_padding),
                        (*y.cache, y.lengths, y.left_padding)):
            assert p.shape == q.shape and p.dtype == q.dtype
            assert bool(mx.array_equal(p, q).item())


@pytest.mark.parametrize('accepted', range(9))
def test_all_arbitrary_prefixes_match_independent_off(accepted):
    target = StateTarget()
    actual, oracle = fresh(), fresh()
    for f in range(5):
        target.forward(mx.array([f]), actual, f)
        target.forward(mx.array([f]), oracle, f)
    identities = tuple(actual)
    journal = AcceptedPrefixJournal(target, actual, 5, 8, mx.default_stream(mx.gpu))
    for j in range(8):
        journal.advance(mx.array([10 + j]))
    journal.complete()
    payload = journal.payload_bytes
    assert payload < 100000
    owner_frontier = [5]
    def publish(end):
        assert owner_frontier[0] == 5
        assert all(x._p6_append_pending and x.size() == end for x in actual)
        owner_frontier[0] = end
    assert journal.settle(accepted, publish=publish) == 5 + accepted
    assert owner_frontier[0] == 5 + accepted
    for j in range(accepted):
        target.forward(mx.array([10 + j]), oracle, 5 + j)
    equal(actual, oracle)
    assert tuple(actual) == identities
    assert journal.phase == 'retired' and journal.objects == ()
    assert not journal.projections and not journal.logits and not journal.initial
    assert all(not x._p6_append_pending and not hasattr(x, '_accepted_prefix_journal') for x in actual)


@pytest.mark.parametrize('after', [False, True])
def test_cancellation_restores_committed_state(after):
    target = StateTarget()
    actual, oracle = fresh(), fresh()
    journal = AcceptedPrefixJournal(target, actual, 0, 3, mx.default_stream(mx.gpu))
    for j in range(3):
        journal.advance(mx.array([j]))
    if after:
        journal.complete()
    assert journal.cancel() == 0
    equal(actual, oracle)


@pytest.mark.parametrize('phase,layer,slot', [
    ('materialize-before', -1, -1), ('materialize-after', -1, -1),
    ('materialize-state', 19, -1),
    ('prepare', 19, -1), ('publish', 0, 1), ('publish', 19, 4),
    ('publish', 39, 'lengths'), ('publication-boundary', -1, -1)])
def test_barrier_failure_burns_all_aliases(phase, layer, slot):
    target, actual = StateTarget(), fresh()
    aliases = tuple(actual)
    def fail(p, i, s):
        assert all(x._p6_append_pending for x in aliases)
        if (p, i, s) == (phase, layer, slot):
            raise RuntimeError('injected')
    journal = AcceptedPrefixJournal(target, actual, 0, 3, mx.default_stream(mx.gpu), fail)
    for j in range(3):
        journal.advance(mx.array([j]))
    with pytest.raises(RuntimeError, match='injected'):
        journal.complete()
        journal.settle(1, publish=lambda end: None)
    assert journal.phase == 'burned' and journal.objects == ()
    assert all(x._p6_append_failed and x._p6_append_invalid for x in aliases)
    assert not journal.projections and not journal.initial


def test_exact_list_substitution_burns_original_and_replacement():
    target, actual = StateTarget(), fresh()
    aliases = tuple(actual)
    journal = AcceptedPrefixJournal(target, actual, 0, 2, mx.default_stream(mx.gpu))
    journal.advance(mx.array([1]))
    replacement = fresh()[0]
    actual[19] = replacement
    with pytest.raises(RuntimeError, match='exact target lease'):
        journal.complete()
    assert all(x._p6_append_invalid for x in (*aliases, replacement))


def test_detached_bounded_slice_does_not_retain_parent_or_lazy_graph():
    # Direct allocation qualification of the journal storage primitive.
    target = StateTarget()
    journal = AcceptedPrefixJournal(target, fresh(), 0, 1, mx.default_stream(mx.gpu))
    gc.collect(); mx.synchronize()
    baseline = mx.get_active_memory()
    huge = mx.ones((1, 10000000), mx.bfloat16)
    mx.eval(huge)
    small = journal.detach(huge[:, :1])
    del huge
    gc.collect(); mx.synchronize()
    assert mx.get_active_memory() - baseline < 4096
    assert small.item() == 1
    journal.cancel()


@pytest.mark.parametrize('frontier', [5, 1001, 10001])
def test_payload_size_is_context_independent(frontier):
    target, actual = StateTarget(), fresh()
    for item in actual:
        item[0] = mx.array([frontier], mx.int32)
        for s in (1, 2, 3, 4, 5):
            n = min(frontier, 4) if s == 1 else frontier // 2 if s in (2, 3) else frontier % 2
            item[s] = mx.ones((1, n, 1), mx.float32)
    journal = AcceptedPrefixJournal(target, actual, frontier, 8, mx.default_stream(mx.gpu))
    for j in range(8):
        journal.advance(mx.array([j]))
    journal.complete()
    # Tails: 40*2*4; row metadata:40*2*4; history:16;
    # evictions:8*40*4; projections:8*40*2*4; history:8*16.
    assert journal.payload_bytes == 4624
    journal.cancel()


def test_incomplete_projection_coverage_burns_not_shape_only_pass():
    target, actual = StateTarget(), fresh()
    journal = AcceptedPrefixJournal(target, actual, 0, 3, mx.default_stream(mx.gpu))
    journal.advance(mx.array([1]))
    del journal.projections[19]
    with pytest.raises(RuntimeError, match='incomplete source projection'):
        journal.complete()
    assert all(x._p6_append_invalid for x in actual)


def test_owner_publication_fault_burns_even_if_frontier_was_advanced():
    target, actual = StateTarget(), fresh()
    journal = AcceptedPrefixJournal(target, actual, 0, 3, mx.default_stream(mx.gpu))
    journal.advance(mx.array([1])); journal.complete()
    owner = [0]
    def publish(end):
        assert all(x._p6_append_pending for x in actual)
        owner[0] = end
        raise RuntimeError('owner publication fault')
    with pytest.raises(RuntimeError, match='owner publication fault'):
        journal.settle(1, publish=publish)
    assert owner[0] == 1
    assert journal.phase == 'burned'
    assert all(x._p6_append_invalid for x in actual)


def test_parent_idle_cannot_resurrect_burned_zero_prefix_aliases():
    from test_m44_target_generation import session
    _, gen = session()
    gen.start(3)
    aliases = tuple(gen._cache)
    for item in aliases:
        item._p6_append_invalid = item._p6_append_failed = True
    with pytest.raises(RuntimeError, match='unavailable'):
        gen.extract_final_state()
    assert gen._failed and all(x._p6_append_invalid for x in aliases)
    gen.close()


def test_parent_idle_rejects_borrow_without_removing_markers():
    from test_m44_target_generation import session
    _, gen = session()
    gen.start(3)
    for item in gen._cache:
        item._accepted_prefix_journal = object()
        item._p6_append_pending = True
    with pytest.raises(RuntimeError, match='during target borrow'):
        gen.extract_final_state()
    assert not gen._stopped
    for item in gen._cache:
        del item._accepted_prefix_journal
        item._p6_append_pending = False
    gen.close()


def test_context_exit_cancels_or_burns_and_drops_resource_handles():
    target, actual = StateTarget(), fresh()
    with AcceptedPrefixJournal(target, actual, 0, 3, mx.default_stream(mx.gpu)) as journal:
        journal.advance(mx.array([1]))
    equal(actual, fresh())
    assert journal.target is journal.mx is journal.stream is None
    with pytest.raises(RuntimeError, match='cancel exception'):
        with AcceptedPrefixJournal(target, actual, 0, 3, mx.default_stream(mx.gpu)) as journal:
            journal.advance(mx.array([1]))
            raise RuntimeError('cancel exception')
    assert journal.phase == 'burned' and all(x._p6_append_invalid for x in actual)


def test_owner_close_burns_borrow_and_does_not_abandon_resources():
    from test_m44_target_generation import session
    _, gen = session()
    target, actual = StateTarget(), fresh()
    gen._cache = actual
    journal = AcceptedPrefixJournal(target, actual, 0, 3, mx.default_stream(mx.gpu))
    journal.advance(mx.array([1]))
    gen.close()
    assert journal.phase == 'burned' and journal.target is None
    assert all(x._p6_append_invalid for x in actual)
    assert gen._final_cache is None


@pytest.mark.parametrize('field', [0, 'lengths', 'left_padding'])
def test_tentative_metadata_corruption_cannot_pass_completion(field):
    target, actual = StateTarget(), fresh()
    journal = AcceptedPrefixJournal(target, actual, 0, 3, mx.default_stream(mx.gpu))
    journal.advance(mx.array([1]))
    if field == 0:
        actual[19][0] = mx.array([0], mx.int32)
    else:
        setattr(actual[19], field, mx.array([123], mx.int32))
    with pytest.raises(RuntimeError, match='mismatch'):
        journal.complete()
    assert journal.phase == 'burned' and all(x._p6_append_invalid for x in actual)


def test_tentative_span_cannot_overrun_row_admission():
    target, actual = StateTarget(), fresh()
    for item in actual:
        item.lengths = mx.array([1], mx.int32)
    journal = AcceptedPrefixJournal(target, actual, 0, 3, mx.default_stream(mx.gpu))
    journal.advance(mx.array([1]))
    with pytest.raises(ValueError, match='unpadded admission'):
        journal.advance(mx.array([2]))
    assert journal.phase == 'burned' and all(x._p6_append_invalid for x in actual)


@pytest.mark.parametrize('accepted', [0, 1, 19, 31, 32])
def test_hard_max_span_matches_independent_off(accepted):
    target, actual, oracle = StateTarget(), fresh(), fresh()
    journal = AcceptedPrefixJournal(target, actual, 0, 32, mx.default_stream(mx.gpu))
    for j in range(32):
        journal.advance(mx.array([10 + j]))
    with pytest.raises(RuntimeError, match='span exhausted'):
        journal.advance(mx.array([42]))
    journal.complete()
    journal.settle(accepted, publish=lambda end: None)
    for j in range(accepted):
        target.forward(mx.array([10 + j]), oracle, j)
    equal(actual, oracle)
    assert journal.phase == 'retired'
