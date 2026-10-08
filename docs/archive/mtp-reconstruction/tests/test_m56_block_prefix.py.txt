"""Bounded block undo tests; arithmetic double, NOT checkpoint qualification."""
import numpy as np
import pytest
from test_m51_accepted_prefix import StateTarget, fresh, equal, mx
from ds41f_mlx.runtime.accepted_prefix import AcceptedPrefixJournal


class BlockTarget(StateTarget):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.model._hasher = self.hash
        self.tap_rows = None

    @staticmethod
    def hash(ids, history, image_mask):
        joined = np.concatenate([np.asarray(history), np.asarray(ids)], 1)
        return None, joined[:, -2:]

    def forward(self, tokens, cache, frontier, journal=None):
        self.calls += 1
        if tokens.shape == (1,):
            out = super().forward(tokens, cache, frontier, journal)
            self.tap_rows = tokens.reshape(1, 1, 1).astype(mx.float32)
            return out
        width = tokens.shape[0]
        for i, item in enumerate(cache):
            new = tokens.reshape(1, width, 1).astype(mx.float32) + i
            journal.window_write(i, item[1], 4, new)
            journal.compressor_write(i, new, new + 1)
            item[1] = mx.concatenate([item[1], new], 1)[:, -4:]
            kv = mx.concatenate([item[4], new], 1)
            gate = mx.concatenate([item[5], new + 1], 1)
            cutoff = kv.shape[1] // 2 * 2
            item[2] = mx.concatenate([item[2], mx.sum(
                kv[:, :cutoff].reshape(1, -1, 2, 1), axis=2)], 1)
            item[3] = mx.concatenate([item[3], mx.sum(
                gate[:, :cutoff].reshape(1, -1, 2, 1), axis=2)], 1)
            item[4], item[5] = kv[:, cutoff:], gate[:, cutoff:]
            item[0] = mx.array([frontier + width], mx.int32)
            if i == 0:
                item[6] = mx.array(self.hash(tokens[None], item[6], None)[1], mx.int64)
            item.lengths -= width
            item.left_padding -= width
        self.tap_rows = tokens.reshape(1, width, 1).astype(mx.float32)
        return self.tap_rows


@pytest.mark.parametrize('frontier', [0, 1, 3, 4, 5])
@pytest.mark.parametrize('accepted', range(9))
def test_block_every_prefix_eviction_and_compression(frontier, accepted):
    target = BlockTarget()
    actual, oracle = fresh(), fresh()
    for f in range(frontier):
        target.forward(mx.array([f]), actual, f)
        target.forward(mx.array([f]), oracle, f)
    identities = tuple(actual)
    journal = AcceptedPrefixJournal(target, actual, frontier, 8, mx.default_stream(mx.gpu))
    before = target.calls
    journal.advance_block(mx.arange(10, 18))
    assert target.calls - before == 1
    assert len(journal.logits) == len(journal.histories) == len(journal.tap_rows) == 8
    assert [int(row.item()) for row in journal.tap_rows] == list(range(10, 18))
    journal.complete()
    journal.settle(accepted, publish=lambda end: None)
    for row in range(accepted):
        target.forward(mx.array([10 + row]), oracle, frontier + row)
    equal(actual, oracle)
    assert tuple(actual) == identities
    assert journal.phase == 'retired' and not journal.tap_rows


@pytest.mark.parametrize('after', [False, True])
def test_block_cancel_zero_prefix(after):
    target = BlockTarget()
    actual, oracle = fresh(), fresh()
    journal = AcceptedPrefixJournal(target, actual, 0, 8, mx.default_stream(mx.gpu))
    journal.advance_block(mx.arange(8))
    if after:
        journal.complete()
    journal.cancel()
    equal(actual, oracle)


@pytest.mark.parametrize('phase,layer,slot', [
    ('materialize-before', -1, -1), ('materialize-after', -1, -1),
    ('materialize-state', 19, -1), ('prepare', 19, -1),
    ('publish', 0, 1), ('publish', 19, 4), ('publish', 39, 'lengths'),
    ('publication-boundary', -1, -1)])
def test_block_fault_burn(phase, layer, slot):
    target, actual = BlockTarget(), fresh()
    aliases = tuple(actual)
    def fault(p, i, s):
        assert all(x._p6_append_pending for x in aliases)
        if (p, i, s) == (phase, layer, slot):
            raise RuntimeError('injected')
    journal = AcceptedPrefixJournal(target, actual, 0, 8, mx.default_stream(mx.gpu), fault)
    journal.advance_block(mx.arange(8))
    with pytest.raises(RuntimeError, match='injected'):
        journal.complete()
        journal.settle(3, publish=lambda end: None)
    assert journal.phase == 'burned'
    assert all(x._p6_append_failed and x._p6_append_invalid for x in aliases)


def test_mixed_blocks_and_rows():
    target, actual, oracle = BlockTarget(), fresh(), fresh()
    journal = AcceptedPrefixJournal(target, actual, 0, 8, mx.default_stream(mx.gpu))
    journal.advance(mx.array([10]))
    journal.advance_block(mx.arange(11, 15))
    journal.advance_block(mx.arange(15, 18))
    journal.complete()
    journal.settle(6, publish=lambda end: None)
    for row in range(6):
        target.forward(mx.array([10 + row]), oracle, row)
    equal(actual, oracle)


@pytest.mark.parametrize('accepted', range(33))
def test_max_span_four_physical_blocks(accepted):
    target, actual, oracle = BlockTarget(), fresh(), fresh()
    journal = AcceptedPrefixJournal(target, actual, 0, 32, mx.default_stream(mx.gpu))
    for start in range(0, 32, 8):
        journal.advance_block(mx.arange(start, start + 8))
    assert target.calls == 4
    journal.complete()
    journal.settle(accepted, publish=lambda end: None)
    for row in range(accepted):
        target.forward(mx.array([row]), oracle, row)
    equal(actual, oracle)


def test_block_forward_fault_burns_no_shadow_cache():
    target, actual = BlockTarget(), fresh()
    aliases = tuple(actual)
    original = target.forward
    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError('target fault')
    target.forward = fail
    journal = AcceptedPrefixJournal(target, actual, 0, 8, mx.default_stream(mx.gpu))
    with pytest.raises(RuntimeError, match='target fault'):
        journal.advance_block(mx.arange(8))
    assert journal.phase == 'burned'
    assert all(x._p6_append_invalid for x in aliases)
