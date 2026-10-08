"""Physical row boundaries; logical prefix/barrier tests remain in M51–M54."""
import gc
import numpy as np
import pytest
from test_m51_accepted_prefix import StateTarget, fresh, mx
from ds41f_mlx.runtime.accepted_prefix import AcceptedPrefixJournal


def test_journal_hooks_never_read_back_inside_backbone(monkeypatch):
    target, cache = StateTarget(), fresh()
    stream = mx.default_stream(mx.gpu)
    journal = AcceptedPrefixJournal(target, cache, 0, 2, stream)
    forward = target.forward
    original_eval, original_array = mx.eval, np.array
    executing = [False]
    counts = {'eval': 0, 'copy': 0}

    def evaluate(*values):
        assert not executing[0], 'layer-local evaluation fragments backbone'
        counts['eval'] += 1
        return original_eval(*values)

    def copy(*args, **kwargs):
        assert not executing[0], 'layer-local host copy fragments backbone'
        counts['copy'] += 1
        return original_array(*args, **kwargs)

    def execute(*args, **kwargs):
        executing[0] = True
        try:
            result = forward(*args, **kwargs)
            target.tap_rows = mx.ones((1, 1, 2)) * args[0].reshape(1, 1, 1)
            return result
        finally:
            executing[0] = False

    monkeypatch.setattr(mx, 'eval', evaluate)
    monkeypatch.setattr(np, 'array', copy)
    target.forward = execute
    journal.advance(mx.array([17]))
    assert counts == {'eval': 1, 'copy': 0}  # one on-device row region, including allocated undo/taps
    assert target.tap_rows is None
    assert np.asarray(journal.tap_rows[0]).tolist() == [[[17., 17.]]]
    assert journal.payload_bytes > 0
    journal.cancel()
    assert journal.phase == 'retired'


def test_row_copy_failure_burns_every_alias(monkeypatch):
    target, cache = StateTarget(), fresh()
    journal = AcceptedPrefixJournal(target, cache, 0, 2, mx.default_stream(mx.gpu))
    def fail(*args, **kwargs):
        raise RuntimeError('row copy fault')
    monkeypatch.setattr(mx, 'take', fail)
    with pytest.raises(RuntimeError, match='row copy fault'):
        journal.advance(mx.array([1]))
    assert journal.phase == 'burned'
    assert all(x._p6_append_invalid and x._p6_append_failed for x in cache)
    assert not journal.tap_rows and not journal.projections


@pytest.mark.parametrize('dtype', [mx.uint8, mx.int64, mx.float32, mx.bfloat16])
def test_device_copy_preserves_exact_storage_and_drops_parent(dtype):
    target = StateTarget()
    journal = AcceptedPrefixJournal(target, fresh(), 0, 1, mx.default_stream(mx.gpu))
    parent = mx.arange(1024 * 1024).astype(dtype).reshape(1, -1)
    mx.eval(parent)
    original = np.asarray(parent[:, :128].view(mx.uint8)).copy()
    before = mx.get_active_memory()
    retained = journal.detach(parent[:, :128])
    empty = journal.detach(parent[:, :0])
    del parent
    gc.collect()
    mx.synchronize()
    assert np.array_equal(np.asarray(retained.view(mx.uint8)), original)
    assert mx.get_active_memory() < before - 512 * 1024
    assert empty.shape == (1, 0) and empty.dtype == dtype
    journal.cancel()
