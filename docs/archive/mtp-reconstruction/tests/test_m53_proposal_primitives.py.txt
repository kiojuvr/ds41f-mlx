"""Reduced real-MLX receipt/ring qualification; not checkpoint qualification."""
from types import SimpleNamespace
import numpy as np
import pytest
from test_m52_generation import session, compare
from test_m44_target_generation import mx
from ds41f_mlx.runtime.hidden_taps import CommittedTapReceipt, PrefillTapCapture
from ds41f_mlx.model_execution.proposal_ring import DSparkContextCache


class Child:
    def __init__(self):
        self.mx = mx
        self.config = SimpleNamespace(window_size=4, dim=1, dspark_target_layer_ids=(37, 38, 39))
        self.active = True
        self.receipts = set()
    def assert_active(self):
        if not self.active:
            raise RuntimeError('retired')


def tapped_session(**kwargs):
    gen, target, cache = session(**kwargs)
    child = Child()
    target.proposal_child = child
    forward = target.forward
    def tapped(token, *args, **kwargs):
        logits = forward(token, *args, **kwargs)
        target.tap_rows = (mx.broadcast_to(token.reshape(1, -1, 1),
                                           (1, token.shape[0], 3)).astype(mx.float32)
                           if target.proposal_child is not None else None)
        return logits
    target.forward = tapped
    return gen, target, child, cache


@pytest.mark.parametrize('accepted', range(6))
def test_committed_receipt_excludes_rejected_tail(accepted):
    gen, target, child, cache = tapped_session()
    oracle, _, _ = session()
    drafts = list(range(12, 17))
    if accepted < 5:
        drafts[accepted] = 99
    try:
        start = gen.token_frontier
        result = gen.speculative_cycle(drafts)
        receipt = gen.take_tap_receipt()
        assert receipt.start == start and receipt.end == start + accepted + 1
        rows = receipt.take(child, gen, start)
        assert rows[:, :, 0].tolist() == [list(range(11, 12 + accepted))]
        assert receipt.retired and not child.receipts
        assert not target.children[-1].tap_rows
        with pytest.raises(RuntimeError):
            receipt.take(child, gen, start)
        oracle.generate(accepted + 1)
        compare(gen, oracle)
        assert result['consumed_positions'] == accepted + 1
    finally:
        gen.close(); oracle.close()


def test_cancel_no_receipt_or_draw():
    gen, target, child, _ = tapped_session()
    try:
        armed = [False]
        gen.speculative_cycle([12, 13, 99], cancelled=lambda: armed[0],
            _fault=lambda phase: armed.__setitem__(0, True) if phase == 'materialized' else None)
        assert not child.receipts and target.children[-1].tap_rows == []
        with pytest.raises(RuntimeError):
            gen.take_tap_receipt()
    finally:
        gen.close()


@pytest.mark.parametrize('action', ['close', 'cancel', 'terminal', 'failure'])
def test_receipts_retire_at_owner_boundary(action):
    gen, target, child, _ = tapped_session()
    try:
        gen.speculative_cycle([12])
        receipt = gen._tap_receipt
        if action == 'failure':
            with pytest.raises(RuntimeError):
                gen.speculative_cycle([14], _fault=lambda p: (_ for _ in ()).throw(RuntimeError('fault'))
                                      if p == 'tentative' else None)
        elif action == 'terminal':
            gen.max_tokens = len(gen.generated_tokens) + 1
            gen.speculative_cycle([14])
        else:
            getattr(gen, action)()
        assert receipt.retired and not child.receipts
    finally:
        gen.close()


def test_physical_ring_absolute_order_and_retire():
    ring = DSparkContextCache(4)
    ring.append(mx.arange(10, 14).reshape(1, 1, 4, 1), start_offset=10)
    assert ring.keys.reshape(-1).tolist() == [12, 13, 10, 11]
    ring.append(mx.array([14, 15, 16]).reshape(1, 1, 3, 1), start_offset=14)
    assert ring.offset == 17
    assert ring.keys.reshape(-1).tolist() == [16, 13, 14, 15]
    with pytest.raises(ValueError):
        ring.append(mx.zeros((1, 1, 1, 1)), start_offset=18)
    ring.retire()
    assert ring.keys is None


def test_prefill_taps_bounded_same_forward_layer_order():
    child, owner = Child(), object()
    capture = PrefillTapCapture(child)
    for layer in child.config.dspark_target_layer_ids:
        for start in (0, 3, 6):
            capture.capture(layer, start, mx.full((1, 3, 2, 1), layer, mx.float32))
    receipt = capture.commit(owner, 9)
    assert receipt.start == 5 and receipt.end == 9
    assert receipt.rows.shape == (1, 4, 3)
    assert receipt.rows[0, 0].tolist() == [37, 38, 39]
    receipt.retire()
    assert not child.receipts and not capture.parts


def test_receipt_rejects_other_child_owner_or_frontier():
    child, owner = Child(), object()
    receipt = CommittedTapReceipt(child, owner, 10, mx.ones((1, 2, 3)))
    for wrong_child, wrong_owner, wrong_start in ((Child(), owner, 10),
        (child, object(), 10), (child, owner, 11)):
        with pytest.raises(RuntimeError):
            receipt.take(wrong_child, wrong_owner, wrong_start)
        assert not receipt.retired
    child.active = False
    with pytest.raises(RuntimeError):
        receipt.take(child, owner, 10)
    receipt.retire()
    assert not child.receipts


def test_receipt_never_changes_ambient_rng():
    child, owner = Child(), object()
    mx.random.seed(53)
    expected = mx.random.uniform(shape=(8,)); mx.eval(expected)
    mx.random.seed(53)
    for _ in range(20):
        receipt = CommittedTapReceipt(child, owner, 0, mx.ones((1, 2, 3)))
        receipt.take(child, owner, 0)
    actual = mx.random.uniform(shape=(8,)); mx.eval(actual)
    np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))
