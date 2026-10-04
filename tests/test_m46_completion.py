"""Owned structural completion and post-mutation failure gate."""
import pytest
from test_m44_target_generation import mx, session
from test_m46_state_production import config
from ds41f_mlx.runtime.state_production import DecodeStateProducer
from omlx.patches.deepseek_v41.cache import DeepseekV41Cache


@pytest.mark.parametrize('layer,slot', [(0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3),
                                      (0, 4), (0, 5), (1, 4), (1, 5), (0, 6), (1, 6)])
def test_completion_rejects_wrong_lifecycle(layer, slot):
    c = config(2)
    cache = [DeepseekV41Cache(2), DeepseekV41Cache(0)]
    for i, item in enumerate(cache):
        for s in range(1, 7):
            length = 4 if s == 1 else (4 if i == 0 and s in (2, 3) else
                                      1 if i == 0 and s in (4, 5) else 0)
            item[s] = mx.zeros((1, length, 128)) if s != 6 else mx.zeros((1, 0), mx.int64)
    # Call original static contract: these are not lifecycle-double configs.
    DecodeStateProducer.validate_completion(cache, c, 9)
    value = cache[layer][slot]
    cache[layer][slot] = mx.zeros((1, value.shape[1]+1, *value.shape[2:]), value.dtype)
    with pytest.raises(RuntimeError, match=f'layer {layer} slot {slot}'):
        DecodeStateProducer.validate_completion(cache, c, 9)


def test_replaced_object_burns_original_and_replacement_aliases(monkeypatch):
    _, gen = session()
    gen.start(3)
    cache = gen._cache
    originals = tuple(cache)
    history = gen.current_token_history()
    forward = gen.target_forward.forward
    def replace(*a):
        logits = forward(*a)
        cache[0] = DeepseekV41Cache(1)
        cache[0][0] = mx.array([len(history)+1], mx.int32)
        return logits
    monkeypatch.setattr(gen.target_forward, 'forward', replace)
    try:
        with pytest.raises(RuntimeError, match='objects replaced'):
            gen.next_token()
        assert all(c._p6_append_invalid for c in (*originals, *cache))
        assert gen.current_token_history() == history
        with pytest.raises(RuntimeError):
            gen.extract_final_state()
    finally:
        gen.close()


def test_metadata_advance_stays_under_owned_barrier(monkeypatch):
    _, gen = session()
    cache = gen.initial_cache
    for item in cache:
        item.lengths = mx.array([10], mx.int32)
        item.left_padding = mx.array([0], mx.int32)
    def forbidden(*a):
        raise AssertionError('external admission-state advance')
    monkeypatch.setattr(DeepseekV41Cache, 'advance', forbidden)
    try:
        gen.start(3)
        assert all(c.lengths.item() == 9 and c.left_padding.item() == -1 for c in cache)
        assert not any(c._p6_append_pending for c in cache)
    finally:
        gen.close()


def test_completion_failure_after_sync_burns_frontier_without_history(monkeypatch):
    _, gen = session()
    gen.start(3)
    cache = gen._cache
    history = gen.current_token_history()
    def fail(*a):
        raise RuntimeError('injected lifecycle completion')
    monkeypatch.setattr(gen.target_forward.producer, 'validate_completion', fail)
    try:
        with pytest.raises(RuntimeError, match='lifecycle completion'):
            gen.next_token()
        assert all(item.size() == len(history)+1 for item in cache)
        assert all(item._p6_append_invalid for item in cache)
        assert gen.current_token_history() == history
        with pytest.raises(RuntimeError):
            gen.extract_final_state()
    finally:
        gen.close()
