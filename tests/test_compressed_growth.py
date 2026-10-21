"""Finite #4365 seams; run with the build-time patched pinned oMLX on PYTHONPATH."""
import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
growth = pytest.importorskip('omlx.patches.deepseek_v41.growth')
from omlx.patches.deepseek_v41.cache import DeepseekV41Cache
from omlx.patches.deepseek_v41.language import LanguageModel
from omlx.patches.deepseek_v41.config import ModelConfig


def equal(a, b):
    assert (a is None) == (b is None)
    if a is not None:
        if a.dtype == mx.bfloat16:
            a, b = a.astype(mx.float32), b.astype(mx.float32)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_long_slots_extract_truncate_replace_and_snapshot():
    cache = DeepseekV41Cache(1)
    cache[0] = mx.array([131072])
    expected = {}
    snapshots = {}
    for slot, width in ((2, 288), (3, 80)):
        value = mx.arange(131072 * width).reshape(1, 131072, width).astype(mx.uint8)
        cache[slot] = value
        expected[slot] = value
        snapshots[slot] = mx.array(value)
    mx.eval(*cache.cache, *snapshots.values())
    for step in range(12):
        cache = cache.extract(0)
        if step == 6:
            # Reject speculative suffix, then overwrite different values.
            for slot in expected:
                growth.truncate(cache, slot, 131074)
                expected[slot] = expected[slot][:, :131074]
        for slot, width in ((2, 288), (3, 80)):
            previous = cache[slot]
            value = mx.full((1, 1, width), step + slot, mx.uint8)
            cache[slot] = growth.append(cache, slot, previous, value, previous.shape[1])
            expected[slot] = mx.concatenate([expected[slot], value], 1)
        cache[0] = mx.array([cache[2].shape[1]])
        mx.eval(*cache.cache)
        for slot in expected:
            equal(cache[slot], expected[slot])
    assert all(cache._ds41_buffers[s]['idle'] is not None for s in expected)
    for slot, width in ((2, 288), (3, 80)):
        equal(snapshots[slot], mx.arange(131072 * width).reshape(1, 131072, width).astype(mx.uint8))
        # from_state/paired restore intentionally does not inherit registry.
        restored = DeepseekV41Cache.from_state([mx.array(v) if v is not None else None for v in cache.cache], cache.meta_state)
        assert not hasattr(restored, '_ds41_buffers')
        value = mx.full((1, 1, width), 99, mx.uint8)
        before = restored[slot]
        restored[slot] = growth.append(restored, slot, before, value, before.shape[1])
        equal(restored[slot], mx.concatenate([before, value], 1))
        # External replacement breaks the chain safely.
        cache[slot] = mx.zeros_like(cache[slot])
        cache[slot] = growth.append(cache, slot, cache[slot], value, cache[slot].shape[1])
        equal(cache[slot][:, :-1], mx.zeros_like(before))


def test_paired_capture_does_not_export_growth_registry():
    from ds41f_mlx.serving.paired_checkpoint import PairedCheckpoint
    from omlx.patches.mlx_lm_mtp.deepseek_v4_dspark import DSparkContextCache
    target = [DeepseekV41Cache(1) for _ in range(40)]
    for cache in target:
        cache[0] = mx.array([16])
        cache[1] = mx.zeros((1, 4, 8), mx.uint8)
        for slot in (2, 3):
            cache[slot] = mx.zeros((1, 15, 8), mx.uint8)
            cache[slot] = growth.append(cache, slot, cache[slot], mx.ones((1, 1, 8), mx.uint8), 15)
    rings = [DSparkContextCache(128) for _ in range(3)]
    for ring in rings:
        ring.offset = 16
        ring.keys = mx.zeros((1, 1, 16, 8))
    images = [dict(start=2, length=4, sha256='image', grid=[1, 1])]
    saved = PairedCheckpoint.capture('identity', target, rings, list(range(16)), mx, images)
    restored, restored_rings, tokens = saved.restore('identity', list(range(18)), mx, images)
    assert tokens == list(range(16))
    assert not any(hasattr(c, '_ds41_buffers') for c in restored)
    for step in range(8):
        for cache in target + restored:
            for slot in (2, 3):
                cache[slot] = growth.append(cache, slot, cache[slot], mx.full((1, 1, 8), step + 2, mx.uint8), 16 + step)
        mx.eval(*[v for c in target + restored for v in c.cache if v is not None])
    again, _, _ = saved.restore('identity', list(range(18)), mx, images)
    for cache in again:
        equal(cache[2][:, -1:], mx.ones((1, 1, 8), mx.uint8))
        assert cache[2].shape[1] == 16
    assert all(r.offset == 16 for r in restored_rings)


def small_model():
    mx.random.seed(341)
    return LanguageModel(ModelConfig(
        vocab_size=64, dim=32, moe_inter_dim=32, n_layers=5,
        n_heads=2, q_lora_rank=32, head_dim=32, rope_head_dim=8,
        o_groups=2, o_lora_rank=16, index_n_heads=8, index_head_dim=32,
        index_topk=3, n_routed_experts=4, n_activated_experts=2,
        window_size=4, dtype='bf16', expert_dtype=None, max_seq_len=256,
        max_batch_size=1, hc_sinkhorn_iters=3, candidate_source_layer=3,
        candidate_block_size=2, candidate_topk_blocks=2,
        index_source_layers=(1, 3, 4), preserve_mtp=True, n_mtp_layers=3,
        dspark_block_size=3, dspark_noise_token_id=2,
        dspark_target_layer_ids=(2, 3, 4), dspark_n_routed_experts=2,
        dspark_n_activated_experts=1, dspark_markov_rank=32,
        compress_ratios=(0, 2, 2, 1, 1, 0, 0, 0)))


@pytest.mark.parametrize('variant', ['flash', 'flash-1017'])
def test_logits_verify_rollback_continuation_equal(monkeypatch, variant):
    from ds41f_mlx import execution_identity
    from ds41f_mlx.mtp_identity import VARIANTS
    monkeypatch.setattr(execution_identity, '_admitted_variant', VARIANTS[variant])
    model = small_model()
    # Match the released BF16 activation boundary rather than nn's FP32 default.
    model.set_dtype(mx.bfloat16)
    model.configure_mtp(True, 3)

    def run():
        cache = model.make_cache()
        outputs = [model(mx.array([[3, 4, 5, 6, 7, 8, 9]]), cache=cache, return_hidden=True)[0]]
        states = []
        for position in range(7, 35):
            if position in (12, 24):
                cache = [c.extract(0) for c in cache]
            if position in (16, 25):
                outputs.append(model(mx.array([[21, 22, 23, 24]]), cache=cache, target_verify=True)[0])
                assert model.mtp_partial_rollback(cache, 1 if position == 16 else 0, 3)
                states.append([[mx.array(v) if v is not None else None for v in c.cache] for c in cache])
            outputs.append(model(mx.array([[3 + position % 20]]), cache=cache, return_hidden=True)[0])
        mx.eval(*outputs, *[v for c in cache for v in c.cache if v is not None])
        return outputs, states, cache

    actual = run()
    # Ratio=2 no-publication tokens must not discard the index growth chain.
    assert actual[2][1]._ds41_buffers[3]['idle'] is not None
    assert all(np.isfinite(np.asarray(v)).all() for v in actual[0])
    with monkeypatch.context() as patch:
        patch.setattr(growth, 'append', lambda cache, slot, previous, values, start: mx.concatenate([previous[:, :start], values], 1))
        expected = run()
    for a, b in zip(actual[0], expected[0]):
        equal(a, b)
    for aa, bb in zip(actual[1], expected[1]):
        for a, b in zip(aa, bb):
            for x, y in zip(a, b):
                equal(x, y)
    for a, b in zip(actual[2], expected[2]):
        assert a.size() == b.size()
        for x, y in zip(a.cache, b.cache):
            equal(x, y)
