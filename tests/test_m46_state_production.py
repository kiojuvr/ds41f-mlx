"""Numerical/state lifecycle at the owned producer seam, real MLX packed arrays."""
import pytest
from test_m44_target_generation import mx, session
from ds41f_mlx.runtime.state_production import DecodeStateProducer
from omlx.patches.deepseek_v41.config import ModelConfig
from omlx.patches.deepseek_v41.language import Block, Attention, Compressor, Indexer
from omlx.patches.deepseek_v41.cache import DeepseekV41Cache


def config(ratio):
    return ModelConfig(dim=128, n_layers=2, n_mtp_layers=0, n_heads=2,
        q_lora_rank=32, head_dim=128, rope_head_dim=32, o_groups=1, o_lora_rank=32,
        window_size=4, compress_ratios=(ratio, ratio),
        kv_source_layers=(0,) if ratio else (), index_source_layers=(0, 1) if ratio else (),
        index_n_heads=2, index_head_dim=128, index_topk=4,
        candidate_source_layer=0 if ratio else -1, candidate_topk_blocks=2,
        candidate_block_size=2, hc_mult=2, moe_inter_dim=32,
        n_routed_experts=2, n_shared_experts=1, n_activated_experts=1,
        expert_dtype=None)


def equal(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert bool(mx.all(a == b).item())


@pytest.mark.parametrize('ratio', [0, 1, 2, 4])
def test_block_source_reuse_candidate_tail_lifecycle_matches_m45(ratio, monkeypatch):
    c = config(ratio)
    layers = [Block(c, i) for i in range(2)]
    control = [DeepseekV41Cache(ratio), DeepseekV41Cache(0)]
    owned = [DeepseekV41Cache(ratio), DeepseekV41Cache(0)]
    originals = tuple(map(id, owned))
    producer = DecodeStateProducer(mx)
    # Every prefix crosses compressor completion, empty/nonempty candidates,
    # chronological top-k, and window eviction. No cloned executable state.
    for start in range(9):
        h = mx.random.normal((1, 1, c.hc_mult, c.dim)).astype(mx.bfloat16)
        pre = mx.ones((1, 1, c.hc_mult), mx.float32)
        ch, cp, oh, op = h, pre, h, pre
        expected_shared, shared = {}, {}
        for i, layer in enumerate(layers):
            ch, cp = layer(ch, cp, control[i], expected_shared, start, None)
            owned[i]._p6_append_pending = True
            oh, op = producer.block(layer, oh, op, owned[i], shared, start)
            equal(ch, oh); equal(cp, op)
            for slot in range(1, 6):
                a, b = control[i][slot], owned[i][slot]
                if a is None or b is None:
                    assert a is b
                else:
                    equal(a, b)
            for key in expected_shared:
                equal(expected_shared[key], shared[key])
            assert owned[i][1].shape[1] == min(start+1, c.window_size)
        if ratio:
            assert owned[0][2].shape[1] == (start+1)//ratio
            assert owned[0][3].shape[1] == (start+1)//ratio
            if ratio > 1:
                assert owned[0][4].shape[1] == owned[0][5].shape[1] == (start+1)%ratio
            assert owned[1][2] is None and owned[1][3] is None
        assert tuple(map(id, owned)) == originals
    # Prove subordinate state calls are unnecessary, not just numerically equal.
    def forbidden(*a, **kw):
        raise AssertionError('donor state producer called')
    for cls in (Block, Attention, Compressor, Indexer):
        monkeypatch.setattr(cls, '__call__', forbidden)
    shared = {}
    for i, layer in enumerate(layers):
        oh, op = producer.block(layer, oh, op, owned[i], shared, 9)
    mx.eval(oh, op, *(v for cache in owned for v in cache.cache if v is not None))


def test_producer_requires_pending_and_rejects_verification_state():
    producer = DecodeStateProducer(mx)
    c = config(2); layer = Block(c, 0); cache = DeepseekV41Cache(2)
    with pytest.raises(RuntimeError, match='pending'):
        producer.block(layer, None, None, cache, {}, 0)
    cache._p6_append_pending = True
    cache._mtp_verify_state = {}
    with pytest.raises(RuntimeError, match='verification'):
        producer.block(layer, None, None, cache, {}, 0)
    assert all(v is None for v in cache.cache)


@pytest.mark.parametrize('phase', ['before', 'window', 'tails', 'index', 'compressed', 'after'])
def test_subordinate_mutation_failure_burns_all_original_objects(phase, monkeypatch):
    model, gen = session()
    # Exercise real packed state-producing block within the 40-layer transaction;
    # remaining layers are explicit lifecycle doubles, never a shadow cache.
    ratio = 1 if phase in ('compressed', 'after') else 2
    c = config(ratio)
    model.layers[0] = Block(c, 0)
    producer = gen.target_forward.producer
    cache = gen.initial_cache
    objects = tuple(map(id, cache))
    old_history = gen.current_token_history()
    def fail(*a, **kw):
        raise RuntimeError('injected producer failure')
    if phase == 'before':
        monkeypatch.setattr(producer, 'block', fail)
    elif phase == 'window':
        monkeypatch.setattr(producer, 'compressor', fail)
    elif phase == 'tails':
        monkeypatch.setattr(producer, 'indexer', fail)
    elif phase == 'index':
        original = producer.indexer
        def index(*a, **kw):
            original(*a, **kw)
            fail()
        monkeypatch.setattr(producer, 'indexer', index)
    elif phase == 'compressed':
        monkeypatch.setattr(producer.math, 'packed_sparse_attention', fail)
    else:
        original = producer.block
        def block(*a, **kw):
            original(*a, **kw)
            fail()
        monkeypatch.setattr(producer, 'block', block)
    # Real input width, source packed slots and compressor rem match frontier 2.
    model.embed = lambda ids: mx.ones((*ids.shape, 128), mx.bfloat16)
    model._config.hc_mult = 2
    cache[0][1] = producer.math.pack_activation(mx.ones((1, 2, 128), mx.bfloat16))
    cache[0][2] = producer.math.pack_activation(mx.ones((1, 2//ratio, 128), mx.bfloat16), 4, 16, True)
    cache[0][3] = producer.math.pack_activation(mx.ones((1, 2//ratio, 128), mx.bfloat16), 4)
    try:
        with pytest.raises(RuntimeError, match='injected producer'):
            gen.start(3)
        assert tuple(map(id, cache)) == objects
        assert all(item._p6_append_invalid and item._p6_append_failed for item in cache)
        assert all(item.size() == 2 for item in cache)
        assert gen.current_token_history() == old_history
        if phase != 'before':
            assert cache[0][1].shape[1] == 3
        if phase in ('tails', 'index'):
            assert cache[0][4].shape[1] == cache[0][5].shape[1] == 1
        if phase in ('compressed', 'after'):
            assert cache[0][2].shape[1] == cache[0][3].shape[1] == 3
        with pytest.raises(RuntimeError):
            gen.extract_final_state()
    finally:
        gen.close()
