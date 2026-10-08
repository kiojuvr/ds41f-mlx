"""Real packed primitives: causal sparse-list geometry, not checkpoint evidence."""
from types import SimpleNamespace
import pytest
from test_m44_target_generation import mx
from test_m46_state_production import config, equal
import ds41f_mlx.model_execution.language as math
from ds41f_mlx.model_execution.cache import DeepseekV41Cache
from ds41f_mlx.model_execution.quantization import QuantizedProjection, causal_block_arithmetic
from ds41f_mlx.runtime.state_production import DecodeStateProducer


@pytest.mark.parametrize('ratio', [1, 2])
@pytest.mark.parametrize('frontier', [3, 127, 255, 511])
@pytest.mark.parametrize('width', range(2, 9))
def test_source_reuse_candidates_match_row_local_attention(ratio, frontier, width):
    mx.random.seed(56)
    c = config(ratio)
    c.window_size = 128
    c.index_topk = 512
    c.candidate_topk_blocks = 16
    c.candidate_block_size = 32
    modules = [math.Attention(c, i) for i in range(2)]
    for module in modules:
        projections = [module.wq_a, module.wq_b, module.wkv, module.wo_b]
        names = ['wq_a', 'wq_b', 'wkv', 'wo_b']
        if hasattr(module, 'indexer'):
            projections += [module.indexer.wq_b]
            names += ['indexer.wq_b']
            if hasattr(module.indexer, 'wk'):
                projections.append(module.indexer.wk)
                names.append('indexer.wk')
        for name, projection in zip(names, projections):
            # Official-like packed projection on an intentionally small model.
            w, scales = mx.quantize(projection.weight.astype(mx.bfloat16),
                                    group_size=32, bits=8, mode='mxfp8')
            q = QuantizedProjection(w, scales, 8, 'mxfp8')
            if '.' in name:
                setattr(module.indexer, name.split('.')[1], q)
            else:
                setattr(module, name, q)
    actual = [DeepseekV41Cache(ratio), DeepseekV41Cache(0)]
    oracle = [DeepseekV41Cache(ratio), DeepseekV41Cache(0)]
    for i in range(2):
        window = math.pack_activation(mx.random.normal((1, min(frontier, 128), c.head_dim)).astype(mx.bfloat16))
        compressed = math.pack_activation(mx.random.normal((1, frontier // ratio, c.head_dim)).astype(mx.bfloat16), 4, 16, True)
        indexed = math.pack_activation(mx.random.normal((1, frontier // ratio, c.index_head_dim)).astype(mx.bfloat16), 4)
        for cache in (actual[i], oracle[i]):
            cache[1] = window
            if i == 0:
                cache[2], cache[3] = compressed, indexed
                if ratio > 1:
                    cache[4] = mx.random.normal((1, frontier % ratio, c.head_dim))
                    cache[5] = mx.random.normal((1, frontier % ratio, c.head_dim))
        if ratio > 1 and i == 0:
            oracle[i][4], oracle[i][5] = actual[i][4], actual[i][5]
    producer = DecodeStateProducer(mx, math)
    inputs = [mx.random.normal((1, width, c.dim)).astype(mx.bfloat16) for _ in range(2)]
    expected = [[], []]
    for row in range(width):
        shared = {}
        for layer in range(2):
            expected[layer].append(producer.attention(modules[layer], inputs[layer][:, row:row + 1],
                                                      oracle[layer], shared, frontier + row))
    shared = {}
    journal = SimpleNamespace(window_write=lambda *args: None, compressor_write=lambda *args: None)
    with causal_block_arithmetic(width):
        for layer in range(2):
            output = producer.attention(modules[layer], inputs[layer], actual[layer], shared, frontier, journal)
            equal(output, mx.concatenate(expected[layer], 1))
            for slot in range(1, 6):
                if actual[layer][slot] is None:
                    assert oracle[layer][slot] is None
                else:
                    equal(actual[layer][slot], oracle[layer][slot])
