"""Existing batched primitive geometry; no checkpoint or serving claims."""
import numpy as np
import pytest
from test_m44_target_generation import mx
from ds41f_mlx.model_execution.quantization import (
    QuantizedProjection, causal_block_arithmetic, causal_matmul, causal_width,
)
from ds41f_mlx.model_execution.head import project_logits


@pytest.mark.parametrize('width', range(2, 9))
@pytest.mark.parametrize('dims,outputs', [(20480, 24), (5120, 256)])
def test_dense_batch_gemv_preserves_row_reduction(width, dims, outputs):
    mx.random.seed(56)
    x = mx.random.normal((1, width, dims)).astype(mx.bfloat16).astype(mx.float32)
    weight = mx.random.normal((outputs, dims))
    oracle = mx.concatenate([x[:, row:row + 1] @ weight.T for row in range(width)], 1)
    with causal_block_arithmetic(width):
        actual = causal_matmul(x, weight)
    assert bool(mx.array_equal(actual, oracle).item())
    assert causal_width() == 1


@pytest.mark.parametrize('width', range(2, 9))
@pytest.mark.parametrize('mode,bits,group', [('mxfp8', 8, 32), ('mxfp4', 4, 32), ('affine', 4, 64)])
def test_quantized_projection_single_invocation_row_geometry(width, mode, bits, group):
    mx.random.seed(56)
    x = mx.random.normal((1, width, 512)).astype(mx.bfloat16)
    weight = mx.random.normal((256, 512)).astype(mx.bfloat16)
    packed = mx.quantize(weight, group_size=group, bits=bits, mode=mode)
    module = QuantizedProjection(packed[0], packed[1], bits, mode,
                                 biases=packed[2] if len(packed) > 2 else None,
                                 group_size=group)
    oracle = mx.concatenate([module(x[:, row:row + 1]) for row in range(width)], 1)
    with causal_block_arithmetic(width):
        actual = module(x)
    assert bool(mx.array_equal(actual, oracle).item())


@pytest.mark.parametrize('width', range(2, 9))
def test_existing_head_shared_dispatch_matches_every_row(width):
    mx.random.seed(56)
    x = mx.random.normal((1, width, 256)).astype(mx.bfloat16)
    weight = mx.random.normal((4096, 256)).astype(mx.bfloat16)
    oracle = mx.concatenate([project_logits(x[:, row:row + 1], weight)
                             for row in range(width)], 1)
    with causal_block_arithmetic(width):
        actual = project_logits(x, weight)
    assert bool(mx.array_equal(actual, oracle).item())


@pytest.mark.parametrize('width', range(2, 9))
def test_compiled_hc_keeps_normalization_and_reduction(width):
    from ds41f_mlx.model_execution.language import _hc_mixes, _causal_hc_mixes
    mx.random.seed(56)
    x = mx.random.normal((1, width, 4, 5120)).astype(mx.bfloat16)
    fn = mx.random.normal((24, 20480))
    args = (fn, mx.array([.01, .02, .03]), mx.random.normal((24,)),
            4, 1e-6, 1e-6, 20)
    rows = [_hc_mixes(x[:, row:row + 1], *args) for row in range(width)]
    actual = _causal_hc_mixes(x, *args)
    for index in range(3):
        oracle = mx.concatenate([row[index] for row in rows], 1)
        assert bool(mx.array_equal(actual[index], oracle).item())


def test_numerical_scope_is_worker_local_and_fault_resets():
    from concurrent.futures import ThreadPoolExecutor
    assert causal_width() == 1
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError, match='fault'):
            with causal_block_arithmetic(4):
                assert causal_width() == 4
                assert pool.submit(causal_width).result() == 1
                raise RuntimeError('fault')
    assert causal_width() == 1
    with pytest.raises(ValueError):
        with causal_block_arithmetic(9):
            pass
