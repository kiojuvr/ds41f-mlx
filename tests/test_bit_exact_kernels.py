"""#4360 selected seams. Requires the isolated, overlay-built MTP oMLX."""
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
gemv = pytest.importorskip('omlx.patches.deepseek_v41.gemv')
from omlx.patches.deepseek_v41 import hyper_connection, quantization
from ds41f_mlx.model_execution import hyper_connection as baseline_hc
from ds41f_mlx.model_execution import quantization as baseline_quant


def exact(a, b):
    # Compare representation, not allclose (including signed zero).
    np.testing.assert_array_equal(np.asarray(a.view(mx.uint8)), np.asarray(b.view(mx.uint8)))


@pytest.mark.parametrize('dtype', [mx.float32, mx.float16, mx.bfloat16])
def test_pack_boundaries_and_strides(dtype):
    codes = np.asarray(mx.from_fp8(mx.arange(127, dtype=mx.uint8), dtype=mx.float32))
    mid = (codes[:-1] + codes[1:]) / 2
    edges = np.concatenate([mid, np.nextafter(mid, -np.inf), np.nextafter(mid, np.inf)])
    rows = np.zeros((len(edges) * 2 + 2, 32), np.float32)
    rows[:2 * len(edges), 0] = np.concatenate([edges, -edges])
    rows[:2 * len(edges), -1] = 448
    rows[-1] = -0.0
    exponents = [-10, 0, 6] if dtype == mx.float16 else [-110, -10, 0, 10, 110]
    for e in exponents:
        x = mx.array(rows * np.float32(2.0**e)).astype(dtype)
        exact(quantization.pack_activation(x), baseline_quant.pack_activation(x))
    mx.random.seed(4360)
    for n in (1, 4, 5, 64, 65, 129):
        x = mx.random.normal((1, n * 2, 512)).astype(dtype)[:, ::2]
        exact(quantization.pack_activation(x), baseline_quant.pack_activation(x))
    for bits, group, e4m3 in ((4, 32, False), (4, 16, True), (8, 16, False)):
        exact(quantization.pack_activation(x, bits, group, e4m3),
              baseline_quant.pack_activation(x, bits, group, e4m3))
    exact(quantization.pack_activation(mx.zeros((1, 0, 512), dtype)),
          baseline_quant.pack_activation(mx.zeros((1, 0, 512), dtype)))


@pytest.mark.parametrize('rows', [1, 4, 5, 63, 64, 65, 129])
@pytest.mark.parametrize('iters', [1, 3, 20])
def test_sinkhorn_serial_order(rows, iters):
    mx.random.seed(rows)
    for x in (mx.softmax(mx.random.normal((1, rows, 4, 4)), axis=-1),
              mx.zeros((1, rows, 4, 4)), mx.full((1, rows, 4, 4), 1e-30)):
        exact(hyper_connection.sinkhorn(x, 1e-6, iters), baseline_hc.sinkhorn(x, 1e-6, iters))


@pytest.mark.parametrize('n,k', [(1280, 5120), (512, 5120), (5120, 2304), (4096, 1280), (32768, 1280)])
@pytest.mark.parametrize('dtype', [mx.bfloat16, mx.float16])
def test_gemv_pinned_mlx(n, k, dtype):
    mx.random.seed(n + k)
    w, s = mx.quantize(mx.random.normal((n, k)) * .02, group_size=32, bits=8, mode='mxfp8')
    for rows in (1, 2, 3, 4, 5, 6):
        x = mx.random.normal((1, rows, k)).astype(dtype)
        y = gemv.mxfp8_gemv(x, w, s)
        if rows not in (1, 4, 5) or rows == 1 and (n > 8192 or k < 4096):
            assert y is None
        else:
            exact(y, mx.quantized_matmul(x, w, s, group_size=32, bits=8, mode='mxfp8'))
    assert gemv.mxfp8_gemv(mx.zeros((5, 1, k), dtype), w, s) is None


@pytest.mark.parametrize('dtype', [mx.bfloat16, mx.float16])
def test_gemv_all_fp8_codes_including_reserved(dtype):
    # Ordinary quantize() does not produce reserved codes; source equivalence
    # must cover them too. Pinned fp8.h decodes 0x7f/0xff as +/-480.
    n, k = 256, 4096
    raw = mx.broadcast_to(mx.arange(256, dtype=mx.uint8)[:, None], (n, k))
    w = mx.contiguous(raw).view(mx.uint32)
    scales = mx.full((n, k // 32), 127, mx.uint8)
    for rows in (1, 4, 5):
        x = mx.full((1, rows, k), 1 / 4096, dtype)
        expected = mx.quantized_matmul(x, w, scales, group_size=32, bits=8, mode='mxfp8')
        actual = gemv.mxfp8_gemv(x, w, scales)
        assert mx.all(mx.isfinite(expected)).item()
        exact(actual, expected)


def test_connected_logits_state_verify_rollback(monkeypatch):
    from test_compressed_growth import small_model
    model = small_model()
    model.configure_mtp(True, 3)
    def run():
        cache = model.make_cache()
        outputs = [model(mx.array([[3, 4, 5, 6, 7, 8, 9]]), cache=cache, return_hidden=True)[0]]
        states = []
        for position in range(7, 22):
            if position == 12:
                cache = [c.extract(0) for c in cache]
            if position in (16, 19):
                outputs.append(model(mx.array([[21, 22, 23, 24]]), cache=cache, target_verify=True)[0])
                assert model.mtp_partial_rollback(cache, 1 if position == 16 else 0, 3)
                states.append([[mx.array(v) if v is not None else None for v in c.cache] for c in cache])
            outputs.append(model(mx.array([[3 + position % 20]]), cache=cache, return_hidden=True)[0])
        mx.eval(*outputs, *[v for c in cache for v in c.cache if v is not None])
        states.append([c.cache for c in cache])
        return outputs, states
    actual = run()
    with monkeypatch.context() as p:
        p.setattr(quantization, 'mxfp8_gemv', lambda *args: None)
        # Recompile BOTH callers with independent function identities. Merely
        # changing lang.sinkhorn would silently reuse MLX's candidate graph.
        import ast
        import omlx.patches.deepseek_v41.language as lang
        namespace = dict(vars(lang), sinkhorn=baseline_hc.sinkhorn)
        tree = ast.parse(Path(lang.__file__).read_text())
        for name in ('_hc_mix_weights', '_hc_mixes'):
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            node.decorator_list = []
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<independent-serial-hc>', 'exec'), namespace)
            namespace[name] = mx.compile(namespace[name])
            p.setattr(lang, name, namespace[name])
        expected = run()
    for a, b in zip(actual[0], expected[0]):
        exact(a, b)
    for astate, bstate in zip(actual[1], expected[1]):
        for aa, bb in zip(astate, bstate):
            for a, b in zip(aa, bb):
                assert (a is None) == (b is None)
                if a is not None:
                    exact(a, b)
