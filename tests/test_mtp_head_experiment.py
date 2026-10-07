"""Small numerical check of the qualification-only kernel, not runtime admission."""
import pytest

mx = pytest.importorskip('mlx.core')


@pytest.mark.parametrize('rows', [1, 2, 3, 4, 5])
def test_head_row_reuse_preserves_fp32_output(rows):
    from omlx.patches.deepseek_v41.head import project_logits
    from tools.bench_mtp_head import make_prototype
    weight = (mx.sin(mx.arange(4096*256, dtype=mx.float32)).reshape(4096,256)*.02).astype(mx.bfloat16)
    x = mx.cos(mx.arange(rows*256, dtype=mx.float32)).reshape(rows,256).astype(mx.bfloat16)
    expected = project_logits(x, weight)
    actual = make_prototype(mx)(x, weight)
    mx.eval(expected, actual)
    assert bool(mx.all(expected == actual).item())
    assert actual.dtype == mx.float32
