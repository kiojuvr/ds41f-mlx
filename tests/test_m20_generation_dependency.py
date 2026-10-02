"""Cheap real-MLX dependency seam tests; never loads checkpoint weights.

Run in a separate process per dependency, with DS41F_OMLX_PATH and PYTHONPATH
pointing at the explicit versioned checkout. Not real-model qualification.
"""
import importlib
import os
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
ROOT = Path(os.environ.get("DS41F_OMLX_PATH", str(Path.home() / "omlx-0.7.0.release")))
if not ROOT.exists():
    pytest.skip("explicit oMLX checkout unavailable", allow_module_level=True)
import sys
sys.path.insert(0, str(ROOT))
from omlx.patches.deepseek_v41.cache import DeepseekV41Cache
from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
from ds41f_mlx.runtime.omlx_generation import OMLXGenerationSession


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.fail = False

    def __call__(self, ids, cache):
        if self.fail:
            raise RuntimeError("injected forward failure")
        self.calls.append(ids.tolist())
        for c in cache:
            c[0] = c[0] + ids.shape[1]
            c[6] = mx.concatenate([c[6], ids.astype(mx.int64)], axis=1)
        return mx.broadcast_to(mx.array([0., 1., 9., 0.]), (*ids.shape, 4))


def session(max_tokens=3, stop=()):
    c = DeepseekV41Cache(1)
    c[0] = mx.array([2], mx.int32)
    c[6] = mx.array([[0, 1]], mx.int64)
    model = TinyModel()
    gen = OMLXGenerationSession.from_prefilled_cache(
        model, [c], [0, 1], OMLXDecodeConfig(omlx_path=ROOT, preserve_mtp=False,
                                         stop_token_ids=stop), max_tokens=max_tokens)
    return model, gen


def assert_empty(gen):
    bg = gen._bg
    assert not bg._generation_batch.uids
    assert not bg._prompt_batch.uids
    assert not bg._unprocessed_sequences


def test_zero_replay_cancel_removes_owner_and_continues():
    model, gen = session()
    gen.start(3)
    assert model.calls == [[[3]]]
    assert gen.initial_cache == [] and gen.prompt_replay_count == 0
    assert gen.next_token().token == 2
    cache, history = gen.extract_final_state("cancelled")
    assert_empty(gen)
    assert history == [0, 1, 3, 2]
    assert cache[0].size() == len(history)
    assert cache[0][6].tolist() == [history]
    gen.close()
    assert gen._final_cache is None
    next_gen = OMLXGenerationSession.from_prefilled_cache(
        model, cache, history, OMLXDecodeConfig(omlx_path=ROOT, preserve_mtp=False), max_tokens=1)
    try:
        next_gen.start(3)
        assert next_gen.next_token().finish_reason == "length"
        final_cache, final_history = next_gen.extract_final_state()
        assert final_cache[0].size() == len(final_history) == 6
        assert_empty(next_gen)
    finally:
        next_gen.close()


@pytest.mark.parametrize("stop,reason", [((), "length"), ((2,), "stop")])
def test_natural_finish_history_and_cleanup(stop, reason):
    _, gen = session(max_tokens=1, stop=stop)
    try:
        gen.start(3)
        rep = gen.next_token()
        assert rep.token == 2 and rep.finish_reason == reason
        cache, history = gen.extract_final_state()
        assert history == [0, 1, 3, 2]
        assert cache[0].size() == 4
        assert_empty(gen)
    finally:
        gen.close()


def test_failed_bootstrap_close_drains_pending_owner():
    model, gen = session()
    model.fail = True
    with pytest.raises(RuntimeError, match="injected"):
        gen.start(3)
    gen.close()
    assert_empty(gen)
    gen.close()


def test_deepseek_native_array_abi_and_pack_execution():
    fast = importlib.import_module("omlx.custom_kernels.glm_moe_dsa.fast")
    assert fast.has_symbol("deepseek_v41_packed_attention")
    quant = importlib.import_module("omlx.patches.deepseek_v41.quantization")
    packed = quant.pack_activation(mx.ones((1, 2, 128), mx.bfloat16), bits=4,
                                   group_size=16, e4m3_scale=True)
    mx.eval(packed)
    assert packed.shape[0:2] == (1, 2)
