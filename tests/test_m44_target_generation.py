"""Owned execution lifecycle on real MLX and temporary packed cache objects."""
import importlib
import os
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
from ds41f_mlx.config import DEFAULT_OMLX
ROOT = Path(os.environ.get("DS41F_OMLX_PATH", str(DEFAULT_OMLX)))
if not ROOT.exists():
    pytest.skip("explicit oMLX checkout unavailable", allow_module_level=True)
import sys
sys.path.insert(0, str(ROOT))
from omlx.patches.deepseek_v41.cache import DeepseekV41Cache
from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
from ds41f_mlx.runtime.target_generation import TargetGenerationSession as OMLXGenerationSession


from types import SimpleNamespace


class TinyBlock(nn.Module):
    def __init__(self, owner, index):
        super().__init__()
        # Avoid an nn.Module ownership cycle.
        object.__setattr__(self, 'owner', owner)
        self.index = index

    def __call__(self, h, pre, cache, shared, start, image_mask):
        if self.owner.fail and self.index == self.owner.fail_layer:
            raise RuntimeError('injected forward failure')
        return h, pre


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.fail = False
        self.fail_layer = 0
        self._config = SimpleNamespace(_state_production_test_double=True,
            n_layers=40, hc_mult=1, engram_layer_ids=(),
            compress_ratios=(1,)*40, kv_source_layers=tuple(range(40)),
            index_head_dim=128, head_dim=128)
        self.layers = [TinyBlock(self, i) for i in range(40)]
        self.head = nn.Linear(4, 4, bias=False)
        self.head.weight = mx.eye(4)

    def __call__(self, *args, **kwargs):
        pytest.fail('external LanguageModel target call')

    def embed(self, ids):
        self.calls.append(ids.tolist())
        return mx.broadcast_to(mx.array([0., 1., 9., 0.]), (*ids.shape, 4))

    def norm(self, h):
        return h

    def _hasher(self, ids, history, image_mask):
        return None, mx.concatenate([history, ids.astype(mx.int64)], axis=1)


def session(max_tokens=3, stop=()):
    cache = [DeepseekV41Cache(1) for _ in range(40)]
    for c in cache:
        c[0] = mx.array([2], mx.int32)
        c[6] = mx.array([[0, 1]], mx.int64)
    model = TinyModel()
    gen = OMLXGenerationSession.from_prefilled_cache(
        model, cache, [0, 1], OMLXDecodeConfig(omlx_path=ROOT, preserve_mtp=False,
                                         stop_token_ids=stop), max_tokens=max_tokens)
    return model, gen


def assert_empty(gen):
    assert gen._cache is None
    assert gen._pending is None
    assert not hasattr(gen, '_bg')


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


def test_qualified_packed_cache_representation_execution():
    # M43/R1 delivery does not promise every optional donor fast symbol.
    # The moved boundary requires the qualified packed representation, not
    # a new kernel-presence admission rule inherited from a legacy host probe.
    quant = importlib.import_module("omlx.patches.deepseek_v41.quantization")
    packed = quant.pack_activation(mx.ones((1, 2, 128), mx.bfloat16), bits=4,
                                   group_size=16, e4m3_scale=True)
    mx.eval(packed)
    assert packed.shape[0:2] == (1, 2)


def test_cancel_bootstrap_discards_unconsumed_sample():
    model, gen = session()
    gen.initial_cache[0]._p6_append_sealed = True
    gen.initial_cache[0]._p6_owner_token = 123
    gen.start(3)
    assert gen._cache[0]._p6_append_sealed
    original = gen._cache
    cache, history = gen.extract_final_state('cancelled')
    assert cache is original
    assert not hasattr(cache[0], '_p6_append_sealed')
    assert not hasattr(cache[0], '_p6_owner_token')
    assert history == [0, 1, 3]
    assert model.calls == [[[3]]]
    assert cache[0].size() == 3
    assert gen.next_token() is None
    gen.close()


def test_step_failure_burns_authority_no_reconstruction():
    model, gen = session()
    gen.start(3)
    model.fail = True
    stale_cache = gen._cache
    with pytest.raises(RuntimeError, match='injected'):
        gen.next_token()
    assert stale_cache[0]._p6_append_failed
    with pytest.raises(RuntimeError, match='not sealed/admissible'):
        OMLXGenerationSession.from_prefilled_cache(model, stale_cache, gen.current_token_history())
    with pytest.raises(RuntimeError, match='failed generation'):
        gen.extract_final_state()
    with pytest.raises(RuntimeError, match='failed generation'):
        gen.next_token()
    with pytest.raises(RuntimeError, match='already attempted'):
        gen.start(3)
    gen.close()
    assert_empty(gen)


def test_terminal_only_continuation_failure_closes_parent_lease():
    from ds41f_mlx.runtime.continuation_session import M8LiveContinuationSession
    model, gen = session()
    cache = gen.initial_cache
    cfg = gen.config
    gen.close()
    parent = M8LiveContinuationSession(model, cache, [0, 1], cfg)
    model.fail = True
    with pytest.raises(RuntimeError, match='injected'):
        parent.begin_turn_from_suffix([3])
    assert parent.closed and parent.live_cache == [] and parent.generation is None
    assert cache[0]._p6_append_failed


def test_idle_frontier_failure_burns_cache_and_restores_wired_limit(monkeypatch):
    _, gen = session()
    gen.start(3)
    stale = gen._cache
    stale[0][0] = mx.array([99], mx.int32)
    restored = []
    previous = gen._old_wired_limit
    original = mx.set_wired_limit
    def set_limit(value):
        restored.append(value)
        return original(value)
    monkeypatch.setattr(mx, 'set_wired_limit', set_limit)
    with pytest.raises(RuntimeError, match='idle transfer cache frontier'):
        gen.extract_final_state()
    assert stale[0]._p6_append_failed
    gen.close()
    if previous is not None:
        assert restored == [previous]
    gen.close()
    assert gen._old_wired_limit is None


def test_sampler_receives_normalized_logprobs_and_no_scheduler(monkeypatch):
    legacy = importlib.import_module('mlx_lm.generate')
    monkeypatch.setattr(legacy, 'BatchGenerator', lambda *a, **kw: pytest.fail('external scheduler'))
    seen = []
    model, gen = session(max_tokens=1)
    def sampler(logprobs):
        seen.append(logprobs)
        return mx.argmax(logprobs, axis=-1)
    gen.sampler = sampler
    gen.start(3)
    gen.next_token()
    assert len(seen) == 2
    for p in seen:
        assert abs(float(mx.logsumexp(p).item())) < 1e-6
    gen.close()
