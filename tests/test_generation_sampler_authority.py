"""Sampling authority regression, not real-model stochastic qualification."""
from types import SimpleNamespace

import pytest

from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend


def sampler(temperature, top_p):
    # make_sampler has no model/executor dependency. Avoid creating an unused
    # inference executor or importing a checkpoint in this contract test.
    return DeepSeekRecipeRuntimeBackend.make_sampler(
        None, SimpleNamespace(temperature=temperature, top_p=top_p))


def test_stochastic_sampler_is_recognized_by_pinned_mtp():
    pytest.importorskip('mlx.core')
    from omlx.patches.mlx_lm_mtp.batch_generator import _is_greedy, _accept_lp_for
    import mlx.core as mx

    selected = sampler(1.0, .95)
    batch = SimpleNamespace(samplers=[selected], fallback_sampler=None)
    assert not _is_greedy(batch)
    assert selected.temp == 1.0
    assert selected.top_p == .95
    assert callable(selected._mtp_sampling_logits)
    assert callable(selected.sample_with_logprobs)

    # The same upstream filter/temperature distribution must be used by the
    # draw and MTP acceptance; no exact sampled token golden expectation.
    logits = mx.array([[2., 1., 0., -3.]])
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    _, density = selected.sample_with_logprobs(lp)
    accepted_density = _accept_lp_for(selected, lp)
    mx.eval(density, accepted_density)
    assert mx.allclose(density, accepted_density).item()


def test_explicit_deterministic_sampler_remains_greedy():
    pytest.importorskip('mlx.core')
    import mlx.core as mx
    from omlx.patches.mlx_lm_mtp.batch_generator import _is_greedy

    selected = sampler(0, 1)
    assert _is_greedy(SimpleNamespace(samplers=[selected], fallback_sampler=None))
    assert selected(mx.array([[0., 3., 1.]])).item() == 1


@pytest.mark.parametrize('temperature,top_p', [
    (float('nan'), 1), (float('inf'), 1), (-1, 1),
    (1, float('nan')), (1, float('inf')), (1, -1), (1, 1.1),
])
def test_invalid_values_do_not_reach_upstream(temperature, top_p):
    with pytest.raises(ValueError):
        sampler(temperature, top_p)
