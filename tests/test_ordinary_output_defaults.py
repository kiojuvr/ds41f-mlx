"""Ordinary budget semantics through authoritative conversion and real encoding."""
import json

import pytest
from deepseek_recipe import ConversionError

from ds41f_mlx.config import load_runtime_config
from ds41f_mlx.serving.capacity import resolve_capacity
from ds41f_mlx.serving.generation_policy import reasoning_policy
from ds41f_mlx.serving.ordinary_admission import convert_ordinary
from ds41f_mlx.serving.server import RequestError, load_v41_tokenizer, prepare_request


@pytest.fixture(scope='module')
def preparation():
    cfg = load_runtime_config()
    tokenizer = load_v41_tokenizer(cfg.recipe_path)
    def prepare(controls, stream=False, checkpoint=None, default='none'):
        body = dict(model=cfg.model_id, messages=[dict(role='user', content='Hello')], stream=stream) | controls
        return prepare_request('chat_completions', json.dumps(body).encode(), tokenizer=tokenizer,
                               checkpoint=checkpoint or cfg.checkpoint_path, ordinary=True,
                               default_reasoning_effort=default)
    return prepare


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('controls,mode,effort,maximum', [
    ({}, False, None, 8192),
    ({'reasoning_effort': 'none'}, False, None, 8192),
    ({'thinking': {'type': 'disabled'}}, False, None, 8192),
    ({'reasoning_effort': 'minimal'}, True, 'low', 65536),
    ({'reasoning_effort': 'low'}, True, 'low', 65536),
    ({'reasoning_effort': 'medium'}, True, 'high', 65536),
    ({'reasoning_effort': 'high'}, True, 'high', 65536),
    ({'reasoning_effort': 'xhigh'}, True, 'high', 65536),
    ({'thinking': {'type': 'enabled'}}, True, 'high', 65536),
    ({'reasoning_effort': 'max'}, True, 'max', 131072),
])
def test_omission_uses_canonical_effective_reasoning(preparation, stream, controls, mode, effort, maximum):
    request = preparation(controls, stream)
    assert reasoning_policy(request) == dict(thinking_mode=mode, reasoning_effort=effort)
    assert request.inference_options.max_tokens is None  # admission did not invent 128
    assert request.resolved_max_tokens == maximum
    assert request.capacity['prompt_tokens'] == len(request.token_ids)
    if controls.get('reasoning_effort') == 'xhigh':
        assert request.converted.conversation.reasoning_effort == 'xhigh'  # recipe unchanged


@pytest.mark.parametrize('effort', ['none', 'low', 'high', 'xhigh', 'max'])
@pytest.mark.parametrize('maximum', [1, 128, 8192, 393216, 'auto'])
def test_explicit_limits_override_reasoning_defaults(preparation, effort, maximum):
    request = preparation(dict(reasoning_effort=effort, max_tokens=maximum))
    assert request.resolved_max_tokens == (393216 if maximum == 'auto' else maximum)
    assert request.inference_options.max_tokens == (1 if maximum == 'auto' else maximum)


def test_xhigh_canonicalization_does_not_change_model_prompt(preparation):
    assert preparation({'reasoning_effort': 'xhigh'}).token_ids == preparation({'reasoning_effort': 'high'}).token_ids


def test_runtime_reasoning_defaults_are_not_changed(preparation):
    assert preparation({}, default='none').resolved_max_tokens == 8192
    assert preparation({}, default='max').resolved_max_tokens == 131072
    assert preparation({'reasoning_effort': 'none'}, default='max').resolved_max_tokens == 8192


def test_resolved_omission_reserves_actual_recipe_prompt(preparation, tmp_path):
    request = preparation({})
    prompt = len(request.token_ids)
    (tmp_path / 'config.json').write_text(json.dumps(dict(max_position_embeddings=prompt + 8192)))
    assert preparation({}, checkpoint=tmp_path).resolved_max_tokens == 8192
    # One extra recipe-expanded token must reject, not silently shrink the default.
    request.token_ids.append(1)
    with pytest.raises(ValueError, match=r'prompt \+ requested output'):
        resolve_capacity(request, checkpoint=tmp_path, ordinary=True)
    (tmp_path / 'config.json').write_text(json.dumps(dict(max_position_embeddings=prompt + 8191)))
    with pytest.raises(RequestError, match=r'prompt \+ requested output'):
        preparation({}, checkpoint=tmp_path)
    assert preparation({'max_tokens': 'auto'}, checkpoint=tmp_path).resolved_max_tokens == 8191


def test_explicit_ceiling_still_rejects_in_admission(preparation):
    with pytest.raises(ConversionError, match='max_tokens'):
        preparation({'max_tokens': 393217})


def test_admission_does_not_resolve_omission():
    body = dict(model='deepseek-v4.1-flash', messages=[dict(role='user', content='Hi')])
    converted, _, automatic = convert_ordinary(json.dumps(body).encode())
    assert converted.inference_options.max_tokens is None and not automatic
