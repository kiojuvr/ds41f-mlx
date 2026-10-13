import json

import pytest

from ds41f_mlx.config import RuntimeConfig, load_runtime_config
from ds41f_mlx.serving.ordinary_admission import convert_ordinary
from ds41f_mlx.serving.generation_policy import capability_contract
from ds41f_mlx.chat_capability import negotiate


@pytest.mark.parametrize('controls,mode,effort', [
    ({}, True, 'max'),
    ({'reasoning_effort': 'none'}, False, None),
    ({'reasoning_effort': 'low'}, True, 'low'),
    ({'thinking': {'type': 'disabled'}}, False, None),
])
def test_runtime_default_uses_recipe_and_respects_explicit_controls(controls, mode, effort):
    body = dict(model='deepseek-v4.1-flash', messages=[dict(role='user', content='hello')], max_tokens=128)
    converted, _, _ = convert_ordinary(json.dumps(body | controls).encode(), default_reasoning_effort='max')
    assert converted.conversation.thinking_mode is mode
    assert converted.conversation.reasoning_effort == effort


def test_environment_default_and_validation(monkeypatch):
    monkeypatch.setenv('DS41F_DEFAULT_REASONING_EFFORT', 'max')
    assert load_runtime_config().default_reasoning_effort == 'max'
    with pytest.raises(ValueError):
        RuntimeConfig(default_reasoning_effort='invalid')


def test_saved_tui_default(monkeypatch, tmp_path):
    from ds41f_mlx.operator_settings import load_reasoning, save_reasoning
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    monkeypatch.delenv('DS41F_DEFAULT_REASONING_EFFORT', raising=False)
    assert load_reasoning() == 'max'
    save_reasoning('low')
    assert load_reasoning() == 'low'
    monkeypatch.setenv('DS41F_DEFAULT_REASONING_EFFORT', 'none')
    assert load_reasoning() == 'none'


def test_frontend_accepts_runtime_max_default():
    contract = capability_contract('max')
    runtime = dict(status='ready', profile='standard-off', generation_policy=contract,
                   chat_defaults=dict(contract['defaults'], reasoning_effort='max', max_tokens='auto'))
    assert negotiate(runtime)['reasoning_effort'] == 'max'


def test_launch_passes_default(monkeypatch):
    from ds41f_mlx.operator_tui import launch
    captured = {}
    def popen(args, **kwargs):
        captured['args'] = args
        return object()
    monkeypatch.setattr('ds41f_mlx.operator_tui.subprocess.Popen', popen)
    launch(default_reasoning_effort='max')
    assert captured['args'][-2:] == ['--default-reasoning-effort', 'max']
