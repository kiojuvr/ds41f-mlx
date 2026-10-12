import pytest

from ds41f_mlx.chat_capability import negotiate, validate, PARAMETERS
from ds41f_mlx.serving.generation_policy import capability_contract


def health():
    contract = capability_contract()
    return dict(status='ready', profile='not-authority', capabilities={'stateful_chat': 'production-sessions-v1'},
                generation_policy=contract, chat_defaults=dict(contract['defaults'], reasoning_effort='none', max_tokens='auto'))


def test_authority_and_ranges():
    runtime = health()
    runtime['generation_policy']['defaults']['temperature'] = .7
    runtime['chat_defaults']['temperature'] = .7
    values = negotiate(runtime)
    assert values['temperature'] == .7
    contract = runtime['generation_policy']
    for name in PARAMETERS:
        spec = contract['parameters'][name]
        if 'maximum' in spec:
            with pytest.raises(ValueError): validate(contract, values | {name: spec['maximum'] + 1})
        if 'exclusiveMinimum' in spec:
            with pytest.raises(ValueError): validate(contract, values | {name: spec['exclusiveMinimum']})
    for value in (False, float('nan'), float('inf')):
        with pytest.raises(ValueError): validate(contract, values | {'temperature': value})
    with pytest.raises(ValueError): validate(contract, values | {'seed': int('9' * 400)})
    off = health(); off['profile'] = 'standard-off'; off.pop('capabilities')
    assert negotiate(off) == off['chat_defaults']


@pytest.mark.parametrize('change', ['missing', 'version', 'capability', 'defaults', 'reasoning', 'loading', 'parameter'])
def test_fail_closed(change):
    runtime = health()
    if change == 'missing': runtime.pop('generation_policy')
    if change == 'version': runtime['generation_policy']['version'] = 'future'
    if change == 'capability': runtime['capabilities']['stateful_chat'] = 'future'
    if change == 'defaults': runtime['chat_defaults']['temperature'] += .1
    if change == 'reasoning': runtime['generation_policy']['reasoning']['change'] = 'allow'
    if change == 'loading': runtime['status'] = 'loading'
    if change == 'parameter': runtime['generation_policy']['parameters']['future'] = {}
    with pytest.raises(ValueError): negotiate(runtime)
