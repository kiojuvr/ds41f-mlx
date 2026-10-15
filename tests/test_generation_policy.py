import json
from types import SimpleNamespace as NS

import pytest
from deepseek_recipe import ChatCompletionRequest, ConversionOptions
from ds41f_mlx.serving.generation_policy import normalize_policy, reasoning_policy, capability_contract
from ds41f_mlx.serving.production_sessions import ProductionSessions


def converted(**changes):
    body = dict(model='deepseek-v4.1-flash', messages=[{'role':'user','content':'Hi'}]) | changes
    return ChatCompletionRequest(json.dumps(body).encode()).convert(ConversionOptions(default_thinking_mode=False))


def test_effective_policy_retains_recipe_values_and_runtime_extensions():
    payload = dict(temperature=1, top_p=.95, top_k=40, min_p=.05,
                   repetition_penalty=1.1, presence_penalty=.2, frequency_penalty=.3, seed=123)
    policy = normalize_policy(converted(**payload), payload)
    assert policy.to_json() == pytest.approx(payload)
    params = policy.sampling_params(32)
    assert all(getattr(params, key) == pytest.approx(value) for key, value in payload.items())
    from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend
    sampler = DeepSeekRecipeRuntimeBackend.make_sampler(None, policy)
    assert sampler.temp == 1 and sampler.top_p == pytest.approx(.95) and sampler.top_k == 40 and sampler.min_p == .05
    assert len(policy.processors()) == 3


@pytest.mark.parametrize('key,value', [('top_k', True), ('top_k', -1), ('top_k', 1.5),
    ('min_p', float('nan')), ('min_p', 2), ('repetition_penalty', 0),
    ('presence_penalty', float('inf')), ('frequency_penalty', -3), ('seed', True),
    ('seed', -1), ('seed', 2**32)])
def test_extensions_reject_invalid_without_silent_omission(key, value):
    with pytest.raises(ValueError):
        normalize_policy(converted(), {key:value})


def test_recipe_normalized_reasoning_aliases_cannot_bypass_session_envelope():
    sessions = ProductionSessions(NS(fatal_error=None))
    rec = sessions.create()
    off = NS(conversation_request=converted(reasoning_effort='none'))
    on = NS(protocol='chat_completions', image_sources=[], token_ids=[1,2],
            conversation_request=converted(reasoning_effort='high'))
    assert reasoning_policy(off)['thinking_mode'] is False
    assert reasoning_policy(on)['thinking_mode'] is True
    rec.envelope = dict(tools=None, reasoning=reasoning_policy(off))
    with pytest.raises(ValueError, match='envelope changed'):
        sessions.validate(rec.session_id, on, {'messages':[{'role':'user','content':'Hi'}]})


def test_xhigh_and_high_share_the_canonical_session_envelope():
    sessions = ProductionSessions(NS(fatal_error=None))
    rec = sessions.create()
    high = NS(conversation_request=converted(reasoning_effort='high'))
    xhigh = NS(protocol='chat_completions', image_sources=[], token_ids=[1, 2],
               conversation_request=converted(reasoning_effort='xhigh'))
    rec.envelope = dict(tools=None, reasoning=reasoning_policy(high))
    assert reasoning_policy(xhigh) == reasoning_policy(high)
    sessions.validate(rec.session_id, xhigh, {'messages': [{'role': 'user', 'content': 'Hi'}]})


def test_contract_records_bounded_acceptance_and_upstream_context_semantics():
    contract = capability_contract()
    assert contract['qualification'] == 'real-model-qualified'
    assert contract['execution_profiles'] == ['standard-off', 'mtp-serving-v1']
    assert contract['defaults']['temperature'] == 1.0
    assert contract['defaults']['top_p'] == .95
    assert contract['reasoning']['scope'] == 'session'
    assert 'xhigh' in contract['reasoning']['efforts']  # accepted compatibility spelling
    assert contract['reasoning']['effective_efforts'] == ['none', 'low', 'high', 'max']
    assert contract['reasoning']['compatibility_aliases'] == {'xhigh': 'high'}
    assert contract['parameters']['presence_penalty']['context_tokens'] == 20


@pytest.mark.parametrize('ordinary', [False, True])
@pytest.mark.parametrize('changes,temperature,top_p', [({}, 1, .95),
    ({'temperature': None, 'top_p': None}, 1, .95),
    ({'temperature': 0, 'top_p': 1}, 0, 1)])
def test_production_defaults_are_recipe_converted_and_greedy_is_explicit(monkeypatch, ordinary, changes, temperature, top_p):
    from ds41f_mlx.serving import server, capacity
    class Encoding:
        def with_tokenizer(self, tokenizer): return self
        def render_conversation(self, conversation): return NS(image_sources=[])
        def encode(self, conversation): return [1, 2]
    monkeypatch.setattr(server, 'DeepseekV41Encoding', Encoding)
    monkeypatch.setattr(capacity, 'resolve_capacity', lambda *a, **k: None)
    body = dict(model='deepseek-v4.1-flash', messages=[{'role':'user','content':'Hi'}]) | changes
    request = server.prepare_request('chat_completions', json.dumps(body).encode(),
        tokenizer=NS(encode=lambda text: [1]), ordinary=ordinary)
    assert request.generation_policy.temperature == request.inference_options.temperature == temperature
    assert request.generation_policy.top_p == request.inference_options.top_p == pytest.approx(top_p)
    assert reasoning_policy(request)['thinking_mode'] is False
