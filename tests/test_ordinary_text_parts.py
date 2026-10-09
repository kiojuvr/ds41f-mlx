"""Ordinary request grammar comes from the installed authoritative recipe."""
import json

import pytest
from deepseek_recipe import ChatCompletionRequest, ConversionError, ConversionOptions

from ds41f_mlx.mtp_profile import validate_chat
from ds41f_mlx.serving.ordinary_admission import convert_ordinary


def body(**changes):
    return dict(model='deepseek-v4.1-flash', max_tokens=16,
                messages=[dict(role='user', content='こんにちは')]) | changes


def convert(value):
    return convert_ordinary(json.dumps(value, ensure_ascii=False).encode())[0]


def recipe(value):
    return ChatCompletionRequest(value).convert(ConversionOptions(default_thinking_mode=False))


def parts():
    return [dict(type='text', text='こんにちは'),
            dict(type='text', text='<system-reminder>Plan mode: read-only.</system-reminder>')]


@pytest.mark.parametrize('changes', [
    dict(messages=[dict(role='user', content=parts())]),
    dict(messages=[dict(role='system', content=parts()), dict(role='user', content='Hi')]),
    dict(messages=[dict(role='assistant', content=parts())]),
    dict(messages=[dict(role='system', content='Only a system message')]),
    dict(messages=[dict(role='latest_reminder', content='Remember this')]),
    dict(messages=[dict(role='user', content=[dict(type='text', text='Hi', client_metadata=True)], name='client')]),
    dict(messages=[dict(role='user', content=[])]),
    dict(n=1, temperature=None, top_p=1, frequency_penalty=0, presence_penalty=0),
    dict(reasoning_effort='high', thinking=dict(type='disabled')),
    dict(stream=True, stream_options=dict(include_usage=False)),
    dict(stream=True, stream_options={}),
    dict(stop='END'), dict(stop=['END', 'STOP']),
    dict(response_format=dict(type='text')),
    dict(messages=[dict(role='user', content='Return json')], response_format=dict(type='json_object')),
    dict(tools=[], tool_choice='none'),
    dict(client_extension={'anything': 'metadata'}),
])
def test_recipe_accepted_grammar_is_not_redefined(changes):
    value = body(**changes)
    authoritative = recipe(value)
    converted = convert(value)
    assert [(m.role, m.content) for m in converted.conversation.messages] == [
        (m.role, m.content) for m in authoritative.conversation.messages]
    assert converted.parsing_options.stop_sequences == authoritative.parsing_options.stop_sequences
    assert converted.parsing_options.parse_json_output == authoritative.parsing_options.parse_json_output
    assert converted.stream == authoritative.stream


@pytest.mark.parametrize('changes', [
    dict(messages=[]), dict(messages=[dict(role='developer', content='Hi')]),
    dict(messages=[dict(role='user', content=None)]),
    dict(messages=[dict(role='user', content=[dict(type='text', text=None)])]),
    dict(messages=[dict(role='assistant', content=None)]),
    dict(n=2), dict(max_tokens=0), dict(temperature=-1),
    dict(stream=False, stream_options={}), dict(stop=['x'] * 17),
    dict(response_format=dict(type='regex', regex='.*')),
    dict(tools=[{}]), dict(tool_choice='invalid'),
])
def test_recipe_rejections_propagate_unchanged(changes):
    value = body(**changes)
    with pytest.raises((ValueError, ConversionError)) as expected:
        recipe(value)
    with pytest.raises(type(expected.value)) as actual:
        convert(value)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize('changes,match', [
    (dict(model='other'), 'model alias'),
    (dict(temperature=1), 'temperature'), (dict(top_p=0.5), 'top_p'),
    (dict(reasoning_effort='high'), 'thinking-off'),
    (dict(thinking=dict(type='enabled')), 'thinking-off'),
    (dict(seed=1), 'seed'), (dict(frequency_penalty=1), 'frequency_penalty'),
    (dict(presence_penalty=1), 'presence_penalty'),
    (dict(parallel_tool_calls=False), 'parallel_tool_calls'),
    (dict(thinking=dict(type='disabled', budget_tokens=10)), 'budget_tokens'),
    (dict(response_format=dict(type='json_schema', json_schema={})), 'JSON Schema'),
    (dict(logprobs=True), 'logprobs'), (dict(top_logprobs=1), 'logprobs'),
    (dict(max_tokens=393217), 'output capability ceiling'),
    (dict(messages=[dict(role='user', content=[dict(type='image_url', image_url=dict(url='https://example.com/a.png'))])]), 'text Chat'),
    (dict(messages=[dict(role='user', content='<|im_start|>')]), 'special-token'),
])
def test_recipe_acceptance_does_not_override_runtime_limits(changes, match):
    value = body(**changes)
    recipe(value)
    with pytest.raises(ValueError, match=match):
        convert(value)


def test_plan_parts_leave_singleton_contract_unchanged():
    value = body(messages=[dict(role='user', content=parts())])
    convert(value)
    with pytest.raises(ValueError, match='bounded string'):
        validate_chat(json.dumps(value).encode())


def test_tool_results_have_body_budget_not_historical_fixture_limit():
    value = body(messages=[dict(role='assistant', content=None, tool_calls=[dict(
        id='call-1', function=dict(name='read', arguments='{}'))]),
        dict(role='tool', tool_call_id='call-1', content=[dict(type='text', text='あ' * 30000)])])
    assert convert(value).conversation.messages[-1].content == 'あ' * 30000


@pytest.mark.parametrize('raw', [b'[]', b'{"messages": [], "messages": []}',
    b'{"temperature": NaN}', b'{"temperature": 1e999}', b'\xff', b'{'])
def test_structural_security_rejections(raw):
    with pytest.raises(ValueError):
        convert_ordinary(raw)
