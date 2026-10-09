"""Ordinary request grammar comes from the installed authoritative recipe."""
import json
from pathlib import Path

import pytest
from deepseek_recipe import ChatCompletionRequest, ConversionError, ConversionOptions, IMAGE_SPECIAL_TOKEN

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
    dict(messages=[dict(role='user', content="Code: if '<|' in value or '｜' in value: pass")]),
    dict(messages=[dict(role='user', content='<｜User｜>Quoted prompt token')]),
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


@pytest.mark.parametrize('path', [
    'ds41f_mlx/serving/production_mtp.py', 'ds41f_mlx/serving/capacity.py',
    'ds41f_mlx/mtp_profile.py', 'ds41f_mlx/serving/ordinary_admission.py',
])
def test_repository_review_tool_results_follow_recipe(path):
    source = (Path(__file__).resolve().parents[1] / path).read_text()
    value = body(messages=[dict(role='user', content='Review the repository'),
        dict(role='assistant', content='Read source', tool_calls=[dict(id='read-1',
            type='function', function=dict(name='read', arguments=json.dumps({'filePath': path})))]),
        dict(role='tool', tool_call_id='read-1', content=source)])
    assert convert(value).conversation.messages[-1].content == source
    assert recipe(value).conversation.messages[-1].content == source


@pytest.mark.parametrize('surface', ['system', 'user', 'assistant', 'reasoning',
                                    'latest_reminder', 'tool', 'arguments', 'description', 'parameters'])
def test_token_like_text_is_recipe_owned_on_all_prompt_surfaces(surface):
    text = "if '<|' in value or '｜' in value: pass; <｜Assistant｜> quoted delimiter"
    value = body()
    if surface in ('system', 'user', 'assistant', 'latest_reminder'):
        value['messages'] = [dict(role=surface, content=text), dict(role='user', content='Continue')]
    elif surface == 'reasoning':
        value['messages'] = [dict(role='assistant', content='Previous reply', reasoning_content=text),
                             dict(role='user', content='Continue')]
    elif surface in ('tool', 'arguments'):
        value['messages'] += [dict(role='assistant', content=None, tool_calls=[dict(id='read-1',
            function=dict(name='read', arguments=json.dumps({'path': text if surface == 'arguments' else 'a'})))]),
            dict(role='tool', tool_call_id='read-1', content=text if surface == 'tool' else 'OK')]
    else:
        function = dict(name='read', parameters=dict(type='object', properties={}))
        if surface == 'description':
            function['description'] = text
        else:
            function['parameters']['description'] = text
        value['tools'] = [dict(type='function', function=function)]
    converted, authoritative = convert(value).conversation, recipe(value).conversation
    def projection(conversation):
        return ([ (m.role, m.content, m.reasoning_content,
                   [(c.name, c.arguments) for c in m.tool_calls or ()]) for m in conversation.messages ],
                [(t.name, t.description, t.parameters) for t in conversation.tools])
    assert projection(converted) == projection(authoritative)
    assert '<|' in str(projection(converted))


@pytest.mark.parametrize('surface', ['user', 'tool', 'arguments', 'description', 'parameters'])
def test_unbacked_image_placeholder_remains_authoritative_rejection(surface):
    value = body()
    if surface == 'user':
        value['messages'][0]['content'] = IMAGE_SPECIAL_TOKEN
    elif surface in ('tool', 'arguments'):
        value['messages'] += [dict(role='assistant', content=None, tool_calls=[dict(id='read-1',
            function=dict(name='read', arguments=json.dumps({'path': IMAGE_SPECIAL_TOKEN if surface == 'arguments' else 'a'}, ensure_ascii=False)))]),
            dict(role='tool', tool_call_id='read-1', content=IMAGE_SPECIAL_TOKEN if surface == 'tool' else 'OK')]
    else:
        function = dict(name='read', parameters=dict(type='object', properties={}))
        if surface == 'description':
            function['description'] = IMAGE_SPECIAL_TOKEN
        else:
            function['parameters']['description'] = IMAGE_SPECIAL_TOKEN
        value['tools'] = [dict(type='function', function=function)]
    with pytest.raises(ConversionError) as expected:
        recipe(value)
    with pytest.raises(ConversionError) as actual:
        convert(value)
    assert str(actual.value) == str(expected.value)


def test_singleton_still_rejects_token_like_source():
    with pytest.raises(ValueError, match='raw special-token source'):
        validate_chat(json.dumps(body(messages=[dict(role='user', content="'<|' and '｜'")])).encode())


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
