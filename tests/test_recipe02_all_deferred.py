"""Focused native conversion, canonical encoding and Runtime boundary regressions."""
import copy
import json
from pathlib import Path
import sys

import deepseek_recipe as d
import pytest

from ds41f_mlx.serving.messages_admission import convert_messages

TOKENIZER = Path(sys.prefix) / 'share/ds41f-mtp/recipe/static/tokenizers/v41/tokenizer.json'


def body():
    return dict(model='deepseek-flash', max_tokens=64, thinking={'type': 'disabled'},
                tools=[dict(name=n, input_schema={'type': 'object'}, defer_loading=True)
                       for n in ('first', 'second')],
                messages=[dict(role='user', content='before'),
                          dict(role='system', content=[addition('second'), addition('first')]),
                          dict(role='user', content='after')])


def addition(name):
    return dict(type='tool_addition', tool=dict(type='tool_reference', name=name))


def convert(value):
    return convert_messages(json.dumps(value).encode())[0]


def test_canonical_empty_initial_schemas_and_order():
    converted = convert(body())
    tools = [d.ToolDefinition(name=n, parameters={'type': 'object'}, defer_loading=True)
             for n in ('first', 'second')]
    expected = d.Conversation(thinking_mode=False, tools=tools, messages=[
        d.UserMessage('before'), d.SystemMessage('', tools=[tools[1], tools[0]]),
        d.UserMessage('after')])
    tok = d.Tokenizer.from_file(str(TOKENIZER))
    encoding = d.DeepseekV41Encoding().with_tokenizer(tok)
    prompt = encoding.render_conversation(converted.conversation).prompt
    assert prompt == encoding.render_conversation(expected).prompt
    assert list(encoding.encode(converted.conversation)) == list(encoding.encode(expected))
    initial, update = prompt.split('### Updated Tool Schemas')
    assert '### Available Tool Schemas' in initial
    assert '"name": "first"' not in initial and '"name": "second"' not in initial
    assert update.index('"name": "second"') < update.index('"name": "first"')
    assert [t.name for t in converted.conversation.tools] == ['first', 'second']


def test_unactivated_definitions_do_not_change_canonical_tokens():
    original = body()
    extended = copy.deepcopy(original)
    extended['tools'].append(dict(name='not_yet', input_schema={'type': 'object'}, defer_loading=True))
    tok = d.Tokenizer.from_file(str(TOKENIZER))
    encoding = d.DeepseekV41Encoding().with_tokenizer(tok)
    assert list(encoding.encode(convert(original).conversation)) == list(encoding.encode(convert(extended).conversation))
    extended['messages'] = [dict(role='user', content='before')]
    prompt = encoding.render_conversation(convert(extended).conversation).prompt
    assert '### Available Tool Schemas' in prompt and '### Updated Tool Schemas' not in prompt
    assert all(f'"name": "{n}"' not in prompt for n in ('first', 'second', 'not_yet'))
    for tools in ([], None):
        empty = extended | {'tools': tools}
        assert '### Available Tool Schemas' not in encoding.render_conversation(convert(empty).conversation).prompt


def test_paired_lookup_uses_actual_canonical_tokens():
    from test_production_mtp import MX, pair
    from ds41f_mlx.serving.paired_checkpoint import PairedCheckpointAuthority
    tok = d.Tokenizer.from_file(str(TOKENIZER))
    encoding = d.DeepseekV41Encoding().with_tokenizer(tok)
    value = body()
    tokens = list(encoding.encode(convert(value).conversation))
    C = len(tokens) - 3
    assert C >= 128
    target, rings, _ = pair(C)
    authority = PairedCheckpointAuthority('all-deferred', MX)
    authority.publish(authority.capture(target, rings, tokens[:C]))
    value['tools'].append(dict(name='not_yet', input_schema={'type': 'object'}, defer_loading=True))
    unchanged = list(encoding.encode(convert(value).conversation))
    assert authority.acquire(unchanged)[2] == tokens[:C]
    value['tools'][0]['description'] = 'changed activated schema'
    changed = list(encoding.encode(convert(value).conversation))
    assert changed != tokens and authority.acquire(changed) is None
    assert authority.acquire(unchanged)[2] == tokens[:C]


@pytest.mark.parametrize('fault', ['unknown', 'duplicate', 'name', 'schema', 'user_role',
                                  'assistant_role', 'removal', 'orphan', 'missing_result'])
def test_invalid_declarations_and_histories_still_rejected(fault):
    value = body()
    if fault == 'unknown':
        value['messages'][1]['content'] = [addition('absent')]
    elif fault == 'duplicate':
        value['tools'].append(copy.deepcopy(value['tools'][0]))
    elif fault == 'name':
        value['tools'][0]['name'] = 'invalid name'
    elif fault == 'schema':
        value['tools'][0]['input_schema'] = {'type': 'not-a-type'}
    elif fault in ('user_role', 'assistant_role'):
        value['messages'][1]['role'] = fault.split('_')[0]
    elif fault == 'removal':
        value['messages'][1]['content'] = [dict(type='tool_removal', tool=dict(type='tool_reference', name='first'))]
    elif fault == 'orphan':
        value['messages'].append(dict(role='user', content=[dict(type='tool_result', tool_use_id='absent', content='bad')]))
    else:
        value['messages'].append(dict(role='assistant', content=[dict(type='tool_use', id='exact-id', name='first', input={})]))
    with pytest.raises((ValueError, d.ConversionError)):
        convert(value)


def test_prepared_runtime_tokens_and_exact_id_continuation():
    from ds41f_mlx.config import load_runtime_config
    from ds41f_mlx.serving.server import prepare_request
    value = body()
    # Historical names may be omitted by Harness addition-only projection.
    value['messages'] += [dict(role='assistant', content=[dict(type='tool_use', id='exact-history-id', name='removed', input={})]),
                          dict(role='user', content=[dict(type='tool_result', tool_use_id='exact-history-id', content='ok')])]
    converted = convert(value)
    assistant = next(m for m in converted.conversation.messages if m.role == 'assistant')
    assert assistant.tool_calls[0].id == 'exact-history-id'
    assert assistant.tool_calls[0].name == 'removed'
    tok = d.Tokenizer.from_file(str(TOKENIZER))
    encoding = d.DeepseekV41Encoding().with_tokenizer(tok)
    prepared = prepare_request('messages', json.dumps(value).encode(), tokenizer=tok,
                               checkpoint=load_runtime_config().checkpoint_path, ordinary=True)
    assert prepared.token_ids == list(encoding.encode(converted.conversation))
    value['messages'][-1]['content'][0]['tool_use_id'] = 'different-id'
    with pytest.raises((ValueError, d.ConversionError)):
        convert(value)
