"""Recipe owns ordinary tool-result binding; singleton stays bounded."""
import json
import pytest

from ds41f_mlx.mtp_profile import WEATHER, validate_chat, validate_completed_calls


def body():
    return dict(model='deepseek-v4.1-flash', max_tokens=128, messages=[dict(role='user', content='Read a file')],
        tools=[dict(type='function', function=dict(name=name, parameters=dict(type='object',
            properties=dict(path=dict(type='string')), required=['path']))) for name in ('read_file', 'stat_file', 'edit_file')])


def admit(value):
    pytest.importorskip('mlx.core')
    pytest.importorskip('omlx')
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    ProductionMTPBackend.validate_ordinary(json.dumps(value).encode())


def test_arbitrary_declarations_and_recipe_supported_choice():
    value = body()
    for choice in ('auto', 'none', 'required', {'type': 'function', 'function': {'name': 'read_file'}}):
        admit(value | dict(tool_choice=choice))
    with pytest.raises(ValueError, match='pinned weather'):
        validate_chat(json.dumps(value).encode())


def test_multiple_calls_result_identity_and_opencode_empty_reasoning():
    value = body()
    calls = [dict(id=f'call-{i}', type='function', function=dict(name=t['function']['name'], arguments='{"path":"a"}'))
             for i, t in enumerate(value['tools'])]
    message = dict(role='assistant', content=None, reasoning_content='', tool_calls=calls)
    value['messages'] += [message] + [dict(role='tool', tool_call_id=c['id'], content='OK') for c in reversed(calls)]
    admit(value)
    # IDs bind results, not names or client-local sessions. Both duplicate and
    # foreign results fail before execution; recipe owns their rendering.
    for identity in ('foreign', 'call-2'):
        changed = json.loads(json.dumps(value))
        changed['messages'][-1]['tool_call_id'] = identity
        from deepseek_recipe import ConversionError
        with pytest.raises(ConversionError, match='tool_call_id'):
            admit(changed)
    with pytest.raises(ValueError):
        validate_completed_calls(message)


@pytest.mark.parametrize('tools', [[{}], [{'type': 'function', 'function': {}}], [{'type': 'function', 'function': {'name': 'read', 'parameters': 'not a schema'}}]])
def test_malformed_declarations_are_recipe_client_errors_not_weather(tools):
    from deepseek_recipe import ConversionError
    with pytest.raises((ValueError, ConversionError)) as error:
        admit(body() | dict(tools=tools))
    assert 'weather' not in str(error.value)


def test_singleton_weather_contract_still_accepts_pinned_round_trip():
    value = body() | dict(tools=[WEATHER])
    call = dict(id='weather-1', type='function', function=dict(name='lookup_weather', arguments='{"city":"Tokyo"}'))
    assistant = dict(role='assistant', content=None, tool_calls=[call])
    value['messages'] += [assistant, dict(role='tool', tool_call_id=call['id'], content='Sunny')]
    validate_chat(json.dumps(value).encode())
    validate_completed_calls(assistant)
    call['function']['arguments'] = '{"path":"a"}'
    with pytest.raises(ValueError, match='city'):
        validate_chat(json.dumps(value).encode())
    with pytest.raises(ValueError, match='city'):
        validate_completed_calls(assistant)
