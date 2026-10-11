import asyncio
import json

import httpx
import pytest

from ds41f_mlx.chat_capability import negotiate, Settings, PARAMETERS
from ds41f_mlx.chat_application import ChatApplication
from ds41f_mlx.serving.generation_policy import capability_contract


def health():
    contract = capability_contract()
    return dict(status='ready', profile='not-authority', capabilities={'stateful_chat': 'production-sessions-v1'},
                generation_policy=contract, chat_defaults=dict(contract['defaults'], reasoning_effort='none', max_tokens='auto'))


def test_authority_and_ranges():
    runtime = health()
    runtime['generation_policy']['defaults']['temperature'] = .7
    runtime['chat_defaults']['temperature'] = .7
    assert negotiate(runtime)['temperature'] == .7
    settings = Settings(runtime)
    for name in PARAMETERS:
        spec = settings.contract['parameters'][name]
        if 'maximum' in spec:
            with pytest.raises(ValueError): settings.change(name, spec['maximum'] + 1)
        if 'exclusiveMinimum' in spec:
            with pytest.raises(ValueError): settings.change(name, spec['exclusiveMinimum'])
    for value in (False, float('nan'), float('inf')):
        with pytest.raises(ValueError): settings.change('temperature', value)
    with pytest.raises(ValueError): settings.change('seed', int('9' * 400))
    settings.change('seed', None)
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


def test_workflow_snapshot_and_session_reasoning():
    settings = Settings(health())
    settings.change('reasoning_effort', 'low')
    settings.begin_turn(); settings.freeze_reasoning()
    original = settings.request()
    settings.change('temperature', .6)
    assert settings.request() == original  # continuation belongs to original user workflow
    with pytest.raises(ValueError, match='session-owned'): settings.change('reasoning_effort', 'none')
    settings.begin_turn()
    assert settings.request()['temperature'] == .6
    assert settings.request()['reasoning_effort'] == 'low'
    new = Settings(health(), settings.values)
    new.change('reasoning_effort', 'high')


def test_tui_serializes_all_parameters_and_keeps_preferences_on_new():
    async def check():
        def handler(req):
            if req.url.path == '/api/status':
                return httpx.Response(200, json={'runtime': health(), 'tools': [
                    {'function': {'name': name}} for name in ('fetch_url', 'fetch_image', 'fetch_pdf')]})
            return httpx.Response(200, json={'state': 'empty', 'request_count': 0})
        app = ChatApplication(transport=httpx.MockTransport(handler))
        await app.new()
        assert all(app.request()[key] == value for key, value in health()['chat_defaults'].items())
        assert [t['function']['name'] for t in app.tools] == ['fetch_url']
        app.settings.change('top_k', 40); app.settings.change('min_p', .05)
        app.settings.begin_turn()
        app.append_results({'messages': [{'role': 'tool', 'tool_call_id': 'x', 'content': 'ok'}]})
        await app.reconcile()
        assert app.request()['top_k'] == 40 and app.request()['min_p'] == .05
        await app.new()
        assert app.settings.values['top_k'] == 40
        await app.shutdown()
    asyncio.run(check())
