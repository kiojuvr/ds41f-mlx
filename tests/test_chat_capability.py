import asyncio
import json

import httpx
import pytest

from ds41f_mlx.chat_capability import negotiate, PRODUCTION_DEFAULTS, LEGACY_DEFAULTS
from ds41f_mlx.chat_application import ChatApplication


def health(**changes):
    return dict(status='ready', profile='name-is-not-authority',
                capabilities={'stateful_chat': 'production-sessions-v1'},
                chat_defaults=dict(PRODUCTION_DEFAULTS), **changes)


def test_authority_not_profile_and_standard_off_compatibility():
    assert negotiate(health()) == PRODUCTION_DEFAULTS
    assert negotiate({'status': 'ready', 'profile': 'standard-off'}) == LEGACY_DEFAULTS


@pytest.mark.parametrize('runtime', [
    {'status': 'ready', 'profile': 'mtp-serving-v1'},
    {'status': 'ready', 'capabilities': {'stateful_chat': 'future-v2'}},
    dict(health(), chat_defaults=dict(PRODUCTION_DEFAULTS, top_p=.95)),
    dict(health(), status='loading'),
])
def test_fail_closed(runtime):
    with pytest.raises(ValueError):
        negotiate(runtime)


def test_versioned_policy_negotiates_authoritative_values_not_frontend_constants():
    from ds41f_mlx.serving.generation_policy import capability_contract
    contract = capability_contract()
    contract['defaults'].update(temperature=1., top_p=.95)
    defaults = dict(contract['defaults'], reasoning_effort='none', max_tokens='auto')
    runtime = dict(health(), generation_policy=contract, chat_defaults=defaults)
    assert negotiate(runtime) == defaults
    runtime['chat_defaults'] = dict(defaults, temperature=0)
    with pytest.raises(ValueError, match='disagree'):
        negotiate(runtime)


def test_tui_freezes_negotiated_defaults_and_text_tools():
    async def check():
        def handler(req):
            if req.url.path == '/api/status':
                return httpx.Response(200, json={'runtime': health(), 'tools': [
                    {'function': {'name': name}} for name in ('fetch_url', 'fetch_image', 'fetch_pdf', 'generic')]})
            return httpx.Response(200, json={'state': 'empty', 'request_count': 0})
        app = ChatApplication(transport=httpx.MockTransport(handler))
        await app.new()
        assert {key: app.request()[key] for key in PRODUCTION_DEFAULTS} == PRODUCTION_DEFAULTS
        assert [t['function']['name'] for t in app.tools] == ['fetch_url', 'generic']
        app.runtime['chat_defaults']['temperature'] = 1
        assert app.request()['temperature'] == 0
        await app.shutdown()
    asyncio.run(check())
