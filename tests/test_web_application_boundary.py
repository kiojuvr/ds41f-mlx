"""Application-only security/effect regressions; no execution doubles as evidence."""
import asyncio
import json
from types import SimpleNamespace

import httpx

from ds41f_mlx import web
from ds41f_mlx.web_tools import ToolRegistry, ToolResult


class CounterTool:
    name = 'fetch_url'
    schema = {'type':'function','function':{'name':'fetch_url','parameters':{'type':'object'}}}
    def __init__(self): self.executions = 0
    def run(self, arguments, context=None):
        self.executions += 1
        return ToolResult(json.dumps({'url':arguments['url'],'excerpt':'one effect'}), {'tool':self.name})


class RuntimeDouble:
    base_url = 'http://127.0.0.1:8000'
    count = 1
    def __init__(self, *args): pass
    def health(self): return {'status':'ready'}
    def get_session(self, sid):
        return {'id':sid,'state':'idle','request_count':self.count,'last_turn':{
            'request_id':'request1','reconstruction':{'executable_tools':True},
            'response_json':{'choices':[{'message':{'tool_calls':[{
                'id':'call1','function':{'name':'fetch_url','arguments':'{"url":"https://example.com"}'}}]}}]}}}


def test_tool_effect_reserved_once_and_restart_observation_never_executes(monkeypatch):
    tool = CounterTool()
    monkeypatch.setattr(web, 'RuntimeClient', RuntimeDouble)
    monkeypatch.setattr(web, 'registry_from_env', lambda: ToolRegistry([tool]))
    app = web.create_app()
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1') as client:
            responses = await asyncio.gather(*[client.post('/api/tools', json={'session_id':'s','request_count':1}) for _ in range(2)])
            assert all(r.status_code == 200 for r in responses)
            assert responses[0].json() == responses[1].json()
            assert tool.executions == 1
            observed = await client.get('/api/tools/result?session_id=s&request_count=1')
            assert observed.json()['result']['messages'][0]['tool_call_id'] == 'call1'
            RuntimeDouble.count = 2  # consumption advances authority; old receipt is not executable
            try:
                observed = await client.get('/api/tools/result?session_id=s&request_count=1')
                assert observed.json() == {'result': None, 'observation_only': True}
                assert tool.executions == 1
            finally:
                RuntimeDouble.count = 1
        restarted = web.create_app()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=restarted), base_url='http://127.0.0.1') as client:
            observed = await client.get('/api/tools/result?session_id=s&request_count=1')
            assert observed.json()['result'] is None
            assert tool.executions == 1
    asyncio.run(run())


def test_completed_effect_budget_failure_remains_observable_and_never_reexecutes(monkeypatch):
    from ds41f_mlx.web_client import RuntimeHTTPError
    tool = CounterTool()
    monkeypatch.setattr(web, 'RuntimeClient', RuntimeDouble)
    monkeypatch.setattr(web, 'registry_from_env', lambda: ToolRegistry([tool]))
    attempts = []
    def fail_budget(*args):
        attempts.append(args)
        raise RuntimeHTTPError(409, 'prefix changed')
    monkeypatch.setattr('ds41f_mlx.web_budget.fit_tool_results', fail_budget)
    app = web.create_app()
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1') as client:
            value = {'session_id': 's', 'request_count': 1, 'request': {'max_tokens': 'auto'}}
            first = await client.post('/api/tools', json=value)
            result = first.json()
            assert first.status_code == 200 and 'budget_error' in result
            assert json.loads(result['messages'][0]['content'])['excerpt'] == 'one effect'
            repeat = await client.post('/api/tools', json=value)
            observed = await client.get('/api/tools/result?session_id=s&request_count=1')
            assert repeat.json() == result == observed.json()['result']
            assert tool.executions == 1 and len(attempts) == 1
    asyncio.run(run())


def test_local_api_rejects_cross_origin_and_form_mutation(monkeypatch):
    monkeypatch.setattr(web, 'RuntimeClient', RuntimeDouble)
    app = web.create_app()
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1') as client:
            response = await client.post('/api/tools', json={}, headers={'Origin':'https://evil.example'})
            assert response.status_code == 403
            response = await client.post('/api/tools', data={'session_id':'s'})
            assert response.status_code == 415
            response = await client.get('/api/status', headers={'Host':'evil.example'})
            assert response.status_code == 400
            response = await client.get('/')
            assert "object-src 'none'" in response.headers['content-security-policy']
    asyncio.run(run())
