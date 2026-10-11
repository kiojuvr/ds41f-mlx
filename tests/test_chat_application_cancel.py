"""Interruption must settle via GET, never from the visible SSE prefix."""
import asyncio
import json

import httpx

from ds41f_mlx.chat_application import ChatApplication


def test_stop_fences_active_identity_and_recovers_canonical_suffix():
    async def check():
        started = asyncio.Event()
        cancelled = False
        posts = []
        identity = None
        active = False

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'event: admitted\ndata: {"request_id":"native-request"}\n\n'
                yield b'data: {"choices":[{"delta":{"content":"visible prefix"}}]}\n\n'
                started.set()
                await asyncio.Event().wait()

        async def handler(req):
            nonlocal cancelled, identity, active
            value = json.loads(req.content or b'{}')
            path = req.url.path
            if req.method == 'POST':
                posts.append((path, value))
            if path == '/api/status':
                return httpx.Response(200, json={'runtime': {'status': 'ready', 'profile': 'standard-off'}, 'tools': []})
            if path == '/api/session':
                return httpx.Response(200, json={})
            if path.endswith('/cancel'):
                assert value == {'request_id': 'native-request'}
                cancelled = True; active = False
                return httpx.Response(200, json={})
            if path.startswith('/api/session/'):
                turn = {'application_request_id': identity, 'request_id': 'native-request', 'cancelled': True,
                        'response_json': {'choices': [{'message': {'role': 'assistant', 'content': 'canonical prefix + suffix'}}]}}
                return httpx.Response(200, json={'state': 'busy' if active else 'idle',
                    'active_request_id': 'native-request' if active else None,
                    'request_count': 1 if cancelled else 0, 'last_turn': turn if cancelled else None})
            if path == '/api/budget':
                return httpx.Response(200, json={})
            if path == '/api/stream':
                active = True; identity = value['application_id']
                return httpx.Response(200, stream=Stream())
            raise AssertionError(path)

        app = ChatApplication(transport=httpx.MockTransport(handler))
        await app.new()
        task = asyncio.create_task(app.send('日本語'))
        await asyncio.wait_for(started.wait(), 2)
        assert app.live['content'] == 'visible prefix'
        await app.stop()
        await asyncio.wait_for(task, 2)
        assert app.messages[-1]['content'] == 'canonical prefix + suffix'
        assert app.pending is app.live is None
        assert app.count == 1 and not app.problem
        assert len([p for p, _ in posts if p == '/api/stream']) == 1
        await app.shutdown()
    asyncio.run(check())
