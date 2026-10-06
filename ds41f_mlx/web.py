"""Single-user localhost application proxy. Runtime owns execution and state.

The browser owns ordinary application history; this process executes only declared
client tools. Stream disconnect propagates through the upstream HTTP connection.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from urllib.parse import quote

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware
from anyio import CancelScope

from ds41f_mlx.web_client import RuntimeClient, RuntimeHTTPError, StatefulToolChatClient
from ds41f_mlx.web_tools import registry_from_env, ToolError

STATIC_DIR = Path(__file__).with_name('web_static')
MAX_BODY = 100 * 1024 * 1024


class WebStreamingResponse(StreamingResponse):
    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            with CancelScope(shield=True):
                await self.body_iterator.aclose()


def create_app(*, runtime_base_url: str | None = None) -> FastAPI:
    runtime = RuntimeClient(runtime_base_url or os.environ.get('DS41F_RUNTIME_URL', 'http://127.0.0.1:8000'))
    tools = registry_from_env()
    app = FastAPI(title='ds41f-local-web-client', version='0.2.0')
    app.state.runtime = runtime
    app.state.tool_registry = tools
    # Bounded living-client tool effect ledger, NOT model/session authority.
    effects = {}
    effect_locks = {}
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', '[::1]'])

    async def call(fn, *args, **kwargs):
        return await run_in_threadpool(fn, *args, **kwargs)

    async def body(request):
        chunks = bytearray()
        async for chunk in request.stream():
            if len(chunks) + len(chunk) > MAX_BODY:
                raise ToolError('application request exceeds 100 MiB')
            chunks.extend(chunk)
        value = json.loads(chunks or b'{}')
        if not isinstance(value, dict): raise ToolError('request must be an object')
        return value

    @app.middleware('http')
    async def local_boundary(request, call_next):
        if request.method in ('POST', 'DELETE') and request.url.path.startswith('/api/'):
            origin = request.headers.get('origin')
            if (origin and origin != f'{request.url.scheme}://{request.url.netloc}') or request.headers.get('sec-fetch-site') == 'cross-site':
                return JSONResponse(status_code=403, content={'error': {'message': 'same-origin local API only'}})
            if request.method == 'POST' and request.headers.get('content-type', '').split(';')[0] != 'application/json':
                return JSONResponse(status_code=415, content={'error': {'message': 'application/json required'}})
            try:
                if (await call(runtime.health)).get('profile') == 'mtp-singleton-v1':
                    return JSONResponse(status_code=400, content={'error': {'message': 'Web application supports standard-OFF only'}})
            except RuntimeHTTPError as exc:
                return JSONResponse(status_code=502, content={'error': {'message': str(exc)}})
        response = await call_next(request)
        response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' data: blob:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.exception_handler(RuntimeHTTPError)
    async def runtime_error(_, exc):
        return JSONResponse(status_code=502 if exc.status >= 500 else exc.status, content={'error': {'message': str(exc), 'runtime_status': exc.status}})

    @app.exception_handler(ToolError)
    async def tool_error(_, exc):
        return JSONResponse(status_code=400, content={'error': {'message': str(exc), 'type': 'tool_error'}})

    @app.get('/')
    async def index(): return FileResponse(STATIC_DIR/'index.html')

    @app.get('/api/status')
    async def status():
        try: health = await call(runtime.health)
        except Exception as exc: health = {'status': 'unreachable', 'error': str(exc)}
        return JSONResponse(content={'runtime': health, 'runtime_base_url': runtime.base_url, 'tools': tools.declarations})

    @app.post('/api/session')
    async def create_session(request: Request):
        value = await body(request)
        return JSONResponse(content=await call(runtime.create_session, value.get('id')))

    @app.get('/api/session/{session_id}')
    async def get_session(session_id: str): return JSONResponse(content=await call(runtime.get_session, session_id))

    @app.delete('/api/session/{session_id}')
    async def close_session(session_id: str):
        try:
            record = await call(runtime.close_session, session_id)
        except RuntimeHTTPError as exc:
            if exc.status == 404:
                effects.pop(session_id, None)
                effect_locks.pop(session_id, None)
            raise
        effects.pop(session_id, None)
        effect_locks.pop(session_id, None)
        return JSONResponse(content=record)

    @app.post('/api/session/{session_id}/cancel')
    async def cancel(session_id: str, request: Request):
        return JSONResponse(content=await call(runtime.request, 'POST', f'/v1/sessions/{quote(session_id, safe="")}/cancel', await body(request)))

    @app.post('/api/stream')
    async def stream(request: Request):
        value = await body(request)
        sid = value.get('session_id')
        if not isinstance(sid, str) or not isinstance(value.get('request'), dict): raise ToolError('session_id and request are required')
        rec = await call(runtime.get_session, sid)
        if rec['request_count'] != value.get('expected_count') or rec['state'] not in ('empty', 'idle'):
            raise ToolError('runtime/application frontier mismatch or busy; reconcile first')
        upstream = dict(value['request'], stream=True, model='deepseek-v4.1-flash')
        # No retries, transcript surgery, image conversion or hidden tool loop.
        async def relay():
            try:
                async with httpx.AsyncClient(timeout=1800) as client:
                    async with client.stream('POST', runtime.base_url + f'/v1/sessions/{quote(sid, safe="")}/chat/completions', json=upstream, headers={'X-DS41F-Expected-Request-Count': str(value['expected_count']), 'X-DS41F-Application-Request-ID': str(value['application_id'])}) as response:
                        if response.status_code != 200:
                            raise RuntimeHTTPError(response.status_code, (await response.aread()).decode('utf-8', 'replace'))
                        request_id = response.headers['x-ds41f-request-id']
                        yield 'event: admitted\ndata: ' + json.dumps({'request_id': request_id}) + '\n\n'
                        async for chunk in response.aiter_bytes():
                            yield chunk
            except asyncio.CancelledError:
                raise  # context managers close upstream socket; runtime settles
            except Exception as exc:
                yield 'event: error\ndata: ' + json.dumps({'message': str(exc)}) + '\n\n'
        return WebStreamingResponse(relay(), media_type='text/event-stream', headers={'X-Accel-Buffering': 'no'})

    @app.get('/api/tools/result')
    async def observe_tool_result(session_id: str, request_count: int):
        rec = await call(runtime.get_session, session_id)
        entry = effects.get(session_id)
        identity = (request_count, (rec.get('last_turn') or {}).get('request_id'))
        result = entry['result'] if entry and entry['identity'] == identity and rec['request_count'] == request_count else None
        return JSONResponse(content={'result': result, 'observation_only': True})

    @app.post('/api/tools')
    async def execute_tools(request: Request):
        value = await body(request)
        sid, count = value.get('session_id'), value.get('request_count')
        if not isinstance(sid, str): raise ToolError('session_id required')
        if sid not in effect_locks:
            if len(effect_locks) >= 32: raise ToolError('tool ledger capacity reached; close application sessions')
            effect_locks[sid] = asyncio.Lock()
        async with effect_locks[sid]:
            rec = await call(runtime.get_session, sid)
            turn = rec.get('last_turn') or {}
            cert = turn.get('reconstruction') or {}
            if rec['state'] != 'idle' or rec['request_count'] != count or not cert.get('executable_tools'):
                raise ToolError('tools require the settled executable canonical turn')
            calls = turn['response_json']['choices'][0]['message'].get('tool_calls') or []
            if not 1 <= len(calls) <= 8: raise ToolError('tool batch must contain 1–8 calls')
            identity = (count, turn['request_id'])
            entry = effects.get(sid)
            if entry is not None and entry['identity'] == identity:
                if entry['result'] is None: raise ToolError('tool effect outcome uncertain; automatic retry forbidden')
                return JSONResponse(content=entry['result'])
            entry = effects[sid] = {'identity': identity, 'result': None}  # reserve before effects
            messages, displays = await call(tools.execute_calls, calls, session_id=sid)
            entry['result'] = {'messages': messages, 'results': displays, 'request_count': count}
            return JSONResponse(content=entry['result'])

    @app.post('/api/session/{session_id}/persist')
    async def persist(session_id: str, request: Request):
        await body(request)
        return JSONResponse(content=await call(runtime.persist, session_id))

    @app.post('/api/session/restore')
    async def restore(request: Request):
        value = await body(request)
        if not value.get('artifact_path'): raise ToolError('artifact_path required')
        return JSONResponse(content=await call(runtime.restore, value['artifact_path'], value.get('id')))

    @app.post('/api/chat')
    async def legacy_chat(request: Request):
        value = await body(request)
        if not isinstance(value.get('message'), str) or not value['message'].strip(): raise ToolError('message required')
        result = await call(StatefulToolChatClient(runtime, tools).run_turn, session_id=value['session_id'],
            transcript=value.get('transcript') or [], user_message=value['message'],
            max_tokens=int(value.get('max_tokens', 512)), temperature=float(value.get('temperature', 0)), tools_enabled=bool(value.get('tools_enabled', True)))
        return JSONResponse(content=result.to_json())

    app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run the localhost ds41f Web application')
    parser.add_argument('--host', default=os.environ.get('DS41F_WEB_HOST', '127.0.0.1'))
    parser.add_argument('--port', type=int, default=int(os.environ.get('DS41F_WEB_PORT', '8080')))
    parser.add_argument('--runtime-url', default=os.environ.get('DS41F_RUNTIME_URL', 'http://127.0.0.1:8000'))
    args = parser.parse_args(argv)
    if args.host not in ('127.0.0.1', 'localhost', '::1'): parser.error('single-user local application must bind loopback')
    import uvicorn
    uvicorn.run(create_app(runtime_base_url=args.runtime_url), host=args.host, port=args.port)

if __name__ == '__main__': main()
