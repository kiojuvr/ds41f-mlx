"""Loopback DeepSeek-recipe HTTP server for ds41f text serving."""
from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
import json
import os
from pathlib import Path
from time import time
from uuid import uuid4
from typing import Any

from anyio import CancelScope
from deepseek_recipe import (
    ChatCompletionRequest, ChatCompletionResponse, ConversationRequest, ConversionError,
    ConversionOptions, DeepseekV41Encoding, InferenceChunk, MessagesRequest, MessagesResponse,
    ResponsesRequest, ResponsesResponse, StreamProcessor, Tokenizer,
)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.types import Receive, Scope, Send

from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend, RecipePreparedRequest, DEFAULT_RECIPE, DEFAULT_MODEL_ID, MODEL_ALIASES

PROTOCOL_TYPES = {
    'chat_completions': (ChatCompletionRequest, ChatCompletionResponse),
    'responses': (ResponsesRequest, ResponsesResponse),
    'messages': (MessagesRequest, MessagesResponse),
}


class RequestError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message); self.status_code = status_code


def load_v41_tokenizer(recipe_path: Path = DEFAULT_RECIPE) -> Any:
    return Tokenizer.from_file(str(Path(recipe_path) / 'static' / 'tokenizers' / 'v41' / 'tokenizer.json'))


def prepare_request(protocol: str, body: bytes, *, tokenizer: Any, recipe_path: Path = DEFAULT_RECIPE, options: ConversionOptions | None = None) -> RecipePreparedRequest:
    request_type, _ = PROTOCOL_TYPES[protocol]
    try:
        request = request_type(body)
    except ValueError as error:
        raise RequestError(str(error)) from error
    include_usage = request.include_usage() if protocol == 'chat_completions' else False
    custom_tool_names = frozenset(request.custom_tool_names()) if protocol == 'responses' else frozenset()
    converted = request.convert(options if options is not None else ConversionOptions())
    encoding = DeepseekV41Encoding().with_tokenizer(tokenizer)
    rendered = encoding.render_conversation(converted.conversation)
    token_ids = [int(x) for x in encoding.encode(converted.conversation)]
    if rendered.image_sources:
        raise RequestError('multimodal/image input is not supported by current ds41f text-only serving backend', 400)
    if len(token_ids) < 2:
        raise RequestError(f'encoded prompt must contain at least 2 tokens, got {len(token_ids)}', 400)
    return RecipePreparedRequest(protocol, converted, converted, token_ids, list(rendered.image_sources), include_usage, custom_tool_names)


class InferenceStreamingResponse(StreamingResponse):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with CancelScope(shield=True):
                await self.body_iterator.aclose()


def sse_frame(event: str | None, data: str) -> str:
    prefix = f'event: {event}\n' if event is not None else ''
    return f'{prefix}data: {data}\n\n'


async def response_body(request: RecipePreparedRequest, infer: Callable[[RecipePreparedRequest], AsyncIterator[InferenceChunk]], *, tokenizer: Any, model_id: str) -> AsyncIterator[str]:
    request_type, response_type = PROTOCOL_TYPES[request.protocol]
    response_id = str(uuid4())
    model = request.model or model_id
    generator = request_type.chunk_generator(request.conversation_request, response_id, model)
    if request.protocol == 'chat_completions':
        generator = generator.with_include_usage(request.include_usage)
    elif request.protocol == 'responses':
        generator = generator.with_custom_tool_names(request.custom_tool_names)
    processor = StreamProcessor(generator, request.conversation_request.parsing_options, tokenizer)
    response = None if request.stream else response_type(response_id, model, int(time()), 0, 0)
    inference = None
    try:
        inference = aiter(infer(request))
        while True:
            try:
                chunk = await anext(inference)
            except StopAsyncIteration:
                chunks = processor.finish()
            else:
                chunks = processor.push(chunk)
            for output in chunks:
                if request.stream:
                    yield sse_frame(response_type.chunk_event_type(output), output.to_json())
                else:
                    response.append(output)
            if processor.finished:
                break
        if request.stream:
            done = response_type.done_message()
            if done is not None:
                yield sse_frame(None, done)
        else:
            yield response.to_json()
    finally:
        try:
            close = getattr(inference, 'aclose', None)
            if close is not None:
                with CancelScope(shield=True):
                    await close()
        finally:
            processor.close()


def _chat_sse_events(events: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> AsyncIterator[str]:
    async def gen() -> AsyncIterator[str]:
        for event in events:
            yield sse_frame(None, json.dumps(event, separators=(',', ':')))
        yield sse_frame(None, '[DONE]')
    return gen()


def create_app(*, backend: DeepSeekRecipeRuntimeBackend | None = None, recipe_path: Path = DEFAULT_RECIPE, options: ConversionOptions | None = None, model_id: str = DEFAULT_MODEL_ID) -> FastAPI:
    backend = backend or DeepSeekRecipeRuntimeBackend(recipe_path=recipe_path, model_id=model_id)
    tokenizer = load_v41_tokenizer(recipe_path)
    app = FastAPI(title='ds41f-deepseek-recipe', version='0.1.0')
    app.state.backend = backend
    app.state.recipe_tokenizer = tokenizer

    @app.exception_handler(ConversionError)
    async def conversion_error_handler(_request: Request, exc: ConversionError) -> Response:
        return Response(content=exc.body, status_code=exc.status_code, media_type='application/json')

    @app.exception_handler(RequestError)
    async def request_error_handler(_request: Request, exc: RequestError) -> Response:
        error_type = 'internal_error' if exc.status_code >= 500 else 'invalid_request_error'
        return JSONResponse(status_code=exc.status_code, content={'error': {'message': str(exc), 'type': error_type, 'param': None, 'code': error_type}})

    @app.exception_handler(ValueError)
    async def value_error_handler(_request: Request, exc: ValueError) -> Response:
        return JSONResponse(status_code=400, content={'error': {'message': str(exc), 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_request_error'}})

    def api_handler(protocol: str) -> Callable[[Request], Awaitable[Response]]:
        async def handler(request: Request) -> Response:
            body = await request.body()
            prepared = await run_in_threadpool(prepare_request, protocol, body, tokenizer=tokenizer, recipe_path=recipe_path, options=options)
            output = response_body(prepared, backend.infer, tokenizer=tokenizer, model_id=model_id)
            if prepared.stream:
                return InferenceStreamingResponse(output, media_type='text/event-stream')
            content = ''.join([part async for part in output])
            return Response(content, media_type='application/json')
        return handler

    @app.get('/health')
    async def health() -> Response:
        fatal = getattr(backend, 'fatal_error', None)
        ready = getattr(backend, '_model', None) is not None
        status = 'unavailable' if fatal else ('ready' if ready else 'alive')
        code = 503 if fatal else 200
        return JSONResponse(status_code=code, content={'status': status, 'process_alive': True, 'model_ready': ready, 'fatal_error': fatal})

    @app.get('/v1/models')
    async def models() -> Response:
        return JSONResponse(content={'object': 'list', 'data': [{'id': model_id, 'object': 'model', 'owned_by': 'ds41f', 'aliases': sorted(MODEL_ALIASES)}]})

    @app.post('/v1/sessions')
    async def create_session(request: Request) -> Response:
        raw = await request.body()
        body = json.loads(raw.decode()) if raw else {}
        try:
            rec = await backend.create_stateful_session(session_id=body.get('id'))
        except RuntimeError as exc:
            raise RequestError(str(exc), 409)
        return JSONResponse(content=rec.to_json())

    @app.get('/v1/sessions/{session_id}')
    async def get_session(session_id: str) -> Response:
        try:
            return JSONResponse(content=backend.get_stateful_session(session_id).to_json())
        except KeyError as exc:
            raise RequestError(str(exc), 404)

    @app.delete('/v1/sessions/{session_id}')
    async def delete_session(session_id: str) -> Response:
        try:
            return JSONResponse(content=await backend.close_stateful_session(session_id))
        except KeyError as exc:
            raise RequestError(str(exc), 404)

    @app.post('/v1/sessions/{session_id}/chat/completions')
    async def session_chat(session_id: str, request: Request) -> Response:
        body = await request.body()
        prepared = await run_in_threadpool(prepare_request, 'chat_completions', body, tokenizer=tokenizer, recipe_path=recipe_path, options=options)
        try:
            turn = await backend.run_stateful_chat_turn(session_id, prepared, tokenizer=tokenizer)
        except KeyError as exc:
            raise RequestError(str(exc), 404)
        except RuntimeError as exc:
            status = 409 if 'active request' in str(exc) or 'maximum live session' in str(exc) else 400
            raise RequestError(str(exc), status)
        if prepared.stream:
            return InferenceStreamingResponse(_chat_sse_events(turn.stream_events), media_type='text/event-stream')
        if turn.response_json is None:
            raise RequestError('stateful turn did not produce a protocol response', 500)
        return JSONResponse(content=turn.response_json)

    @app.post('/v1/sessions/{session_id}/persist')
    async def persist_session(session_id: str, request: Request) -> Response:
        raw = await request.body()
        body = json.loads(raw.decode()) if raw else {}
        try:
            artifact = await backend.persist_stateful_session(session_id, artifact_root=Path(body['artifact_root']) if body.get('artifact_root') else None)
        except KeyError as exc:
            raise RequestError(str(exc), 404)
        except RuntimeError as exc:
            raise RequestError(str(exc), 409)
        return JSONResponse(content={'session_id': session_id, 'artifact': artifact})

    @app.post('/v1/sessions/restore')
    async def restore_session(request: Request) -> Response:
        body = await request.json()
        if not body.get('artifact_path'):
            raise RequestError('artifact_path is required')
        rec = await backend.restore_stateful_session(artifact_path=Path(body['artifact_path']), tokenizer=tokenizer, session_id=body.get('id'))
        return JSONResponse(content=rec.to_json())

    if os.environ.get('DS41F_ENABLE_DIAGNOSTIC_ENDPOINTS') == '1':
        @app.get('/_ds41f/diagnostics')
        async def diagnostics() -> Response:
            traces = [t.to_json() for t in getattr(backend, 'traces', [])]
            return JSONResponse(content={
                'trace_count': len(traces),
                'trace_limit': getattr(getattr(backend, 'traces', None), 'maxlen', None),
                'traces': traces,
                'session_traces': list(getattr(backend, 'session_traces', [])),
                'sessions': {sid: rec.to_json() for sid, rec in getattr(backend, 'sessions', {}).items()},
                'last_trace': None if getattr(backend, 'last_trace', None) is None else backend.last_trace.to_json(),
                'lock_locked': backend._lock.locked(),
                'active_generation_sessions': getattr(backend, 'active_generation_sessions', None),
                'model_ready': getattr(backend, '_model', None) is not None,
                'fatal_error': getattr(backend, 'fatal_error', None),
            })

    for path, protocol in (('/v1/chat/completions','chat_completions'),('/v1/responses','responses'),('/v1/messages','messages')):
        app.add_api_route(path, api_handler(protocol), methods=['POST'], name=protocol)
    return app


app = create_app()
