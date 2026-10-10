"""Loopback DeepSeek-recipe HTTP server for ds41f text serving."""
from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
import json
import os
from pathlib import Path
from time import time, perf_counter
from uuid import uuid4
from typing import Any

from anyio import CancelScope
from deepseek_recipe import (
    ChatCompletionRequest, ChatCompletionResponse, ConversationRequest, ConversionError,
    ConversionOptions, DeepseekV41Encoding, EOS_TOKEN, InferenceChunk, MessagesRequest, MessagesResponse,
    ResponsesRequest, ResponsesResponse, StreamProcessor, Tokenizer,
)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.types import Receive, Scope, Send

from ds41f_mlx.config import load_runtime_config
from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend, RecipePreparedRequest, DEFAULT_RECIPE, DEFAULT_MODEL_ID, MODEL_ALIASES
from ds41f_mlx.serving.request_policy import validate_stateful_chat_request_policy

PROTOCOL_TYPES = {
    'chat_completions': (ChatCompletionRequest, ChatCompletionResponse),
    'responses': (ResponsesRequest, ResponsesResponse),
    'messages': (MessagesRequest, MessagesResponse),
}


class RequestError(Exception):
    def __init__(self, message: str, status_code: int = 400, code: str | None = None) -> None:
        super().__init__(message); self.status_code = status_code; self.code = code


def load_v41_tokenizer(recipe_path: Path = DEFAULT_RECIPE) -> Any:
    return Tokenizer.from_file(str(Path(recipe_path) / 'static' / 'tokenizers' / 'v41' / 'tokenizer.json'))


def prepare_request(protocol: str, body: bytes, *, tokenizer: Any, recipe_path: Path = DEFAULT_RECIPE, options: ConversionOptions | None = None, checkpoint: Path | None = None, ordinary: bool = False) -> RecipePreparedRequest:
    prepare_t0 = perf_counter()
    request_type, _ = PROTOCOL_TYPES[protocol]
    if ordinary:
        if protocol != 'chat_completions':
            raise RequestError('ordinary MTP supports Chat Completions only')
        from .ordinary_admission import convert_ordinary
        converted, include_usage, automatic = convert_ordinary(body)
        custom_tool_names = frozenset()
    else:
        # Runtime extension, resolved only after the real recipe/image expansion.
        payload = json.loads(body)
        automatic = isinstance(payload, dict) and payload.get('max_tokens') == 'auto'
        if automatic:
            payload['max_tokens'] = 1
            body = json.dumps(payload).encode()
        try:
            request = request_type(body)
        except ValueError as error:
            raise RequestError(str(error)) from error
        include_usage = request.include_usage() if protocol == 'chat_completions' else False
        custom_tool_names = frozenset(request.custom_tool_names()) if protocol == 'responses' else frozenset()
        converted = request.convert(options if options is not None else ConversionOptions())
    convert_s = perf_counter() - prepare_t0
    encoding = DeepseekV41Encoding().with_tokenizer(tokenizer)
    render_t0 = perf_counter()
    rendered = encoding.render_conversation(converted.conversation)
    render_s = perf_counter() - render_t0
    encode_t0 = perf_counter()
    token_ids = [int(x) for x in encoding.encode(converted.conversation)]
    encode_s = perf_counter() - encode_t0
    multimodal = None
    if rendered.image_sources:
        if protocol != 'chat_completions':
            raise RequestError('multimodal support is qualified for Chat Completions only', 400)
        from ds41f_mlx.model_execution.config import ModelConfig
        from ds41f_mlx.runtime.multimodal import prepare_multimodal
        config = ModelConfig.from_dict(json.loads(((checkpoint or load_runtime_config().checkpoint_path) / 'config.json').read_text()))
        try:
            multimodal = prepare_multimodal(token_ids, list(rendered.image_sources), config)
        except ValueError as error:
            raise RequestError(str(error), 400) from error
        token_ids = list(multimodal.token_ids)
    elif 129264 in token_ids:
        raise RequestError('missing image source for image placeholder', 400)
    if len(token_ids) < 2:
        raise RequestError(f'encoded prompt must contain at least 2 tokens, got {len(token_ids)}', 400)
    try:
        stop_token_ids = tuple(int(t) for t in tokenizer.encode(EOS_TOKEN))
    except Exception:
        stop_token_ids = ()
    prepared = RecipePreparedRequest(protocol, converted, converted, token_ids, list(rendered.image_sources), include_usage, custom_tool_names, stop_token_ids, multimodal)
    from .capacity import resolve_capacity
    try:
        resolve_capacity(prepared, checkpoint=checkpoint or load_runtime_config().checkpoint_path, automatic=automatic, ordinary=ordinary)
    except ValueError as error:
        raise RequestError(str(error), 400, 'context_capacity_exhausted') from error
    if ordinary:
        prepared.phase_timings = dict(request_prepare_s=perf_counter()-prepare_t0,
                                     recipe_convert_s=convert_s, recipe_render_s=render_s,
                                     recipe_encode_s=encode_s,
                                     recipe_convert_or_encode_s=convert_s+render_s+encode_s)
        prepared.arrival_t0 = prepare_t0
    return prepared


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


def create_app(*, backend: DeepSeekRecipeRuntimeBackend | None = None, recipe_path: Path | None = None, options: ConversionOptions | None = None, model_id: str | None = None, runtime_config=None, profile='standard-off') -> FastAPI:
    if profile not in ('standard-off', 'mtp-singleton-v1', 'mtp-serving-v1'):
        raise ValueError('unknown capability profile')
    local_mtp = profile == 'mtp-singleton-v1'
    ordinary_mtp = profile == 'mtp-serving-v1'
    if ordinary_mtp:
        from .production_mtp import ProductionMTPBackend
        if not isinstance(backend, ProductionMTPBackend) or not getattr(backend, 'dependency_identity', None):
            raise ValueError('ordinary MTP requires an explicitly admitted production backend')
    runtime_config = runtime_config or load_runtime_config()
    if local_mtp:
        from .mtp_public import LocalMTPBackend
        if not isinstance(backend, LocalMTPBackend):
            raise ValueError('MTP requires its explicit singleton backend; no fallback')
        if (runtime_config.host != '127.0.0.1' or not 1 <= runtime_config.port <= 65535 or
                runtime_config.max_live_sessions != 1 or runtime_config.trace_history_limit != 32 or
                runtime_config.enable_diagnostics or runtime_config.model_id != DEFAULT_MODEL_ID or
                runtime_config.production_prefill_selector != 'DENSE_P0_P7' or
                any(getattr(runtime_config, k) != 'ON' for k in ('mtp','dspark','speculative_decode'))):
            raise ValueError('contradictory local MTP process configuration')
    runtime_config.apply_environment()
    runtime_config.apply_import_paths()
    recipe_path = Path(recipe_path) if recipe_path is not None else runtime_config.recipe_path
    model_id = model_id if model_id is not None else runtime_config.model_id
    backend = backend or DeepSeekRecipeRuntimeBackend(recipe_path=recipe_path, model_id=model_id, runtime_config=runtime_config)
    if not (local_mtp or ordinary_mtp) and isinstance(backend, DeepSeekRecipeRuntimeBackend):
        from ds41f_mlx.runtime.resource_admission import load_protocol_tokenizer
        tokenizer = load_protocol_tokenizer(recipe_path)
    else:
        tokenizer = load_v41_tokenizer(recipe_path)  # unchanged MTP/fixture path
    if local_mtp:
        from ds41f_mlx.mtp_profile import PROFILE, LIMITS, validate_chat, strict_json
        from .mtp_public import LocalBoundary, public_record
        options = ConversionOptions(default_thinking_mode=False)
    if ordinary_mtp:
        from .mtp_public import LocalBoundary
        options = ConversionOptions(default_thinking_mode=False)
    app = FastAPI(title='ds41f-deepseek-recipe', version='0.1.0',
                  docs_url=None if local_mtp else '/docs',
                  redoc_url=None if local_mtp else '/redoc',
                  openapi_url=None if local_mtp else '/openapi.json')
    if local_mtp or ordinary_mtp:
        app.add_middleware(LocalBoundary, authority=f'127.0.0.1:{runtime_config.port}', ordinary=ordinary_mtp)
    app.state.backend = backend
    app.state.recipe_tokenizer = tokenizer
    if ordinary_mtp:
        @app.on_event('shutdown')
        async def shutdown_ordinary():
            await backend.shutdown_serving()

    @app.exception_handler(ConversionError)
    async def conversion_error_handler(_request: Request, exc: ConversionError) -> Response:
        if local_mtp:
            return JSONResponse(status_code=exc.status_code,content={'error':{'code':'conversion_rejected','message':'conversion_rejected'}})
        return Response(content=exc.body, status_code=exc.status_code, media_type='application/json')

    @app.exception_handler(RequestError)
    async def request_error_handler(_request: Request, exc: RequestError) -> Response:
        error_type = 'internal_error' if exc.status_code >= 500 else 'invalid_request_error'
        code = exc.code or ({400:'invalid_profile_request',404:'session_not_live',409:'session_conflict'}.get(exc.status_code,'request_failed') if local_mtp else error_type)
        return JSONResponse(status_code=exc.status_code, content={'error': {'message': code if local_mtp else str(exc), 'type': error_type, 'param': None, 'code': code}})

    @app.exception_handler(ValueError)
    async def value_error_handler(_request: Request, exc: ValueError) -> Response:
        code = 'invalid_profile_request' if local_mtp else 'invalid_request_error'
        return JSONResponse(status_code=400, content={'error': {'message': code if local_mtp else str(exc), 'type': 'invalid_request_error', 'param': None, 'code': code}})

    def api_handler(protocol: str) -> Callable[[Request], Awaitable[Response]]:
        async def handler(request: Request) -> Response:
            arrival_t0 = perf_counter()
            body = await request.body()
            if ordinary_mtp:
                if protocol != 'chat_completions':
                    raise RequestError('ordinary MTP supports Chat Completions only')
            prepared = await run_in_threadpool(prepare_request, protocol, body, tokenizer=tokenizer, recipe_path=recipe_path, options=options, checkpoint=getattr(backend, 'checkpoint', runtime_config.checkpoint_path), ordinary=ordinary_mtp)
            if ordinary_mtp:
                prepared.arrival_t0 = arrival_t0
                prepared.phase_timings['request_prepare_s'] = perf_counter()-arrival_t0
                release = request.scope.get('ds41f.release_preparation')
                if release is not None:
                    release()
                try:
                    return await backend.ordinary_response(prepared, tokenizer=tokenizer, http_request=request)
                except ValueError as exc:
                    raise RequestError(str(exc)) from exc
                except RuntimeError as exc:
                    raise RequestError(str(exc), 503) from exc
                except Exception as exc:
                    from omlx.exceptions import SchedulerQueueFullError
                    if isinstance(exc, SchedulerQueueFullError):
                        return JSONResponse({'error': {'message': str(exc), 'code': 'queue_full'}},
                                            status_code=503, headers={'Retry-After': '1'})
                    raise
            output = response_body(prepared, backend.infer, tokenizer=tokenizer, model_id=model_id)
            if prepared.stream:
                return InferenceStreamingResponse(output, media_type='text/event-stream')
            content = ''.join([part async for part in output])
            return Response(content, media_type='application/json')
        return handler

    @app.get('/health')
    async def health() -> Response:
        fatal = getattr(backend, 'fatal_error', None)
        if local_mtp:
            ready = getattr(backend, '_model', None) is not None
            return JSONResponse(status_code=503 if fatal else 200, content={
                'status':'unavailable' if fatal else 'ready' if ready else 'alive',
                'process_alive':True, 'model_ready':ready, 'profile':PROFILE,
                'limits':LIMITS, 'mtp':'ON', 'dspark':'ON', 'depth':5,
                'qualification':'release-candidate; see M41 evidence',
                'dependency_identity':getattr(backend, 'dependency_identity', None)})
        ready = getattr(backend, '_model', None) is not None
        status = 'unavailable' if fatal else ('ready' if ready else 'alive')
        code = 503 if fatal else 200
        state = {'status': status, 'process_alive': True, 'model_ready': ready, 'fatal_error': fatal}
        if ordinary_mtp:
            scheduler = getattr(backend, 'scheduler', None)
            state.update(profile=profile, dependency_identity=backend.dependency_identity,
                         queued_requests=0 if scheduler is None else len(scheduler.waiting),
                         active_requests=0 if scheduler is None else len(scheduler.running),
                         prefix_cache=None if scheduler is None else scheduler.checkpoints.paged.get_stats().to_dict())
        return JSONResponse(status_code=code, content=state)

    @app.get('/v1/models')
    async def models() -> Response:
        model = {'id': model_id, 'object': 'model', 'owned_by': 'ds41f', 'aliases': sorted(MODEL_ALIASES)}
        if ordinary_mtp:
            from .capacity import ORDINARY_OUTPUT_CEILING
            model.update(context_length=backend.context_tokens, max_output_tokens=ORDINARY_OUTPUT_CEILING)
        return JSONResponse(content={'object': 'list', 'data': [model]})

    @app.post('/v1/sessions')
    async def create_session(request: Request) -> Response:
        raw = await request.body()
        body = strict_json(raw) if local_mtp and raw else json.loads(raw.decode()) if raw else {}
        if local_mtp and body != {}:
            raise RequestError('create accepts only empty object')
        try:
            rec = await backend.create_stateful_session(session_id=body.get('id'))
        except RuntimeError as exc:
            raise RequestError(str(exc), 409)
        return JSONResponse(content=public_record(rec) if local_mtp else rec.to_json())

    @app.get('/v1/sessions/{session_id}')
    async def get_session(session_id: str) -> Response:
        try:
            rec = backend.get_stateful_session(session_id)
            return JSONResponse(content=public_record(rec) if local_mtp else rec.to_json())
        except KeyError as exc:
            raise RequestError(str(exc), 404)

    @app.delete('/v1/sessions/{session_id}')
    async def delete_session(session_id: str) -> Response:
        try:
            return JSONResponse(content=await backend.close_stateful_session(session_id))
        except KeyError as exc:
            raise RequestError(str(exc), 404)
        except RuntimeError as exc:
            raise RequestError(str(exc), 409)

    @app.post('/v1/sessions/{session_id}/budget')
    async def session_budget(session_id: str, request: Request) -> Response:
        if local_mtp or getattr(backend, 'qualification_response', None) is not None:
            raise RequestError('budget is standard-OFF only', 400)
        body = await request.body()
        try:
            validate_stateful_chat_request_policy(body)
            prepared = await run_in_threadpool(prepare_request, 'chat_completions', body, tokenizer=tokenizer, recipe_path=recipe_path, options=options, checkpoint=getattr(backend, 'checkpoint', runtime_config.checkpoint_path))
            rec = backend.validate_stateful_admission(session_id, prepared)
            expected = request.headers.get('X-DS41F-Expected-Request-Count')
            if expected is not None and (not expected.isdecimal() or rec.request_count != int(expected)):
                raise RequestError('observed request_count changed; reconcile before budget', 409)
            return JSONResponse(content={**prepared.capacity, 'request_count': rec.request_count})
        except KeyError as error:
            raise RequestError(str(error), 404) from error
        except (ValueError, RuntimeError) as error:
            raise RequestError(str(error), 400) from error

    @app.post('/v1/sessions/{session_id}/chat/completions')
    async def session_chat(session_id: str, request: Request) -> Response:
        body = await request.body()
        if local_mtp:
            validate_chat(body)
        qualification = getattr(backend, 'qualification_response', None)
        if qualification is not None:
            if len(body) > backend.MAX_BODY_BYTES:
                raise RequestError('internal request body exceeds 1 MiB', 400)
            try:
                backend.get_stateful_session(session_id)
            except KeyError as exc:
                raise RequestError(str(exc), 404)
        try:
            validate_stateful_chat_request_policy(body)
        except ValueError as exc:
            raise RequestError(str(exc), 400)
        prepared = await run_in_threadpool(prepare_request, 'chat_completions', body, tokenizer=tokenizer, recipe_path=recipe_path, options=options, checkpoint=getattr(backend, 'checkpoint', runtime_config.checkpoint_path))
        try:
            # Internal qualification and the explicit public process profile
            # share the same guarded owner, never a request/env mode selector.
            qualification = getattr(backend, 'qualification_response', None)
            if qualification is not None:
                raw_sequence = request.headers.get('X-DS41F-Request-Sequence')
                request_sequence = int(raw_sequence) if raw_sequence is not None else None
                projection = request.headers.get('X-DS41F-Outcome-Projection', 'delivery')
                if projection not in ('delivery', 'json'):
                    raise RequestError('unsupported retained outcome projection', 400)
                return await qualification(session_id, prepared, tokenizer=tokenizer, body=body,
                    sequence=request_sequence, outcome_projection=projection == 'json')
            if prepared.stream:
                expected = request.headers.get('X-DS41F-Expected-Request-Count')
                if expected is not None:
                    if not expected.isdecimal() or backend.get_stateful_session(session_id).request_count != int(expected):
                        raise RequestError('observed request_count changed; reconcile before generation', 409)
                application_id = request.headers.get('X-DS41F-Application-Request-ID')
                if application_id is not None:
                    from uuid import UUID
                    try:
                        if str(UUID(application_id)) != application_id:
                            raise ValueError('not canonical UUID')
                    except ValueError:
                        raise RequestError('application request correlation must be a canonical UUID', 400)
                stream = await backend.open_stateful_stream(session_id, prepared, tokenizer=tokenizer, body=body, options=options, application_id=application_id)
                return InferenceStreamingResponse(stream, media_type='text/event-stream', headers={
                    'X-DS41F-Request-ID': stream.request_id, 'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})
            turn = await backend.run_stateful_chat_turn(session_id, prepared, tokenizer=tokenizer)
        except KeyError as exc:
            raise RequestError(str(exc), 404)
        except RuntimeError as exc:
            status = 409 if local_mtp or 'active request' in str(exc) or 'maximum live session' in str(exc) else 400
            raise RequestError(str(exc), status, getattr(exc,'code',None) if local_mtp else None)
        if turn.response_json is None:
            raise RequestError('stateful turn did not produce a protocol response', 500)
        return JSONResponse(content=turn.response_json)

    @app.post('/v1/sessions/{session_id}/cancel')
    async def cancel_session_turn(session_id: str, request: Request) -> Response:
        if local_mtp:
            raise RequestError('standard-OFF streaming cancellation only', 400)
        body = await request.json()
        if not isinstance(body.get('request_id'), str):
            raise RequestError('request_id is required')
        try:
            return JSONResponse(content=await backend.cancel_stateful_stream(session_id, body['request_id']))
        except KeyError as exc:
            raise RequestError(str(exc), 404)
        except RuntimeError as exc:
            raise RequestError(str(exc), 409)

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
        try:
            rec = await backend.restore_stateful_session(artifact_path=Path(body['artifact_path']), tokenizer=tokenizer, session_id=body.get('id'))
        except RuntimeError as exc:
            raise RequestError(str(exc), 409)
        return JSONResponse(content=rec.to_json())

    if not local_mtp and os.environ.get('DS41F_ENABLE_DIAGNOSTIC_ENDPOINTS') == '1':
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


# Launcher resolves capability and dependencies before creating an app.
# No default backend is constructed as an import side effect.
