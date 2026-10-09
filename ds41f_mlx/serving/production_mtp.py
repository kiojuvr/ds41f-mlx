"""Opt-in ordinary serving over the pinned Scheduler and qualified MTP core.

Scheduler queues/IDs/abort admission remain upstream. Only its model-specific
external-prefill, decode, and completion hooks differ: qualified P5/recipe
settlement replaces generic prefill/parser/backend-finish cache publication.
"""
import asyncio
import json
import logging
from time import perf_counter
from uuid import uuid4

from anyio import CancelScope
from fastapi.responses import JSONResponse
from omlx.scheduler import Scheduler, SchedulerConfig, SchedulerOutput
from omlx.request import Request, RequestOutput, RequestStatus, SamplingParams

from .internal_mtp import InternalMTPQualificationBackend, QualificationSession
from .paired_checkpoint import PairedCheckpointAuthority

logger = logging.getLogger('uvicorn.error.ds41f')


class ProductionScheduler(Scheduler):
    def __init__(self, backend):
        import mlx.core as mx
        from mlx_lm.generate import generation_stream
        tokenizer = getattr(backend._execution_tokenizer, 'tokenizer', backend._execution_tokenizer)
        super().__init__(backend._model.language_model, tokenizer,
                         SchedulerConfig(max_num_seqs=1, completion_batch_size=1,
                                         prefill_step_size=2048, decode_fairness=False,
                                         model_name=backend.dependency_identity), generation_stream)
        self.backend = backend
        self.checkpoints = PairedCheckpointAuthority(backend.dependency_identity, mx)
        self.paged_cache_manager = self.checkpoints.paged
        self._completed = []
        self.last_settlement = None

    def _new_batch(self, *args, **kwargs):
        from mlx_lm.generate import BatchGenerator
        if self.batch_generator is not None:
            raise RuntimeError('Scheduler already owns an executable batch')
        self.batch_generator = BatchGenerator(*args, **kwargs)
        return self.batch_generator

    def _schedule_waiting(self):
        if self.running or not self.waiting:
            return [], []
        request = self.waiting.popleft()
        rec, trace = request.execution, request.trace
        def capture(target, rings, tokens):
            request.prompt_checkpoint = self.checkpoints.capture(target, rings, tokens)
        try:
            restored = self.checkpoints.acquire(request.prompt_token_ids)
            if restored is not None:
                rec.cache, rec.rings, rec.canonical = restored
            request.cached_tokens = trace['cached_tokens'] = len(rec.canonical)
            request.remaining_tokens = request.prompt_token_ids[len(rec.canonical):]
            self.backend._start(rec, request.prepared, request.recipe_tokenizer, trace,
                                checkpoint_capture=capture, batch_generator_factory=self._new_batch)
            # P5 revoked the producer and inserted the held-out terminal into
            # THIS Scheduler's native batch. Register the unique request/UID.
            uid = rec.owner.uid
            if rec.owner._bg is not self.batch_generator or uid is None:
                raise RuntimeError('foreign P5 executable owner')
            request.batch_uid = uid
            self.request_id_to_uid[request.request_id] = uid
            self.uid_to_request_id[uid] = request.request_id
            request.status = RequestStatus.RUNNING
            self.running[request.request_id] = request
            self.total_prompt_tokens += request.num_prompt_tokens
            return [request], []
        except BaseException as exc:
            self._retire(request, error=exc)
            return [], []

    def _retire(self, request, *, error=None, cancelled=False):
        rec, trace = request.execution, request.trace
        publication = False
        try:
            if error is None:
                if rec.owner is not None:
                    self.backend._settle(rec, trace)
                if 'quiescence' not in trace or rec.owner is not None:
                    raise RuntimeError('publication without canonical settlement')
                if not rec.poisoned:
                    # Both payloads are immutable before their hashes become
                    # discoverable. A prompt capture is never an active alias.
                    # At the envelope ceiling no admitted future prompt can
                    # extend this frontier with P5 holdout. Retain the earlier
                    # prompt pair, not an unusable 8192-token root/full block.
                    final = (self.checkpoints.capture(rec.cache, rec.rings, rec.canonical)
                             if len(rec.canonical) < 8192 else None)
                    if request.prompt_checkpoint is not None:
                        self.checkpoints.publish(request.prompt_checkpoint)
                    if final is not None:
                        self.checkpoints.publish(final)
                    publication = True
        except BaseException as exc:
            error = exc
            rec.poisoned = True
        finally:
            try:
                self.backend._retire(rec)
            except BaseException as exc:
                error = error or exc
            # Even native retirement failure cannot leave an executable alias
            # eligible for the next request. Fail the engine, rather than replay.
            if error is not None:
                self.backend.fatal_error = repr(error)
                self.checkpoints.clear()
            self.waiting = type(self.waiting)(r for r in self.waiting if r.request_id != request.request_id)
            self.batch_generator = None
            uid = self.request_id_to_uid.pop(request.request_id, None)
            if uid is not None:
                self.uid_to_request_id.pop(uid, None)
            self.running.pop(request.request_id, None)
            self.requests.pop(request.request_id, None)
            request.prompt_checkpoint = None
            choices = (trace.get('response') or {}).get('choices') or []
            semantic_finish = (choices[0].get('finish_reason') if choices else None) or 'stop'
            request.status = (RequestStatus.FINISHED_ABORTED if cancelled else
                              RequestStatus.FINISHED_LENGTH_CAPPED if semantic_finish == 'length' else
                              RequestStatus.FINISHED_STOPPED)
            request.output_token_ids = list(trace.get('canonical_generated', ()))
            self.total_completion_tokens += len(request.output_token_ids)
            trace.update(cancelled=cancelled, cache_published=publication,
                         elapsed_s=perf_counter()-trace['t0'])
            self.last_settlement = trace
            logger.info('MTP settled request=%s cached=%d canonical=%s published=%s cancelled=%s error=%s',
                        request.request_id, request.cached_tokens, trace.get('canonical_frontier'),
                        publication, cancelled, error)
            output = RequestOutput(request_id=request.request_id, finished=True,
                                   finish_reason='error' if error else 'abort' if cancelled else semantic_finish,
                                   error=str(error) if error else None)
            output.recipe_events = rec.pending_delivery[request.delivery_cursor:]
            output.recipe_response = trace.get('response')
            output.serving_trace = trace
            self._completed.append(output)
            rec.guard = rec.processor = rec.owner = None
            rec.cache = rec.rings = None
            rec.canonical.clear()

    def _do_abort_request(self, request_id):
        request = self.requests.get(request_id)
        if request is None:
            return False
        if request_id not in self.running:
            self.waiting = type(self.waiting)(r for r in self.waiting if r.request_id != request_id)
            self.requests.pop(request_id, None)
            output = RequestOutput(request_id=request_id, finished=True, finish_reason='abort')
            output.recipe_events, output.recipe_response = (), None
            self._completed.append(output)
        else:
            self._retire(request, cancelled=True)
        return True

    def step(self):
        # No generic cache-corruption recovery/re-prefill: failed mutations burn.
        # The qualified physical adapter is subordinate to this Scheduler and
        # operates only its batch; it has no independent queues or admission.
        output = SchedulerOutput()
        self._process_pending_aborts()
        if self.backend.fatal_error:
            for rid in list(self.requests):
                self._retire(self.requests[rid], error=RuntimeError(self.backend.fatal_error))
        else:
            scheduled, _ = self._schedule_waiting()
            output.scheduled_request_ids = [r.request_id for r in scheduled]
            for request in list(self.running.values()):
                rec = request.execution
                try:
                    self.backend._advance_application(rec, request.trace)
                    if request.trace.get('application_terminal'):
                        self._retire(request)
                    else:
                        events = rec.pending_delivery[request.delivery_cursor:]
                        request.delivery_cursor = len(rec.pending_delivery)
                        item = RequestOutput(request_id=request.request_id)
                        item.recipe_events, item.recipe_response = events, None
                        output.outputs.append(item)
                except BaseException as exc:
                    self._retire(request, error=exc)
        output.outputs.extend(self._completed)
        self._completed.clear()
        output.finished_request_ids = {r.request_id for r in output.outputs if r.finished}
        output.has_work = bool(output.outputs or self.waiting or self.running)
        return output

    def close_serving(self):
        for rid in list(self.requests):
            self._do_abort_request(rid)
        self.checkpoints.clear()
        self.shutdown()


class ProductionMTPBackend(InternalMTPQualificationBackend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scheduler = None
        self._deliveries = {}
        self._settled = {}
        self._pump_task = None
        self.serving_traces = self.session_traces

    @staticmethod
    def validate_ordinary(raw):
        from ds41f_mlx.mtp_profile import strict_json, validate_chat, LIMITS
        if len(raw) > LIMITS['body_bytes']:
            raise ValueError('body limit exceeded')
        body = strict_json(raw)
        # Ordinary OpenAI defaults, without widening the qualified execution
        # controls. Recipe still converts the ORIGINAL request below.
        validation = dict(body)
        validation.setdefault('max_tokens', 128)
        stream_options = validation.pop('stream_options', None)
        if stream_options is not None and stream_options != {'include_usage': True}:
            raise ValueError('unsupported stream_options')
        validate_chat(json.dumps(validation).encode(), ordinary_tools=True)
        # Use the authoritative request converter, not a serving-layer tool or
        # JSON Schema validator. Malformed declarations/choices are client errors
        # even when this boundary is invoked without HTTP preparation.
        from deepseek_recipe import ChatCompletionRequest, ConversionOptions
        ChatCompletionRequest(raw).convert(ConversionOptions(default_thinking_mode=False))

    def _enqueue(self, prepared, tokenizer, request_id):
        if self.fatal_error:
            raise RuntimeError(self.fatal_error)
        self.load()
        if self.scheduler is None:
            self.scheduler = ProductionScheduler(self)
        request = Request(request_id=request_id, prompt=list(prepared.token_ids),
                          sampling_params=SamplingParams(max_tokens=self.max_tokens(prepared.inference_options), temperature=0))
        request.prepared, request.recipe_tokenizer = prepared, tokenizer
        request.execution = QualificationSession(request_id)
        request.prompt_checkpoint, request.delivery_cursor = None, 0
        request.trace = dict(response_id='chatcmpl-'+request_id, t0=perf_counter(), generated=0,
                             decode_s=0., first_canonical_s=None, canonical_emitted=0)
        self.scheduler.add_request(request)

    async def _pump(self):
        try:
            while True:
                result = await self._call(self.scheduler.step)
                for output in result.outputs:
                    queue = self._deliveries.get(output.request_id)
                    if queue is not None:
                        queue.put_nowait(output)
                    if output.finished:
                        event = self._settled.get(output.request_id)
                        if event is not None:
                            event.set()
                        if hasattr(output, 'serving_trace'):
                            self.serving_traces.append(output.serving_trace)
                if not self.scheduler.requests:
                    break
                await asyncio.sleep(0)
        finally:
            self._pump_task = None

    async def _cancel(self, request_id):
        if self.scheduler is not None:
            # _call waits for any in-flight mutation before queuing the abort.
            with CancelScope(shield=True):
                await self._call(self.scheduler.abort_request, request_id)
                if self._pump_task is None and self.scheduler.requests:
                    self._pump_task = asyncio.create_task(self._pump())
                event = self._settled.get(request_id)
                if event is not None and request_id in self.scheduler.requests:
                    await event.wait()
        self._deliveries.pop(request_id, None)
        self._settled.pop(request_id, None)

    async def ordinary_response(self, prepared, *, tokenizer, http_request=None):
        from .server import InferenceStreamingResponse, sse_frame
        if prepared.protocol != 'chat_completions' or prepared.image_sources:
            raise ValueError('ordinary MTP supports text Chat Completions only')
        limit = self.max_tokens(prepared.inference_options)
        if not 1 <= limit <= 768 or not 3 <= len(prepared.token_ids) or len(prepared.token_ids)+limit > 8192:
            raise ValueError('ordinary MTP bounded to 8192 total / 768 output tokens')
        rid = uuid4().hex
        queue = self._deliveries[rid] = asyncio.Queue()
        self._settled[rid] = asyncio.Event()
        try:
            with CancelScope(shield=True):
                await self._call(self._enqueue, prepared, tokenizer, rid)
        except BaseException:
            await self._cancel(rid)
            raise
        if self._pump_task is None:
            self._pump_task = asyncio.create_task(self._pump())
        completed = False
        # JSON responses do not have StreamingResponse's receive/close loop.
        # Use Starlette's request-scoped disconnect probe for both transports;
        # either the probe or streaming iterator cleanup settles the SAME ID.
        watcher_stopped = False
        async def watch_disconnect():
            while not watcher_stopped and rid in self._deliveries:
                if await http_request.is_disconnected():
                    await self._cancel(rid)
                    return
                await asyncio.sleep(0.05)
        watcher = asyncio.create_task(watch_disconnect()) if http_request is not None else None
        async def stop_watcher():
            nonlocal watcher_stopped
            # Request.is_disconnected uses a cancelling AnyIO scope and can
            # swallow task.cancel(). A stop predicate prevents cleanup from
            # waiting forever on a probe whose cancellation was consumed.
            watcher_stopped = True
            if watcher is not None:
                watcher.cancel()
                with CancelScope(shield=True):
                    await asyncio.gather(watcher, return_exceptions=True)
        async def chunks():
            nonlocal completed
            try:
                while True:
                    output = await queue.get()
                    for event in output.recipe_events:
                        yield sse_frame(None, event)
                    if output.finished:
                        completed = True
                        if output.error:
                            raise RuntimeError(output.error)
                        yield sse_frame(None, '[DONE]')
                        break
            finally:
                if not completed:
                    await self._cancel(rid)
                await stop_watcher()
                self._deliveries.pop(rid, None)
                self._settled.pop(rid, None)
        if prepared.stream:
            backend = self
            class OwnedResponse(InferenceStreamingResponse):
                async def __call__(self, scope, receive, send):
                    try:
                        await super().__call__(scope, receive, send)
                    finally:
                        if not completed:
                            await backend._cancel(rid)
                        await stop_watcher()
            return OwnedResponse(chunks(), media_type='text/event-stream')
        try:
            while True:
                output = await queue.get()
                if output.finished:
                    completed = True
                    if output.error:
                        raise RuntimeError(output.error)
                    return JSONResponse(output.recipe_response)
        finally:
            if not completed:
                await self._cancel(rid)
            await stop_watcher()
            self._deliveries.pop(rid, None)
            self._settled.pop(rid, None)

    async def shutdown_serving(self):
        with CancelScope(shield=True):
            for rid in list(self._deliveries):
                await self._cancel(rid)
            if self.scheduler is not None:
                await self._call(self.scheduler.close_serving)
                self.scheduler = None
        self.close()
