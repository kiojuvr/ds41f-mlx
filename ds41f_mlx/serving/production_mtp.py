"""Default ordinary serving over the pinned Scheduler and qualified MTP core.

Scheduler queues/IDs/abort admission remain upstream. Only its model-specific
external-prefill, decode, and completion hooks differ: qualified P5/recipe
settlement replaces generic prefill/parser/backend-finish cache publication.
"""
import asyncio
import logging
from time import perf_counter, monotonic
from uuid import uuid4

from anyio import CancelScope
from fastapi.responses import JSONResponse
from omlx.scheduler import Scheduler, SchedulerConfig, SchedulerOutput
from omlx.request import Request, RequestOutput, RequestStatus, SamplingParams

from .internal_mtp import InternalMTPQualificationBackend, QualificationSession
from .paired_checkpoint import PairedCheckpointAuthority
from .capacity import context_envelope, validate_ordinary_capacity
from ds41f_mlx.observability import phase, metrics

logger = logging.getLogger('ds41f.operator.settlement')


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
        from .production_sessions import ProductionSessions
        # One pin per retained logical conversation plus two publication slots.
        # All pairs remain in the SAME upstream refcounted authority.
        self.checkpoints = PairedCheckpointAuthority(backend.dependency_identity, mx,
            retained_checkpoints=ProductionSessions.MAX_LIVE+2, context_tokens=backend.context_tokens)
        self.paged_cache_manager = self.checkpoints.paged
        self._completed = []
        self.last_settlement = None
        self.session_leases = {}

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
            request.prompt_checkpoint = self.checkpoints.capture(target, rings, tokens, request.image_identities)
        try:
            phase(self.backend, trace, 'CACHE_LOOKUP')
            logical = getattr(request, 'logical_session', None)
            # Ordinary prefix hints are bounded and may forget a pinned root
            # tail. Sessions restore the SAME authority's exact upstream lease,
            # never a longer ordinary hit or replay from a shorter hint/miss.
            observe = lambda name: phase(self.backend, trace, name)
            admit = None
            mm = request.prepared.multimodal
            if mm is not None:
                from ds41f_mlx.runtime.multimodal import MAX_REQUEST_PATCHES
                if sum(s.grid[0] * s.grid[1] for s in mm.spans) > MAX_REQUEST_PATCHES:
                    def admit(checkpoint):
                        from .paired_checkpoint import (pair_bytes, projected_pair_bytes,
                            PATCH_DISCOUNT_PAIR_BYTES, PATCH_DISCOUNT_ROOT_BYTES)
                        from ds41f_mlx.execution_identity import numerical_profile
                        if numerical_profile() != 'flash-1017-pre-woa-fp8-v1':
                            raise ValueError('aggregate image patch budget exceeded (unqualified numerical profile)')
                        C = len(checkpoint.tokens)
                        mm.validate_prefix(C, list(checkpoint.images))
                        if C < request.required_image_frontier:
                            raise ValueError('historical image pair unavailable; image replay/re-encode forbidden')
                        mm.validate_workload(C)
                        # Include the pinned five-position speculative block,
                        # not only the canonical output reservation.
                        end = len(request.prompt_token_ids) + self.backend.request_max_tokens(request.prepared) + 5
                        roots = sum(pair_bytes(c) for c in self.checkpoints._payloads.values())
                        projected = projected_pair_bytes(checkpoint, end)
                        if roots > PATCH_DISCOUNT_ROOT_BYTES or projected > PATCH_DISCOUNT_PAIR_BYTES:
                            raise ValueError('aggregate image patch budget exceeded (paired-byte qualification fallback)')
                        trace.update(historical_patch_discount=True, retained_pair_bytes=roots,
                                     projected_pair_bytes=projected)
            if logical is not None and logical.canonical:
                restored = self.checkpoints.acquire_retained(self.session_leases.get(logical.session_id),
                    request.prompt_token_ids, timings=trace, observe=observe, images=request.image_identities, admit=admit)
            else:
                restored = self.checkpoints.acquire(request.prompt_token_ids, timings=trace, observe=observe,
                                                    images=request.image_identities, admit=admit)
            if restored is not None:
                rec.cache, rec.rings, rec.canonical = restored
            logical = getattr(request, 'logical_session', None)
            if logical is not None and logical.canonical and tuple(rec.canonical) != logical.canonical:
                raise RuntimeError('canonical session pair unavailable or mismatched; prompt replay forbidden')
            if request.prepared.multimodal is not None:
                request.prepared.multimodal.validate_prefix(len(rec.canonical),
                    [s for s in request.image_identities if s['start'] < len(rec.canonical)])
                if len(rec.canonical) < request.required_image_frontier:
                    raise ValueError('historical image pair unavailable; image replay/re-encode forbidden')
                request.prepared.multimodal.validate_workload(len(rec.canonical))
            request.cached_tokens = trace['cached_tokens'] = len(rec.canonical)
            request.remaining_tokens = request.prompt_token_ids[len(rec.canonical):]
            trace['remaining_suffix_tokens'] = len(request.remaining_tokens)
            metrics(self.backend, trace)
            request.native_started = True
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
        finally:
            if request.prepared.multimodal is not None:
                request.prepared.multimodal.release_sources()

    def _retire(self, request, *, error=None, cancelled=False):
        rec, trace = request.execution, request.trace
        # This worker has no remaining image-input consumer, including when a
        # fatal engine retires requests still waiting for cache admission.
        if request.prepared.multimodal is not None:
            request.prepared.multimodal.release_sources()
        publication = False
        phase(self.backend, trace, 'SETTLING')
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
                    # extend this frontier with a genuine append, P5 holdout
                    # and at least one output token. Keep the earlier prompt
                    # pair instead. This is extendability, not an 8K limit.
                    final = (self.checkpoints.capture(rec.cache, rec.rings, rec.canonical, request.image_identities)
                             if len(rec.canonical) + 3 <= self.checkpoints.context_tokens else None)
                    if request.prompt_checkpoint is not None:
                        self.checkpoints.publish(request.prompt_checkpoint)
                    if final is not None:
                        self.checkpoints.publish(final)
                    publication = True
                    logical = getattr(request, 'logical_session', None)
                    if logical is not None and final is not None:
                        lease = self.checkpoints.retain(rec.canonical, final.images)
                        previous = self.session_leases.get(logical.session_id)
                        self.session_leases[logical.session_id] = lease
                        if previous is not None:
                            self.checkpoints.release(previous)
                        request.session_pair_retained = True
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
            if error is not None and request.native_started:
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
            response = trace.get('response') or {}
            if request.prepared.protocol == 'messages':
                semantic_finish = response.get('stop_reason') or 'end_turn'
                length_capped = semantic_finish == 'max_tokens'
            else:
                choices = response.get('choices') or []
                semantic_finish = (choices[0].get('finish_reason') if choices else None) or 'stop'
                length_capped = semantic_finish == 'length'
            request.status = (RequestStatus.FINISHED_ABORTED if cancelled else
                              RequestStatus.FINISHED_LENGTH_CAPPED if length_capped else
                              RequestStatus.FINISHED_STOPPED)
            request.output_token_ids = list(trace.get('canonical_generated', ()))
            self.total_completion_tokens += len(request.output_token_ids)
            trace.update(cancelled=cancelled, cache_published=publication,
                         elapsed_s=perf_counter()-trace['t0'])
            self.last_settlement = trace
            if getattr(request, 'logical_session', None) is not None:
                if rec.certificate is not None:
                    rec.certificate.pop('witness', None)  # no diagnostic second history owner
                try:
                    self.backend.production_sessions.publish(request, error)
                except BaseException as exc:
                    error = exc
                    self.backend.fatal_error = repr(exc)
                    self.checkpoints.clear()
                    self.backend.production_sessions.publish(request, exc)
            metrics(self.backend, trace)
            projection = getattr(self.backend, 'telemetry', None)
            if projection is not None:
                try:
                    projection.cache_summary(self.checkpoints.paged.get_stats().to_dict())
                except Exception:
                    pass
                projection.finish(trace.get('diagnostic_id'),
                                  'error' if error else 'abort' if cancelled else semantic_finish)
                if error:
                    projection.lifecycle('FAILED')
            try:
                logger.info('MTP settled request=%s cached=%d canonical=%s published=%s cancelled=%s error=%s replay=%s repack=%s',
                            request.request_id, request.cached_tokens, trace.get('canonical_frontier'),
                            publication, cancelled, type(error).__name__ if error else None,
                            trace.get('prompt_replay'), trace.get('full_cache_repack'))
            except Exception:
                pass  # logging is never settlement or delivery authority
            output = RequestOutput(request_id=request.request_id, finished=True,
                                   finish_reason='error' if error else 'abort' if cancelled else semantic_finish,
                                   error=str(error) if error else None)
            if request.prepared.protocol == 'messages' and (cancelled or rec.poisoned):
                output.error = str(error or trace.get('recovery_error') or 'Messages request cancelled')
            output.recipe_events = (() if output.error or cancelled else rec.pending_delivery[request.delivery_cursor:])
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
            if request.prepared.multimodal is not None:
                request.prepared.multimodal.release_sources()
            self.waiting = type(self.waiting)(r for r in self.waiting if r.request_id != request_id)
            self.requests.pop(request_id, None)
            projection = getattr(self.backend, 'telemetry', None)
            if projection is not None:
                projection.finish(request.trace.get('diagnostic_id'), 'abort')
            output = RequestOutput(request_id=request_id, finished=True, finish_reason='abort')
            output.recipe_events, output.recipe_response = (), None
            if getattr(request, 'logical_session', None) is not None:
                self.backend.production_sessions.unstarted(request.logical_session, request_id,
                                                          'cancelled before native admission; no committed turn')
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
                        end = len(rec.pending_delivery)
                        if request.prepared.protocol == 'messages':
                            # A tool block stop is not effect authority. Hold the
                            # tool suffix until the SAME worker canonically settles.
                            import json
                            for i in range(request.delivery_cursor, end):
                                event = json.loads(rec.pending_delivery[i])
                                if event.get('content_block', {}).get('type') == 'tool_use':
                                    end = i
                                    break
                        events = rec.pending_delivery[request.delivery_cursor:end]
                        request.delivery_cursor = end
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

    def release_session(self, session_id):
        lease = self.session_leases.pop(session_id, None)
        if lease is not None:
            self.checkpoints.release(lease)

    def close_serving(self):
        for rid in list(self.requests):
            self._do_abort_request(rid)
        self.session_leases.clear()
        self.checkpoints.clear()
        self.shutdown()


class ProductionMTPBackend(InternalMTPQualificationBackend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scheduler = None
        self._deliveries = {}
        self._settled = {}
        self._pump_task = None
        self._retirement_task = None
        self.serving_traces = self.session_traces
        from .production_sessions import ProductionSessions
        self.production_sessions = ProductionSessions(self)
        self.sessions = self.production_sessions.records

    def _encode_new_images(self, multimodal, frontier):
        import mlx.core as mx
        from omlx.patches.deepseek_v41.model import Model
        model = self._model
        if type(model) is not Model or not model.config.preserve_mtp:
            raise RuntimeError('Vision requires the admitted pinned MTP Model')
        def encode_span(patches, height, width, kinds):
            # Only this NEW span is embedded; no language forward, prompt
            # replay, cache mutation, or second model/weight owner is involved.
            ids = mx.full((1, len(kinds)), model.config.image_token_id, dtype=mx.int32)
            return model.get_input_embeddings(ids, pixel_values=patches,
                image_grids=[(height, width)], image_spans=[(0, len(kinds))],
                image_types=[list(kinds)]).inputs_embeds[0]
        return multimodal.encode_new(model, frontier, span_encoder=encode_span)

    def _prefill_tap_start(self, ids, frontier, rings):
        from ds41f_mlx.prefill_fp8_mlx.p6_append import P6_FINAL_TAIL_THRESHOLD
        capacity = rings[0].max_size
        if any(r.max_size != capacity for r in rings):
            raise RuntimeError('incompatible DSpark ring capacities')
        prompt_capture = len(ids) - 2
        # Reuse qualified DwarfStar for the bulk, then an ordinary completing
        # append for every row needed by the bounded physical DSpark rings.
        # No target token is run twice and no long late-layer tensor is retained.
        return prompt_capture - capacity if prompt_capture - frontier >= P6_FINAL_TAIL_THRESHOLD else frontier

    @property
    def context_tokens(self):
        return context_envelope(self.checkpoint)

    @staticmethod
    def validate_ordinary(raw):
        from .ordinary_admission import convert_ordinary
        converted, _, _ = convert_ordinary(raw)
        return converted

    def _enqueue(self, prepared, tokenizer, request_id, logical_session=None, body=None, application_id=None):
        if self.fatal_error:
            raise RuntimeError(self.fatal_error)
        self.load()
        if self.scheduler is None:
            self.scheduler = ProductionScheduler(self)
        from .generation_policy import policy_for
        policy = policy_for(prepared)
        request = Request(request_id=request_id, prompt=list(prepared.token_ids),
                          sampling_params=policy.sampling_params(self.request_max_tokens(prepared)))
        request.prepared, request.recipe_tokenizer = prepared, tokenizer
        request.native_started = False
        request.image_identities = () if prepared.multimodal is None else tuple(prepared.multimodal.identities())
        # Prior-turn images require retained executable state, never replay.
        origins = prepared.image_is_historical
        if origins is None:
            if prepared.protocol == 'messages' and request.image_identities:
                raise ValueError('Messages image source origin unavailable')
            messages = prepared.conversation_request.conversation.messages
            historical = sum(len(m.image_sources) for m in messages[:-1])
            origins = tuple(i < historical for i in range(len(request.image_identities)))
        request.required_image_frontier = max(
            (s['start'] + s['length'] for s, historical in
             zip(request.image_identities, origins, strict=True) if historical), default=0)
        request.execution = QualificationSession(request_id)
        request.logical_session = logical_session
        request.session_pair_retained = False
        if logical_session is not None:
            request.session_body, request.application_id = body, application_id
            request.execution.reconstruction_body = body
            request.execution.reconstruction_tokenizer = tokenizer
        request.prompt_checkpoint, request.delivery_cursor = None, 0
        request.trace = dict(response_id='chatcmpl-'+request_id,
                             t0=getattr(prepared, 'arrival_t0', perf_counter()), generated=0,
                             decode_s=0., first_canonical_s=None, canonical_emitted=0)
        request.trace.update(getattr(prepared, 'phase_timings', {}))
        request.trace['generation_policy'] = policy.to_json()
        request.trace['diagnostic_id'] = getattr(prepared, 'diagnostic_id', None)
        self.scheduler.add_request(request)
        # Projection only: logical requests wait on the existing Scheduler lane.
        phase(self, request.trace, 'QUEUED')

    async def _pump(self):
        try:
            while True:
                result = await self._call(self.scheduler.step)
                for output in result.outputs:
                    queue = self._deliveries.get(output.request_id)
                    if queue is not None:
                        queue.put_nowait(output)
                    if output.finished:
                        for rec in self.sessions.values():
                            if rec.active_request_id == output.request_id:
                                rec.active_request_id = None
                                rec.touched_at = monotonic()
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

    async def ordinary_response(self, prepared, *, tokenizer, http_request=None,
                                request_id=None, logical_session=None, body=None, application_id=None):
        from .server import InferenceStreamingResponse, sse_frame
        if prepared.protocol not in ('chat_completions', 'messages'):
            raise ValueError('ordinary MTP supports Chat Completions and Messages only')
        if logical_session is not None and prepared.protocol != 'chat_completions':
            raise ValueError('production sessions support Chat Completions only')
        if prepared.image_sources and prepared.multimodal is None:
            raise ValueError('image input was not prepared')
        if getattr(prepared, 'multimodal', None) is not None and any(
                s.start + s.length > len(prepared.token_ids)-2 for s in prepared.multimodal.spans):
            raise ValueError('Vision requires two terminal text tokens for paired P5 handoff')
        limit = self.request_max_tokens(prepared)
        envelope = (context_envelope(self.checkpoint, multimodal=True, ordinary=True)
                    if getattr(prepared, 'multimodal', None) is not None else self.context_tokens)
        validate_ordinary_capacity(len(prepared.token_ids), limit, envelope)
        if len(prepared.token_ids) < 3:
            raise ValueError('encoded prompt must contain at least 3 tokens')
        rid = request_id or uuid4().hex
        queue = self._deliveries[rid] = asyncio.Queue()
        self._settled[rid] = asyncio.Event()
        try:
            with CancelScope(shield=True):
                if logical_session is None:
                    await self._call(self._enqueue, prepared, tokenizer, rid)
                else:
                    await self._call(self._enqueue, prepared, tokenizer, rid, logical_session, body, application_id)
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
                    if output.error:
                        completed = output.finished
                        raise RuntimeError(output.error)
                    for event in output.recipe_events:
                        if prepared.protocol == 'messages':
                            import json
                            name = json.loads(event)['type']  # Native serialized event, unchanged.
                        else:
                            name = None
                        yield sse_frame(name, event)
                    if output.finished:
                        completed = True
                        from deepseek_recipe import ChatCompletionResponse, MessagesResponse
                        response_type = MessagesResponse if prepared.protocol == 'messages' else ChatCompletionResponse
                        done = response_type.sse_done_data()
                        if done is not None:
                            yield sse_frame(None, done)
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
            return OwnedResponse(chunks(), media_type='text/event-stream', headers={
                'X-DS41F-Request-ID': rid, 'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})
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

    def start_idle_retirement(self):
        if self._retirement_task is None:
            self._retirement_task = asyncio.create_task(self._retire_idle())

    async def _retire_idle(self):
        while True:
            await asyncio.sleep(30)
            await self.retire_idle_sessions()

    async def retire_idle_sessions(self):
        for sid in self.production_sessions.expired():
            if sid not in self.production_sessions.expired():
                continue  # A heartbeat may have arrived while another pin released.
            # Expiry observes logical inactivity only. Never cancel active/queued
            # work or evict a pin while it owns a canonical settlement.
            try:
                await self.close_stateful_session(sid)
            except (KeyError, RuntimeError):
                pass

    async def create_stateful_session(self, *, session_id=None):
        await self.retire_idle_sessions()
        return self.production_sessions.create(session_id)

    def get_stateful_session(self, session_id):
        return self.production_sessions.get(session_id)

    async def close_stateful_session(self, session_id):
        rec = self.get_stateful_session(session_id)
        if rec.busy:
            raise RuntimeError('session has an active request; cancel/reconcile before close')
        # Reserve close against admission while releasing the sole Scheduler pin.
        rec.closing = True
        try:
            with CancelScope(shield=True):
                if self.scheduler is not None:
                    await self._call(self.scheduler.release_session, session_id)
                rec.closed = True
                result = rec.to_json()
                result['active_request_id'] = None
                del self.sessions[session_id]
                return result
        finally:
            rec.closing = False

    async def production_session_response(self, session_id, prepared, *, tokenizer, body,
                                          expected_count, application_id, http_request=None):
        import json
        if prepared.protocol != 'chat_completions':
            raise ValueError('production sessions support Chat Completions only')
        if not prepared.stream:
            raise ValueError('production session Chat Completions requires streaming; no legacy turn path')
        payload = json.loads(body)
        rid = uuid4().hex
        rec = self.production_sessions.reserve(session_id, prepared, payload,
                                               expected_count, application_id, rid)
        try:
            return await self.ordinary_response(prepared, tokenizer=tokenizer, http_request=http_request,
                request_id=rid, logical_session=rec, body=payload, application_id=application_id)
        except BaseException as exc:
            if self.scheduler is not None and rid in self.scheduler.requests:
                # Failed transport cleanup is NOT proof of non-admission. Leave
                # the active identity fenced until Scheduler proves retirement.
                rec.publication = dict(rec.publication, state='unrecoverable', last_error=str(exc))
            else:
                self.production_sessions.unstarted(rec, rid, exc)
            raise

    async def cancel_stateful_stream(self, session_id, request_id):
        rec = self.get_stateful_session(session_id)
        if rec.active_request_id != request_id:
            if (rec.publication['last_turn'] or {}).get('request_id') == request_id:
                return dict(request_id=request_id, completed=True, cancelled=False)
            raise RuntimeError('active request identity mismatch')
        await self._cancel(request_id)
        return dict(request_id=request_id, state=rec.to_json()['state'])

    async def qualification_response(self, *args, **kwargs):
        raise RuntimeError('qualification execution is not a production capability')

    async def run_stateful_chat_turn(self, *args, **kwargs):
        raise RuntimeError('legacy stateful execution is not a production capability')

    async def open_stateful_stream(self, *args, **kwargs):
        raise RuntimeError('standard-OFF execution is not a production capability')

    async def shutdown_serving(self):
        if self._retirement_task is not None:
            self._retirement_task.cancel()
            await asyncio.gather(self._retirement_task, return_exceptions=True)
            self._retirement_task = None
        with CancelScope(shield=True):
            for rid in list(self._deliveries):
                await self._cancel(rid)
            for sid in list(self.sessions):
                await self.close_stateful_session(sid)
            if self.scheduler is not None:
                await self._call(self.scheduler.close_serving)
                self.scheduler = None
        self.close()
