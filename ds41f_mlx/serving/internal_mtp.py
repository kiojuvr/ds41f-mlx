"""M35 qualification-only injection. Not a release selector or public API option.

Uses the actual recipe server routes, but only when an experiment explicitly
constructs this backend. One logical session and one response owner. Socket send
is deliberately NOT a CanonicalTransportHistory acknowledgement.
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
import json
from time import perf_counter, time
from typing import Any
from uuid import uuid4

from anyio import CancelScope
from anyio.lowlevel import checkpoint

from .deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend, MODEL_ALIASES


@dataclass
class QualificationSession:
    session_id: str
    cache: Any = None
    rings: Any = None
    canonical: list[int] = field(default_factory=list)
    owner: Any = None
    processor: Any = None
    guard: Any = None
    busy: bool = False
    closed: bool = False
    poisoned: bool = False
    request_count: int = 0
    last_turn: Any = None
    unrecoverable: bool = False
    certificate: Any = None
    fence: Any = None
    consumed_sequence: int = 0
    reconstruction_body: Any = None
    reconstruction_tokenizer: Any = None
    # Sole in-process delivery owner; socket progress is not canonical ACK.
    # Serialized before a worker completion can be lost to cancellation.
    pending_delivery: tuple[str, ...] = ()

    def to_json(self):
        return dict(id=self.session_id, state='closed' if self.closed else
                    'busy' if self.busy else 'poisoned' if self.poisoned else
                    'idle' if self.cache is not None else 'empty',
                    request_count=self.request_count, last_turn=self.last_turn,
                    canonical_frontier=len(self.canonical),
                    outcome_state=self.fence['state'] if self.fence and (self.last_turn or {}).get('certified_outcome') and self.fence['state'] not in ('active', 'not_admitted') else 'active' if self.busy else 'not_admitted' if self.fence and self.fence['state'] == 'not_admitted' else 'poisoned' if self.poisoned and not self.unrecoverable else
                    'unrecoverable' if self.unrecoverable else 'recoverable' if self.certificate and self.certificate['representable'] else 'not_admitted',
                    certificate=self.certificate, request_fence=self.fence,
                    next_sequence=self.consumed_sequence + 1)


class InternalMTPQualificationBackend(DeepSeekRecipeRuntimeBackend):
    """Explicit object injection only; no env, request or release MTP selector."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_live_sessions = 1
        self._namespace = uuid4().hex
        self._issued = 0
        self.retired_diagnostics = deque(maxlen=16)
        self.traces = deque(maxlen=32)
        self.session_traces = deque(maxlen=32)
        # Experiment controller only, never read from a request.
        self.delivery_delay_s = 0.0
        self.progress = None

    def load(self):
        if self._model is not None:
            return
        import omlx.scheduler  # install ordinary native patches
        from omlx.patches.deepseek_v41.loading import load
        from omlx.patches.mlx_lm_mtp import batch_generator, cache_rollback
        assert batch_generator.apply() and cache_rollback.apply()
        self._model, self._execution_tokenizer = load(self.checkpoint, preserve_mtp=True, engram_ssd_offload=True)
        self._model.language_model.configure_mtp(True, 5)
        self._model.language_model._p7_enable_overlap = True

    async def infer(self, request):
        raise ValueError('internal MTP qualification requires singleton stateful Chat Completions')
        yield  # async iterator contract

    MAX_LIFETIMES = (1 << 128) - 1
    MAX_SEQUENCE = (1 << 64) - 1
    MAX_BODY_BYTES = 1048576

    def identity_state(self, session_id):
        """Derive admission without consulting optional diagnostics or history."""
        if not isinstance(session_id, str) or len(session_id) != 69:
            return 'never_valid'
        parts = session_id.split('_')
        if len(parts) != 3 or parts[0] != 'mtp' or any(
                len(p) != 32 or any(c not in '0123456789abcdef' for c in p)
                for p in parts[1:]):
            return 'never_valid'
        if parts[1] != self._namespace:
            return 'stale_namespace'
        serial = int(parts[2], 16)
        if not 1 <= serial <= self._issued:
            return 'never_valid'
        return 'live' if session_id in self.sessions else 'retired'

    def get_stateful_session(self, session_id):
        state = self.identity_state(session_id)
        if state != 'live':
            raise KeyError(f'{state} session identity')
        return self.sessions[session_id]

    async def create_stateful_session(self, *, session_id=None):
        if session_id is not None:
            raise ValueError('internal session IDs are server-issued; requested IDs unsupported')
        if self.sessions or self._lock.locked():
            raise RuntimeError('maximum live session count reached')
        if self._issued == self.MAX_LIFETIMES:
            raise RuntimeError('process lifetime namespace exhausted; no wrap permitted')
        serial = self._issued + 1
        sid = f'mtp_{self._namespace}_{serial:032x}'
        rec = QualificationSession(sid)
        self.sessions[sid] = rec
        self._issued = serial
        return rec

    async def persist_stateful_session(self, *args, **kwargs):
        raise RuntimeError('internal MTP persistence unsupported before I/O')

    async def restore_stateful_session(self, *args, **kwargs):
        raise RuntimeError('internal MTP restore unsupported before mutation or I/O')

    async def close_stateful_session(self, session_id):
        rec = self.get_stateful_session(session_id)
        if rec.busy or self._lock.locked():
            raise RuntimeError('session already has an active request')
        previous_outcome = rec.to_json()['outcome_state']
        await self._lock.acquire()
        rec.busy = True
        try:
            with CancelScope(shield=True):
                await self._call(self._retire, rec)
                summary = dict(id=rec.session_id, state='closed',
                               request_count=rec.request_count,
                               canonical_frontier=len(rec.canonical),
                               outcome_state=previous_outcome)
                rec.closed = True
                # Identity correctness is the issuance invariant, not this record.
                del self.sessions[session_id]
                self.retired_diagnostics.append(summary)
                rec.canonical.clear()
                rec.owner = rec.processor = rec.guard = None
                rec.cache = rec.rings = None
                rec.last_turn = rec.certificate = rec.fence = None
                rec.reconstruction_body = rec.reconstruction_tokenizer = None
                rec.pending_delivery = ()
                self.session_traces.clear()
                self.traces.clear()
                self.progress = self.last_trace = None
                return dict(summary)
        except BaseException:
            rec.unrecoverable = False
            rec.poisoned = True
            raise
        finally:
            rec.busy = False
            self._lock.release()

    def _retire(self, rec):
        import mlx.core as mx
        from mlx_lm.generate import generation_stream
        from omlx.patches.mlx_lm_mtp import prompt_priming
        with mx.stream(generation_stream):
            if rec.owner is not None:
                rec.owner.close()
                rec.owner = None
            if rec.processor is not None:
                rec.processor.close()
                rec.processor = None
            rec.cache = rec.rings = None
            if self._model is not None:
                prompt_priming.drop_ctx(self._model.language_model)
            mx.synchronize(generation_stream)
            mx.clear_cache()

    def _start(self, rec, request, tokenizer, trace, *, checkpoint_capture=None, batch_generator_factory=None):
        import deepseek_recipe as d
        import mlx.core as mx
        import numpy as np
        from mlx_lm.generate import generation_stream
        from ds41f_mlx.prefill_fp8_mlx import DeferredPrefillAppend, LivePrefillResult, handoff_to_generation
        from ds41f_mlx.runtime.mtp_lifecycle import DSparkCommittedContext, OMLXMTPGenerationSession
        from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
        from ds41f_mlx.runtime.recipe_semantic_guard import RecipeSemanticGuard
        from contextlib import ExitStack
        from ds41f_mlx.runtime.mtp_resources import MTPWiredLimitLease
        with mx.stream(generation_stream), ExitStack() as startup_resources:
            load_start = perf_counter()
            self.load()
            trace['load_s'] = perf_counter()-load_start
            lm = self._model.language_model
            if rec.cache is None:
                from omlx.patches.mlx_lm_mtp import prompt_priming
                prompt_priming.drop_ctx(lm)
                rec.cache, rec.rings = lm.make_cache(), lm.make_mtp_cache()
            ids = request.token_ids
            C = len(rec.canonical)
            # Already proved pre-admission on the event-loop thread. Recheck
            # immediately before mutation under the singleton worker lease.
            assert ids[:C] == rec.canonical and len(ids) > C
            taps = {i: [] for i in lm._config.dspark_target_layer_ids}
            original = {i: lm.layers[i] for i in taps}
            class Tap:
                def __init__(self, index, layer): self.index, self.layer = index, layer
                def __getattr__(self, name): return getattr(self.layer, name)
                def __call__(self, h, *args, **kwargs):
                    reduced = mx.mean(h, axis=-2)
                    if reduced.ndim == 2: reduced = reduced[None]
                    taps[self.index].append(reduced)
                    return self.layer(h, *args, **kwargs)
            t0 = perf_counter()
            # Acquire the same native generation resource before dense prefix
            # allocations. Include acquisition in request/handoff latency and
            # restore on every pre-transfer failure; no state is published here.
            wired = startup_resources.enter_context(MTPWiredLimitLease(mx, generation_stream))
            # The ordinary path retains a paired prompt checkpoint with one
            # prefill token still to execute. Exact reuse therefore needs no
            # recurrent trim and still reaches P5 through a real sealed append.
            ends = [len(ids)-1] if checkpoint_capture is None else [len(ids)-2, len(ids)-1]
            app = None
            try:
                for i, layer in original.items(): lm.layers[i] = Tap(i, layer)
                for end in ends:
                    if end > C:
                        if app is None:
                            app = DeferredPrefillAppend.create(lm, rec.cache, ids[:end], committed_frontier=C, mx=mx)
                        else:
                            app = DeferredPrefillAppend.continue_from_commit(lm, app.commit_certificate, ids[:end], mx=mx)
                        app.execute_all()
                        hidden = mx.concatenate([mx.concatenate(taps[i], axis=1) for i in taps], axis=-1)
                        lm.dspark_append_context(hidden, rec.rings, start_offset=C)
                        for values in taps.values(): values.clear()
                        C = end
                        mx.eval(*[c.keys for c in rec.rings]); mx.synchronize(generation_stream)
                    if checkpoint_capture is not None and end == len(ids)-2:
                        checkpoint_capture(rec.cache, rec.rings, ids[:end])
            finally:
                for i, layer in original.items(): lm.layers[i] = layer
            context = DSparkCommittedContext.from_native(rec.rings, frontier=len(ids)-1, target_layer_ids=lm._config.dspark_target_layer_ids)
            live = LivePrefillResult.from_committed(app.commit_certificate, prefix_token_ids=ids[:-1])
            live.dspark_committed_context = context
            generator = d.ChatCompletionRequest.chunk_generator(request.conversation_request, trace['response_id'], request.model or self.model_id)
            generator = generator.with_include_usage(request.include_usage)
            rec.processor = d.StreamProcessor(generator, request.conversation_request.parsing_options, tokenizer)
            rec.guard = RecipeSemanticGuard(rec.processor, control_token_ids=request.stop_token_ids,
                frontier=len(ids), response=d.ChatCompletionResponse(trace['response_id'], self.model_id, int(time()), 0, 0))
            rec.guard._push(d.InferenceChunk.ready(prompt_usage=d.PromptUsage(prompt_tokens=len(ids), prompt_cache_hit_tokens=trace.get('cached_tokens', 0))))
            cfg = OMLXDecodeConfig(omlx_path=self.omlx_path, checkpoint_path=self.checkpoint,
                                  preserve_mtp=True, speculation_enabled=True, stop_token_ids=request.stop_token_ids)
            class Factory:
                @classmethod
                def from_prefilled_cache(cls, model, cache, prefix, config, *, max_tokens, sampler):
                    return OMLXMTPGenerationSession(model, cache, np.asarray(prefix), context,
                        config=config, sampler=sampler, max_tokens=max_tokens, semantic_guard=rec.guard,
                        wired_limit_lease=wired, batch_generator_factory=batch_generator_factory,
                        canonical_sampling_policy='greedy' if request.inference_options.temperature in (None, 0) else None)
            rec.owner = handoff_to_generation(live, self._model, terminal_prompt_token=ids[-1], config=cfg,
                max_tokens=self.max_tokens(request.inference_options), sampler=self.make_sampler(request.inference_options), session_factory=Factory)
            trace.update(prefill_handoff_s=perf_counter()-t0, prompt_replay=rec.owner.prompt_replay_count,
                         full_cache_repack=app.final_execution.runner.full_cache_repack_count)

    def _next(self, rec, trace):
        import mlx.core as mx
        from mlx_lm.generate import generation_stream
        with mx.stream(generation_stream):
            t0 = perf_counter()
            token = rec.owner.next_token(transport_delivered=False)
            trace['decode_s'] += perf_counter()-t0
            trace['generated'] += token is not None
            if trace['first_canonical_s'] is None and token is not None:
                trace['first_canonical_s'] = perf_counter()-trace['t0']
            trace['canonical_emitted'] = len(rec.owner.history.canonical_generated_tokens)
            return token

    def _advance_application(self, rec, trace):
        """Publish consumed protocol input before awaited worker completion.

        No parser replay or executable state is introduced. Native phases remain
        owned by _next/_settle; delivery owns only immutable serialized events.
        A known terminal settles in this worker turn, before transport cancellation
        can win the event-loop checkpoint. Any serialization/settlement failure
        remains fail closed through the existing retirement path.
        """
        token = self._next(rec, trace)
        # The existing last_turn owner accumulates immutable call segments while
        # its request fence remains active. No task outcome/effect reservation is
        # published until full official response settlement.
        certificates = getattr(rec.guard, 'call_certificates', ())
        trace['call_certificates'] = tuple(dict(
            processor_identity=id(c.processor), lifetime=c.lifetime,
            revision=c.revision, consumed_frontier=c.frontier,
            choice=c.choice, index=c.index, call=json.loads(c.call),
            candidate_ids=c.candidate_ids, response_snapshot=c.response_snapshot,
            consuming_outcome=c.consuming_outcome) for c in certificates)
        rec.last_turn = trace
        new_events = rec.guard.events[len(rec.pending_delivery):]
        encoded = tuple(event.to_json() for event in new_events)
        rec.pending_delivery += encoded
        if rec.guard.finished or token is None:
            self._settle(rec, trace)
            trace['application_terminal'] = True
        return token

    def _settle(self, rec, trace):
        import deepseek_recipe as d
        import mlx.core as mx
        from mlx_lm.generate import generation_stream
        with mx.stream(generation_stream):
            t0 = perf_counter()
            quiet = rec.owner.quiesce()
            quiet.counters.require_m28_zeroes()
            rec.cache, rec.rings = quiet.target_cache, list(quiet.dspark_context.caches)
            rec.canonical = list(quiet.canonical_tokens)
            gb = rec.owner._bg._generation_batch
            h = getattr(gb, '_omlx_semantic_horizon', None)
            state = None if h is None else h.last_state
            assert h is None or h.pending is None
            assert state is None or not state.queue
            if state is not None:
                import dataclasses
                trace['mtp_stats'] = dataclasses.asdict(state.stats)
            # Finish native runtime ownership exactly once, before protocol
            # accumulation (which can itself fail). No second quiescence on a
            # drained/retired owner is permitted during exception cleanup.
            trace['quiescence'] = quiet.to_json()
            connection = rec.owner.connection
            trace['authority_connection'] = dict(lifetime=connection.lifetime,
                revision=connection.revision, disposition=connection.disposition,
                consumed_frontier=len(connection.consumed_tokens),
                emitted_frontier=rec.owner.history.canonical_frontier,
                queue_ahead=list(connection.queue_ahead),
                pending_prediction=connection.pending_prediction, rng_draws=connection.rng_draws)
            canonical_generated = list(rec.owner.history.canonical_generated_tokens)
            rec.owner.close(); rec.owner = None
            assert len(rec.cache) == 40 and all(c.size() == len(rec.canonical) for c in rec.cache)
            assert all(c.offset == len(rec.canonical) for c in rec.rings)
            # Native response.append transfers chunk ownership. Serialize the
            # evidence while recipe chunks still own their native payloads.
            # Settlement can drain committed queued input not yet observed by
            # the async iterator. Preserve that delivery too, before append
            # transfers native chunk ownership and invalidates their serializers.
            encoded_events = tuple(e.to_json() for e in rec.guard.events)
            rec.pending_delivery = encoded_events
            protocol_events = [json.loads(e) for e in encoded_events]
            response = getattr(rec.guard, 'response', None)
            if response is None:
                response = d.ChatCompletionResponse(trace['response_id'], self.model_id, int(time()), 0, 0)
                for event in rec.guard.events: response.append(event)
            trace.update(quiescence=quiet.to_json(), canonical_frontier=len(rec.canonical),
                         target_offsets=[c.size() for c in rec.cache], dspark_offsets=[c.offset for c in rec.rings],
                         terminal_matches=list(rec.guard.matches), response=json.loads(response.to_json()),
                         protocol_events=protocol_events,
                         canonical_generated=canonical_generated,
                         prediction_retired=True, queue_empty=True)
            self._qualify_settled_protocol(rec, trace)
            if getattr(rec, 'reconstruction_body', None) is not None:
                from .recovery_certificate import reconstruction_certificate
                completed = rec.guard.finished and rec.guard.tool_complete
                rec.certificate = reconstruction_certificate(rec.reconstruction_body, trace['response'], rec.canonical,
                    tokenizer=rec.reconstruction_tokenizer, recipe_path=self.recipe_path, completed_tool_block=completed,
                    options=getattr(self, 'conversion_options', None))
                trace['certificate'] = rec.certificate
                if not rec.certificate['representable']:
                    rec.unrecoverable = True
                    rec.poisoned = True  # legacy containment/retirement label, not internal-failure outcome
                    trace['recovery_error'] = 'no official-recipe reconstruction certificate; DELETE required'
            self._finalize_application_certificate(rec, trace)
            rec.processor.close(); rec.processor = None
            trace['cleanup_s'] = perf_counter()-t0
            # Single immutable application authority, published on the consuming
            # worker. Transport cleanup cannot change this known disposition.
            if rec.fence is not None:
                state = ('unrecoverable' if rec.unrecoverable else 'poisoned' if rec.poisoned
                         else 'recoverable' if rec.certificate and rec.certificate['representable'] else 'poisoned')
                trace['certified_outcome'] = json.dumps(dict(session_id=rec.session_id,
                    sequence=rec.fence['sequence'], body_sha256=rec.fence['body_sha256'],
                    outcome_state=state, certificate=rec.certificate, response=trace['response'],
                    canonical_events=list(rec.pending_delivery),
                    call_certificates=trace.get('call_certificates', ()),
                    metrics=self._application_metrics(trace)))
                rec.last_turn = trace
                rec.consumed_sequence = rec.fence['sequence']
                rec.fence['state'] = state

    @staticmethod
    def _application_metrics(trace):
        metrics = {k:trace.get(k) for k in ('generated', 'decode_s', 'prefill_handoff_s',
            'load_s', 'cleanup_s', 'prompt_replay', 'full_cache_repack')}
        metrics['elapsed_s'] = perf_counter() - trace['t0']
        metrics['aligned_idle'] = bool(trace.get('target_offsets')) and set(
            trace.get('target_offsets', []) + trace.get('dspark_offsets', [])) == {trace.get('canonical_frontier')}
        stats = trace.get('mtp_stats', {})
        metrics['considered_drafts'] = sum(stats.get('depth_drafted', []))
        metrics['accepted_drafts'] = sum(stats.get('depth_accepted', []))
        metrics['settlement'] = trace.get('quiescence', {}).get('counters')
        return metrics

    def _finalize_application_certificate(self, rec, trace):
        """Profile constraints run before the one application certification."""

    @staticmethod
    def _qualify_settled_protocol(rec, trace):
        # Native cache quiescence is not a complete protocol-result certificate.
        # The recipe can emit argument deltas before the DSML terminal becomes
        # canonical. Never advertise that partial call as an idle agent result,
        # even if its current arguments happen to be valid JSON. Only the native
        # recipe terminal/finish owner can establish a complete call.
        choices = trace['response'].get('choices', [])
        has_calls = any(c.get('message', {}).get('tool_calls') for c in choices)
        completed_block = getattr(rec.guard, 'tool_complete', False)
        if has_calls and not (rec.guard.finished and completed_block and
                              all(c.get('finish_reason') == 'tool_calls' for c in choices)):
            rec.poisoned = True
            trace['recovery_error'] = 'unfinished canonical tool protocol; ordinary continuation not qualified'

    async def qualification_response(self, session_id, request, *, tokenizer, body=None, sequence=None, outcome_projection=False):
        """Acquire before HTTP 200. Retain lease until generator cleanup, not send."""
        from fastapi.responses import JSONResponse
        from .server import InferenceStreamingResponse, sse_frame
        # Bounds checked before reservation, consumption or native authority.
        if body is not None and (not isinstance(body, bytes) or len(body) > self.MAX_BODY_BYTES):
            raise ValueError('internal request body exceeds 1 MiB or is not bytes')
        if sequence is not None and (type(sequence) is not int or not 1 <= sequence <= self.MAX_SEQUENCE):
            raise ValueError('internal request sequence outside fixed 64-bit budget')
        if request.protocol != 'chat_completions' or request.image_sources:
            raise ValueError('internal MTP supports text Chat Completions only')
        if request.model is not None and request.model not in MODEL_ALIASES:
            raise ValueError('unsupported model')
        limit = self.max_tokens(request.inference_options)  # validate before admission
        if limit > 768 or len(request.token_ids) + limit > 8192:
            raise ValueError('internal qualification bounded to 768 responses and 8192 total tokens')
        self.make_sampler(request.inference_options)
        rec = self.get_stateful_session(session_id)
        from .request_fence import observe_retry, reserve, finish
        retry = observe_retry(rec, sequence, body)
        if retry is not None:
            if request.stream and not outcome_projection and retry['outcome_state'] == 'recoverable':
                # Exact whole-delivery reprojection, not a cursor/ACK protocol.
                # Capture before any following request can replace the slot.
                retained = tuple(retry.get('canonical_events', rec.pending_delivery))
                if retained != rec.pending_delivery:
                    raise RuntimeError('conflicting retained canonical delivery')
                async def replay():
                    for event in retained:
                        yield sse_frame(None, event)
                    yield sse_frame(None, '[DONE]')
                return InferenceStreamingResponse(replay(), media_type='text/event-stream')
            return JSONResponse(content=retry)
        if rec.consumed_sequence >= self.MAX_SEQUENCE or rec.request_count >= self.MAX_SEQUENCE:
            raise RuntimeError('session request lifetime exhausted; DELETE required')
        if rec.poisoned:
            raise RuntimeError('session poisoned; DELETE required')
        if rec.busy or self._lock.locked():
            raise RuntimeError('singleton already has an active request')
        if request.token_ids[:len(rec.canonical)] != rec.canonical or len(request.token_ids) <= len(rec.canonical):
            raise ValueError('request must exactly extend retained canonical prefix')
        self._validate_result_reentry(rec, body, sequence)
        await self._lock.acquire()
        rec.busy = True
        reserve(rec, sequence, body)
        rec.pending_delivery = ()
        rec.reconstruction_body = json.loads(body) if body is not None else None
        rec.reconstruction_tokenizer = tokenizer
        trace = dict(session_id=session_id, response_id=uuid4().hex, t0=perf_counter(),
                     generated=0, decode_s=0., first_canonical_s=None, first_chunk_s=None,
                     formatting_s=0., delivery_wait_s=0., protocol_chunks_produced=0,
                     iterator_yields=0, canonical_emitted=0, rows=[], cancelled=False,
                     client_acknowledged_ordinal=0)
        self.progress = trace
        self.session_traces.append(trace)
        lease_released = False
        async def chunks():
            nonlocal lease_released
            cursor = 0
            settled = False
            try:
                if rec.fence is not None:
                    rec.fence['started'] = True
                with CancelScope(shield=True):
                    await self._call(self._start, rec, request, tokenizer, trace)
                # Native phases are shielded, but the response owner MUST admit
                # transport cancellation at the intervening coherent boundary.
                await checkpoint()
                while True:
                    with CancelScope(shield=True):
                        token = await self._call(self._advance_application, rec, trace)
                    settled = bool(trace.get('application_terminal'))
                    await checkpoint()
                    t0 = perf_counter()
                    events = rec.pending_delivery[cursor:]
                    cursor = len(rec.pending_delivery)
                    frames = [sse_frame(None, event) for event in events]
                    trace['formatting_s'] += perf_counter()-t0
                    trace['protocol_chunks_produced'] += len(frames)
                    terminal = settled
                    # The worker has already settled terminal ownership. Socket
                    # publication is only a projection of its retained outcome.
                    for frame in frames:
                        ordinal = trace['canonical_emitted']
                        row = dict(canonical_ordinal=ordinal, produced_s=perf_counter()-trace['t0'])
                        trace['rows'].append(row)
                        t0 = perf_counter()
                        if self.delivery_delay_s:
                            await asyncio.sleep(self.delivery_delay_s)
                        trace['delivery_wait_s'] += perf_counter()-t0
                        row['yield_s'] = perf_counter()-trace['t0']
                        trace['iterator_yields'] += 1
                        if trace['first_chunk_s'] is None: trace['first_chunk_s'] = row['yield_s']
                        yield frame
                    if terminal:
                        yield sse_frame(None, '[DONE]')
                        break
            except (GeneratorExit, asyncio.CancelledError):
                raise
            except BaseException:
                rec.unrecoverable = False
                # Protocol/serialization failures are fail closed too. They
                # cannot trigger a second runtime settlement or implicit resume.
                if not trace.get('certified_outcome'):
                    rec.poisoned = True
                raise
            finally:
                import sys
                failure = sys.exc_info()[1]
                if failure is not None:
                    trace['exit_exception'] = repr(failure)
                # asyncio cancellation may discard _call's return even though
                # its worker completed. Consult worker-published disposition,
                # never a local variable dependent on delivery of that return.
                settled = settled or bool(trace.get('application_terminal'))
                trace['cancelled'] = failure is not None or not settled or trace['iterator_yields'] < trace['protocol_chunks_produced']
                with CancelScope(shield=True):
                    try:
                        if not settled and rec.owner is not None:
                            await self._call(self._settle, rec, trace)
                    except BaseException as exc:
                        rec.unrecoverable = False
                        rec.poisoned = True
                        trace['cleanup_error'] = repr(exc)
                        raise
                    finally:
                        try:
                            # A start/forward failure without a recoverable
                            # certificate is fail closed, never advertised idle.
                            if rec.poisoned or (not settled and 'quiescence' not in trace):
                                rec.poisoned = True
                                await self._call(self._retire, rec)
                        except BaseException as exc:
                            rec.unrecoverable = False
                            rec.poisoned = True
                            trace['retirement_error'] = repr(exc)
                            raise
                        finally:
                            # Even failed retirement cannot strand admission.
                            # The poisoned logical singleton still blocks reuse.
                            trace['elapsed_s'] = perf_counter()-trace['t0']
                            rec.last_turn = trace
                            rec.request_count += 1
                            finish(rec)
                            lease_released = True
                            rec.busy = False
                            self._lock.release()
        iterator = chunks()
        if request.stream:
            class OwnedResponse(InferenceStreamingResponse):
                async def __call__(self, scope, receive, send):
                    nonlocal lease_released
                    try:
                        await super().__call__(scope, receive, send)
                    finally:
                        # Closing an unstarted async generator does not execute
                        # its finally block. The response owns that admission
                        # reservation too (disconnect before first iteration).
                        if not lease_released:
                            with CancelScope(shield=True):
                                # This exact response's reservation, never a
                                # subsequent request's busy/lock ownership.
                                lease_released = True
                                finish(rec, unstarted=True)
                                rec.busy = False
                                self_backend._lock.release()
            self_backend = self
            return OwnedResponse(iterator, media_type='text/event-stream')
        async for _ in iterator:
            pass
        return JSONResponse(content=trace['response'])

    @staticmethod
    def _validate_result_reentry(rec, body, sequence):
        if body is None:
            return
        incoming = json.loads(body).get('messages', [])
        previous = rec.reconstruction_body
        if not previous:
            if any(m.get('role') == 'tool' for m in incoming):
                raise ValueError('foreign tool result without certified request')
            return
        trace = rec.last_turn or {}
        certified = trace.get('certified_outcome')
        if not certified:
            if any(m.get('role') == 'tool' for m in incoming[len(previous['messages']):]):
                raise ValueError('tool result without certified outcome')
            return
        out = json.loads(certified)
        if out['session_id'] != rec.session_id or sequence != out['sequence'] + 1:
            raise ValueError('result request lifetime/sequence mismatch')
        message = out['response']['choices'][0]['message']
        prefix = previous['messages'] + [message]
        if incoming[:len(prefix)] != prefix:
            raise ValueError('conversation differs from certified application outcome')
        additions = incoming[len(prefix):]
        calls = message.get('tool_calls') or []
        results = additions[:len(calls)]
        if calls:
            if (out['outcome_state'] != 'recoverable' or
                    not out['certificate']['executable_tools'] or len(results) != len(calls) or
                    any(r.get('role') != 'tool' or r.get('tool_call_id') != c['id'] or
                        not isinstance(r.get('content'), str) or len(r['content'].encode()) > 65536
                        for r, c in zip(results, calls))):
                raise ValueError('result is not bound to certified completed tool call')
        if any(m.get('role') == 'tool' for m in additions[len(calls):]):
            raise ValueError('duplicate/foreign tool result')

    def close(self):
        # Experiment runner awaits DELETE before synchronous executor shutdown.
        if any(s.busy for s in self.sessions.values()):
            raise RuntimeError('cannot close qualification backend with active owner')
        self._executor.shutdown(wait=True)
