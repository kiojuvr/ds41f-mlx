"""M35 qualification-only injection. Not a release selector or public API option.

Uses the actual recipe server routes, but only when an experiment explicitly
constructs this backend. One logical session and one response owner. Socket send
is deliberately NOT a CanonicalTransportHistory acknowledgement.
"""
from __future__ import annotations

import asyncio
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

    def to_json(self):
        return dict(id=self.session_id, state='closed' if self.closed else
                    'busy' if self.busy else 'poisoned' if self.poisoned else
                    'idle' if self.cache is not None else 'empty',
                    request_count=self.request_count, last_turn=self.last_turn,
                    canonical_frontier=len(self.canonical),
                    outcome_state='active' if self.busy else 'not_admitted' if self.fence and self.fence['state'] == 'not_admitted' else 'poisoned' if self.poisoned and not self.unrecoverable else
                    'unrecoverable' if self.unrecoverable else 'recoverable' if self.certificate and self.certificate['representable'] else 'not_admitted',
                    certificate=self.certificate, request_fence=self.fence,
                    next_sequence=self.consumed_sequence + 1)


class InternalMTPQualificationBackend(DeepSeekRecipeRuntimeBackend):
    """Explicit object injection only; no env, request or release MTP selector."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_live_sessions = 1
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
        self._model, _ = load(self.checkpoint, preserve_mtp=True, engram_ssd_offload=True)
        self._model.language_model.configure_mtp(True, 5)
        self._model.language_model._p7_enable_overlap = True

    async def infer(self, request):
        raise ValueError('internal MTP qualification requires singleton stateful Chat Completions')
        yield  # async iterator contract

    async def create_stateful_session(self, *, session_id=None):
        if any(not s.closed for s in self.sessions.values()):
            raise RuntimeError('maximum live session count reached')
        sid = session_id or f'sess_{uuid4().hex}'
        if sid in self.sessions:
            raise ValueError('qualification session IDs cannot be reused')
        rec = QualificationSession(sid)
        self.sessions[sid] = rec
        return rec

    async def persist_stateful_session(self, *args, **kwargs):
        raise RuntimeError('internal MTP persistence unsupported before I/O')

    async def restore_stateful_session(self, *args, **kwargs):
        raise RuntimeError('internal MTP restore unsupported before mutation or I/O')

    async def close_stateful_session(self, session_id):
        rec = self.get_stateful_session(session_id)
        if rec.busy or self._lock.locked():
            raise RuntimeError('session already has an active request')
        await self._lock.acquire()
        rec.busy = True
        try:
            await self._call(self._retire, rec)
            rec.closed = True
            return rec.to_json()
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

    def _start(self, rec, request, tokenizer, trace):
        import deepseek_recipe as d
        import mlx.core as mx
        import numpy as np
        from mlx_lm.generate import generation_stream
        from ds41f_mlx.prefill_fp8_mlx import DeferredPrefillAppend, LivePrefillResult, handoff_to_generation
        from ds41f_mlx.runtime.mtp_lifecycle import DSparkCommittedContext, OMLXMTPGenerationSession
        from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
        from ds41f_mlx.runtime.recipe_semantic_guard import RecipeSemanticGuard
        with mx.stream(generation_stream):
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
            try:
                for i, layer in original.items(): lm.layers[i] = Tap(i, layer)
                app = DeferredPrefillAppend.create(lm, rec.cache, ids[:-1], committed_frontier=C, mx=mx)
                app.execute_all()
            finally:
                for i, layer in original.items(): lm.layers[i] = layer
            hidden = mx.concatenate([mx.concatenate(taps[i], axis=1) for i in taps], axis=-1)
            lm.dspark_append_context(hidden, rec.rings, start_offset=C)
            mx.eval(*[c.keys for c in rec.rings]); mx.synchronize(generation_stream)
            context = DSparkCommittedContext.from_native(rec.rings, frontier=len(ids)-1, target_layer_ids=lm._config.dspark_target_layer_ids)
            live = LivePrefillResult.from_committed(app.commit_certificate, prefix_token_ids=ids[:-1])
            live.dspark_committed_context = context
            generator = d.ChatCompletionRequest.chunk_generator(request.conversation_request, trace['response_id'], request.model or self.model_id)
            generator = generator.with_include_usage(request.include_usage)
            rec.processor = d.StreamProcessor(generator, request.conversation_request.parsing_options, tokenizer)
            rec.guard = RecipeSemanticGuard(rec.processor, control_token_ids=request.stop_token_ids)
            rec.guard.events.extend(rec.processor.push(d.InferenceChunk.ready(prompt_usage=d.PromptUsage(prompt_tokens=len(ids), prompt_cache_hit_tokens=0))))
            cfg = OMLXDecodeConfig(omlx_path=self.omlx_path, checkpoint_path=self.checkpoint,
                                  preserve_mtp=True, speculation_enabled=True, stop_token_ids=request.stop_token_ids)
            class Factory:
                @classmethod
                def from_prefilled_cache(cls, model, cache, prefix, config, *, max_tokens, sampler):
                    return OMLXMTPGenerationSession(model, cache, np.asarray(prefix), context,
                        config=config, sampler=sampler, max_tokens=max_tokens, semantic_guard=rec.guard)
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
            canonical_generated = list(rec.owner.history.canonical_generated_tokens)
            rec.owner.close(); rec.owner = None
            assert len(rec.cache) == 40 and all(c.size() == len(rec.canonical) for c in rec.cache)
            assert all(c.offset == len(rec.canonical) for c in rec.rings)
            # Native response.append transfers chunk ownership. Serialize the
            # evidence while recipe chunks still own their native payloads.
            protocol_events = [json.loads(e.to_json()) for e in rec.guard.events]
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
                completed = rec.guard.finished and any(m['identity'][0] == 'DSML_TOOL_CALL_BLOCK_END'
                                                       for m in trace['terminal_matches'])
                rec.certificate = reconstruction_certificate(rec.reconstruction_body, trace['response'], rec.canonical,
                    tokenizer=rec.reconstruction_tokenizer, recipe_path=self.recipe_path, completed_tool_block=completed)
                trace['certificate'] = rec.certificate
                if not rec.certificate['representable']:
                    rec.unrecoverable = True
                    rec.poisoned = True  # legacy containment/retirement label, not internal-failure outcome
                    trace['recovery_error'] = 'no official-recipe reconstruction certificate; DELETE required'
            rec.processor.close(); rec.processor = None
            trace['cleanup_s'] = perf_counter()-t0

    @staticmethod
    def _qualify_settled_protocol(rec, trace):
        # Native cache quiescence is not a complete protocol-result certificate.
        # The recipe can emit argument deltas before the DSML terminal becomes
        # canonical. Never advertise that partial call as an idle agent result,
        # even if its current arguments happen to be valid JSON. Only the native
        # recipe terminal/finish owner can establish a complete call.
        choices = trace['response'].get('choices', [])
        has_calls = any(c.get('message', {}).get('tool_calls') for c in choices)
        completed_block = any(m['identity'][0] == 'DSML_TOOL_CALL_BLOCK_END'
                              for m in trace.get('terminal_matches', []))
        if has_calls and not (rec.guard.finished and completed_block and
                              all(c.get('finish_reason') == 'tool_calls' for c in choices)):
            rec.poisoned = True
            trace['recovery_error'] = 'unfinished canonical tool protocol; ordinary continuation not qualified'

    async def qualification_response(self, session_id, request, *, tokenizer, body=None, sequence=None):
        """Acquire before HTTP 200. Retain lease until generator cleanup, not send."""
        from fastapi.responses import JSONResponse
        from .server import InferenceStreamingResponse, sse_frame
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
            return JSONResponse(content=retry)
        if rec.poisoned:
            raise RuntimeError('session poisoned; DELETE required')
        if rec.busy or self._lock.locked():
            raise RuntimeError('singleton already has an active request')
        if request.token_ids[:len(rec.canonical)] != rec.canonical or len(request.token_ids) <= len(rec.canonical):
            raise ValueError('request must exactly extend retained canonical prefix')
        await self._lock.acquire()
        rec.busy = True
        reserve(rec, sequence, body)
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
                        token = await self._call(self._next, rec, trace)
                    await checkpoint()
                    t0 = perf_counter()
                    events = rec.guard.events[cursor:]
                    cursor = len(rec.guard.events)
                    frames = [sse_frame(None, event.to_json()) for event in events]
                    trace['formatting_s'] += perf_counter()-t0
                    trace['protocol_chunks_produced'] += len(frames)
                    terminal = rec.guard.finished or token is None
                    # Settle before exposing a semantic terminal. Ordinary chunks
                    # already have canonical parser/model ownership, not idle caches.
                    if terminal:
                        with CancelScope(shield=True):
                            await self._call(self._settle, rec, trace)
                        settled = True
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
                rec.poisoned = True
                raise
            finally:
                import sys
                failure = sys.exc_info()[1]
                if failure is not None:
                    trace['exit_exception'] = repr(failure)
                trace['cancelled'] = not settled or trace['iterator_yields'] < trace['protocol_chunks_produced']
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

    def close(self):
        # Experiment runner awaits DELETE before synchronous executor shutdown.
        if any(s.busy for s in self.sessions.values()):
            raise RuntimeError('cannot close qualification backend with active owner')
        self._executor.shutdown(wait=True)
