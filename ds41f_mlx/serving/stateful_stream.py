"""Demand-driven stateful HTTP transport, distinct from diagnostic history.

One response reservation; one worker-owned LiveRecipeTurn; one protocol batch.
No producer runs ahead of HTTP demand. A cancelled in-flight _call drains exactly
that protected worker operation before settlement on the same worker.
"""
from __future__ import annotations

import asyncio
from collections import deque
import json
from threading import Event
from time import time, perf_counter
from uuid import uuid4

from anyio import CancelScope

from ds41f_mlx.runtime.live_turn import LiveRecipeTurn
from ds41f_mlx.runtime.tool_boundary_session import M11RecipeToolSession
from .recovery_certificate import reconstruction_certificate


class StatefulStream:
    def __init__(self, backend, session_id, prepared, tokenizer, body, options, application_id=None):
        self.backend = backend
        self.rec = backend.validate_stateful_admission(session_id, prepared)
        self.frontier_before = None if self.rec.m11 is None else self.rec.m11.m8.frontier
        self.prepared, self.tokenizer, self.body, self.options = prepared, tokenizer, body, options
        self.request_id = str(uuid4())
        slot = self.rec.response_reservation
        self.reservation = slot if slot is not None and slot.state == 'active' else None
        if self.reservation is not None:
            self.reservation.request_id = self.request_id
        self.application_id = application_id  # correlation only, not a retry fence
        self.cancel = Event()
        self.rec.busy = True  # before response headers, even if iterator never starts
        self.rec.active_stream = self
        self.rec.updated_at = time()
        self.locked = self.closed = self.started = False
        self.failed = False
        self.cursor = None
        self.pending = deque()
        self.done_sent = False
        self.gate = asyncio.Lock()  # transport control serialization
        self.start_time = perf_counter()
        self.trace = dict(session_id=session_id, request_id=self.request_id, stream=True,
                          prompt_tokens=len(prepared.token_ids), started_at=time(), delivered_events=0)
        backend.session_traces.append(self.trace)

    def __aiter__(self):
        return self

    def publish(self, turn, cancelled):
        rec = self.rec
        record = turn.to_json()
        record.update(request_id=self.request_id, application_request_id=self.application_id, cancelled=cancelled,
                      capacity=None if self.prepared.capacity is None else {**self.prepared.capacity, 'binding': True},
                      termination_reason=('user_stop' if cancelled else 'context_capacity' if turn.finish_reason == 'length' and self.prepared.capacity and self.prepared.capacity['output_mode'] == 'auto' else 'output_limit' if turn.finish_reason == 'length' else turn.finish_reason))
        if cancelled or turn.tool_calls:
            certificate = reconstruction_certificate(
                json.loads(self.body), turn.response_json, rec.m11.m8.token_history,
                tokenizer=self.tokenizer, recipe_path=self.backend.recipe_path,
                options=self.options, checkpoint=self.backend.checkpoint,
                completed_tool_block=turn.finish_reason == 'tool_calls')
            certificate.pop('witness', None)  # no second transcript/image owner
            record['reconstruction'] = certificate
            if not certificate['representable']:
                # Already committed tokens are not rolled back or rebuilt. Retire
                # native authority and retain the diagnostic outcome for GET.
                rec.recovery_state = 'unrecoverable'
                rec.m11.m8.close()
            else:
                rec.recovery_state = 'ready'
        rec.last_turn = record
        rec.request_count += 1
        rec.updated_at = time()
        rec.last_error = None
        diag = rec.m11.diagnostics()
        m8 = diag['m8']
        self.trace.update(ok=True, cancelled=cancelled, finish_reason=turn.finish_reason,
                          generated_token_count=len(turn.generated_tokens), frontier=turn.frontier_after_commit,
                          prompt_replay_count=turn.prompt_replay_count, full_cache_repack_count=turn.full_cache_repack_count,
                          image_encoded_count=diag['last_image_encoded_count'],
                          all_cache_offsets_equal_frontier=m8['all_cache_offsets_equal_frontier'],
                          cache_offsets_all=m8['cache_offsets_all'], elapsed_seconds=perf_counter()-self.start_time)

    def start_worker(self):
        if self.cancel.is_set():
            return
        rec, backend = self.rec, self.backend
        sampler = backend.make_sampler(self.prepared.inference_options)
        maximum = backend.request_max_tokens(self.prepared)
        if rec.m11 is None:
            rec.m11 = M11RecipeToolSession.start_from_prepared(
                model=backend._model, tokenizer=self.tokenizer, checkpoint=backend.checkpoint,
                omlx_path=backend.omlx_path, recipe_path=backend.recipe_path,
                prepared=self.prepared, sampler=sampler, max_tokens=maximum)
            self.trace['created_runtime_session'] = True
        else:
            before = rec.m11.m8.frontier
            rec.m11.m8.sampler = sampler
            rec.m11.continue_from_prepared(self.prepared, max_tokens=maximum)
            self.trace.update(created_runtime_session=False, frontier_before=before, exact_prefix_extension=True)
        # Publish the cursor inside the worker, before cancellation can lose the
        # future's result. Startup stays protected; Stop then settles before decode.
        rec.m11.execution_strategy = getattr(backend, 'execution_strategy', 'off')
        self.cursor = LiveRecipeTurn(rec.m11, self.prepared, self.publish, cancelled=self.cancel.is_set)

    async def __anext__(self):
        async with self.gate:
            return await self._next()

    async def _next(self):
        if self.closed or self.done_sent:
            raise StopAsyncIteration
        try:
            # ASGI socket sends may not suspend; explicitly admit disconnects.
            await asyncio.sleep(0)
            if not self.started:
                await self.backend._lock.acquire()
                self.locked = True
                if not self.cancel.is_set():
                    await self.backend._call(self.backend.load)
                    await self.backend._call(self.start_worker)
                self.started = True
            if self.cursor is None:
                await self._close()
                raise StopAsyncIteration
            if self.cancel.is_set() and not self.cursor.closed:
                self.pending.extend(await self.backend._call(lambda: self.cursor.finish(cancelled=True)))
            while not self.pending and not self.cursor.closed:
                self.pending.extend(await self.backend._call(self.cursor.advance))
                await asyncio.sleep(0)
                if self.cancel.is_set() and not self.cursor.closed:
                    self.pending.extend(await self.backend._call(lambda: self.cursor.finish(cancelled=True)))
            if self.pending:
                event = self.pending.popleft()
                self.trace['delivered_events'] += 1  # iterator yield, NOT socket/client ACK
                frame = 'data: ' + json.dumps(event, separators=(',', ':')) + '\n\n'
                if self.reservation is not None:
                    self.reservation.append(frame)
                return frame
            self.done_sent = True
            await self._close()
            return 'data: [DONE]\n\n'
        except StopAsyncIteration:
            raise
        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                if self.reservation is not None:
                    self.reservation.burn()  # never freeze a worker error as successful EOF
                m8 = None if self.rec.m11 is None else self.rec.m11.m8
                self.failed = self.cursor is not None or (m8 is not None and (m8.state != 'idle' or m8.frontier != self.frontier_before))
                self.rec.last_error = str(exc)
                self.trace.update(ok=False, error=str(exc))
            await self._close()
            raise

    def retire_worker(self):
        m8 = self.rec.m11.m8
        try:
            m8.close()
        finally:
            try:
                # A burnt target cannot yield idle state, but must still release
                # its wired policy/resources. Never retry state extraction.
                if m8.generation is not None:
                    generation, m8.generation = m8.generation, None
                    try:
                        generation.close()
                    finally:
                        m8.live_cache = []
                        m8.closed = True
            finally:
                if self.cursor is not None:
                    self.cursor.abort()

    async def aclose(self):
        self.cancel.set()
        with CancelScope(shield=True):
            async with self.gate:
                await self._close()

    async def _close(self):
        if self.closed:
            return
        self.closed = True
        self.cancel.set()
        try:
            with CancelScope(shield=True):
                if self.failed and self.rec.m11 is not None:
                    self.rec.recovery_state = 'unrecoverable'
                    try:
                        await self.backend._call(self.retire_worker)
                    except Exception as exc:
                        self.rec.last_error += '; retirement: ' + str(exc)
                elif self.cursor is not None and not self.cursor.closed:
                    try:
                        final_events = await self.backend._call(lambda: self.cursor.finish(cancelled=True))
                        if self.reservation is not None:
                            # Serialize once into the same reserved response, even
                            # when the socket vanished before terminal delivery.
                            for event in list(self.pending) + list(final_events):
                                self.reservation.append('data: ' + json.dumps(event, separators=(',', ':')) + '\n\n')
                            self.pending.clear()
                    except BaseException as exc:
                        self.rec.last_error = str(exc)
                        self.rec.recovery_state = 'unrecoverable'
                        await self.backend._call(self.retire_worker)
                elif self.rec.m11 is not None and ((self.cursor is not None and self.cursor.turn is None) or (self.cursor is None and (self.rec.m11.m8.generation is not None or self.rec.m11.m8.closed))):
                    self.rec.recovery_state = 'unrecoverable'
                    await self.backend._call(self.retire_worker)
        except BaseException:
            if self.reservation is not None:
                self.reservation.burn()
            raise
        finally:
            if self.reservation is not None and self.reservation.state == 'active':
                try:
                    if self.failed or (self.cursor is not None and self.cursor.turn is None):
                        self.reservation.burn()
                    else:
                        for event in self.pending:
                            self.reservation.append('data: ' + json.dumps(event, separators=(',', ':')) + '\n\n')
                        self.reservation.append('data: [DONE]\n\n')
                        self.reservation.complete()
                except Exception as exc:
                    self.reservation.burn()
                    self.rec.recovery_state = 'unrecoverable'
                    self.rec.last_error = str(exc)
            self.pending.clear()
            # Exact response reservation; never clear a later request's owner.
            if self.rec.active_stream is self:
                self.rec.active_stream = None
                self.rec.busy = False
                self.rec.updated_at = time()
            if self.locked:
                self.locked = False
                self.backend._lock.release()
            # Break cursor -> on_commit -> transport cycles explicitly. Request
            # image bytes/CPU patches/parser objects must not await cyclic GC.
            self.cursor = None
            self.prepared = None
            self.body = None  # original request/image bytes are request-local
