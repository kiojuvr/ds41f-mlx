"""Synthetic transport lifecycle regressions, not checkpoint qualification."""
import asyncio
import json
from types import SimpleNamespace
import unittest

from ds41f_mlx.serving.internal_mtp import InternalMTPQualificationBackend


class Event:
    def to_json(self): return '{"choices":[{"delta":{"content":"x"}}]}'


class FixtureBackend(InternalMTPQualificationBackend):
    # Historical lease tests use local fixture labels, not the qualified wire ID
    # contract. The actual backend still issues every identity; labels resolve only
    # inside this synthetic fixture. M39 tests use the real issuance methods.
    async def create_stateful_session(self, *, session_id=None):
        return await super().create_stateful_session()
    def get_stateful_session(self, session_id):
        if session_id in ('one', 'negative') and self.sessions:
            session_id = next(iter(self.sessions))
        return super().get_stateful_session(session_id)
    async def close_stateful_session(self, session_id):
        if session_id in ('one', 'negative') and self.sessions:
            session_id = next(iter(self.sessions))
        return await super().close_stateful_session(session_id)
    def make_sampler(self, options): return None
    async def _call(self, fn, *args): return fn(*args)
    def _start(self, rec, request, tokenizer, trace):
        rec.owner = object()
        rec.guard = SimpleNamespace(events=[], finished=False)
    def _next(self, rec, trace):
        rec.guard.events.append(Event())
        trace['canonical_emitted'] += 1
        return 1
    def _settle(self, rec, trace):
        rec.canonical = [1, 2, 3]
        trace['quiescence'] = {'canonical_frontier': 3}
        trace['response'] = {'choices': []}
        rec.owner = None
    def _retire(self, rec): rec.owner = None


def request():
    return SimpleNamespace(protocol='chat_completions', image_sources=[], model=None,
                           token_ids=[1, 2], inference_options=SimpleNamespace(max_tokens=3), stream=True)


class LeaseTests(unittest.TestCase):
    def run_async(self, test): asyncio.run(test())

    def test_live_iterator_close_preserves_canonical_and_releases(self):
        async def test():
            b = FixtureBackend()
            rec = await b.create_stateful_session(session_id='one')
            r = await b.qualification_response('one', request(), tokenizer=None)
            self.assertTrue(rec.busy)
            await anext(r.body_iterator)
            self.assertEqual(b.progress['client_acknowledged_ordinal'], 0)
            with self.assertRaisesRegex(RuntimeError, 'active request'):
                await b.qualification_response('one', request(), tokenizer=None)
            await r.body_iterator.aclose()
            self.assertFalse(rec.busy)
            self.assertFalse(b._lock.locked())
            self.assertEqual(rec.canonical, [1, 2, 3])
            self.assertIsNone(rec.owner)
            b.close()
        self.run_async(test)

    def test_response_send_failure_before_iterator_start_releases_reservation(self):
        async def test():
            b = FixtureBackend()
            rec = await b.create_stateful_session(session_id='one')
            r = await b.qualification_response('one', request(), tokenizer=None)
            async def receive():
                await asyncio.sleep(10)
                return {'type': 'http.disconnect'}
            async def send(message): raise OSError('socket closed before response headers')
            with self.assertRaises(BaseException):
                await r({'type':'http', 'asgi':{'spec_version':'2.4'}}, receive, send)
            self.assertFalse(rec.busy)
            self.assertFalse(b._lock.locked())
            self.assertIsNone(rec.owner)
            b.close()
        self.run_async(test)

    def test_retired_response_cannot_release_subsequent_request_lease(self):
        async def test():
            b = FixtureBackend()
            rec = await b.create_stateful_session(session_id='one')
            first = await b.qualification_response('one', request(), tokenizer=None)
            await anext(first.body_iterator)
            await first.body_iterator.aclose()
            next_request = request()
            next_request.token_ids = [1, 2, 3, 4]
            second = await b.qualification_response('one', next_request, tokenizer=None)
            async def receive(): return {'type': 'http.disconnect'}
            async def send(_message): raise OSError('header send failed')
            scope = {'type':'http', 'asgi':{'spec_version':'2.4'}}
            with self.assertRaises(BaseException):
                await first(scope, receive, send)
            self.assertTrue(rec.busy)
            self.assertTrue(b._lock.locked())
            with self.assertRaises(BaseException):
                await second(scope, receive, send)
            self.assertFalse(rec.busy)
            self.assertFalse(b._lock.locked())
            b.close()
        self.run_async(test)

    def test_protected_failure_is_poisoned_not_idle(self):
        async def test():
            b = FixtureBackend()
            rec = await b.create_stateful_session(session_id='one')
            def fail(*args): raise RuntimeError('protected forward failed')
            b._next = fail
            r = await b.qualification_response('one', request(), tokenizer=None)
            with self.assertRaisesRegex(RuntimeError, 'protected forward'):
                await anext(r.body_iterator)
            self.assertTrue(rec.poisoned)
            self.assertFalse(rec.busy)
            self.assertFalse(b._lock.locked())
            with self.assertRaisesRegex(RuntimeError, 'poisoned'):
                await b.qualification_response('one', request(), tokenizer=None)
            b.close()
        # Fixture settle would recover arbitrary failures unlike real poisoned
        # M34 session. Emulate its mandatory quiescence refusal.
        original = FixtureBackend._settle
        def refuse(*args): raise RuntimeError('protected forward failed')
        FixtureBackend._settle = refuse
        try: self.run_async(test)
        finally: FixtureBackend._settle = original

    def test_delete_holds_singleton_lease_until_retirement_finishes(self):
        async def test():
            b = FixtureBackend()
            rec = await b.create_stateful_session(session_id='one')
            entered, release = asyncio.Event(), asyncio.Event()
            async def retire_call(fn, *args):
                entered.set()
                await release.wait()
                return fn(*args)
            b._call = retire_call
            closing = asyncio.create_task(b.close_stateful_session('one'))
            await entered.wait()
            self.assertTrue(rec.busy)
            with self.assertRaisesRegex(RuntimeError, 'active request'):
                await b.qualification_response('one', request(), tokenizer=None)
            release.set()
            await closing
            self.assertTrue(rec.closed)
            self.assertFalse(rec.busy)
            self.assertFalse(b._lock.locked())
            b.close()
        self.run_async(test)

    def test_retirement_error_releases_admission_but_remains_poisoned(self):
        async def test():
            b = FixtureBackend()
            rec = await b.create_stateful_session(session_id='one')
            def fail(*args): raise RuntimeError('retirement failed')
            b._start = fail
            b._retire = fail
            r = await b.qualification_response('one', request(), tokenizer=None)
            with self.assertRaisesRegex(RuntimeError, 'retirement failed'):
                await anext(r.body_iterator)
            self.assertTrue(rec.poisoned)
            self.assertFalse(rec.busy)
            self.assertFalse(b._lock.locked())
            with self.assertRaisesRegex(RuntimeError, 'maximum live session'):
                await b.create_stateful_session()
            with self.assertRaisesRegex(RuntimeError, 'poisoned'):
                await b.qualification_response('one', request(), tokenizer=None)
            b.close()
        self.run_async(test)

    def test_cancel_scope_is_observed_between_shielded_native_responses(self):
        from anyio import CancelScope
        async def test():
            b = FixtureBackend()
            rec = await b.create_stateful_session(session_id='one')
            next_call = b._next
            calls = []
            with CancelScope() as scope:
                def emit_then_disconnect(rec, trace):
                    calls.append(1)
                    result = next_call(rec, trace)
                    scope.cancel()
                    return result
                b._next = emit_then_disconnect
                r = await b.qualification_response('one', request(), tokenizer=None)
                await anext(r.body_iterator)
                self.fail('cancelled owner must not yield another chunk')
            self.assertEqual(len(calls), 1)
            self.assertEqual(rec.canonical, [1, 2, 3])
            self.assertEqual(rec.pending_delivery, (Event().to_json(),))
            self.assertFalse(rec.busy)
            self.assertFalse(rec.poisoned)
            self.assertFalse(b._lock.locked())
            b.close()
        self.run_async(test)

    def test_terminal_worker_completion_lost_to_cancel_keeps_exact_outcome(self):
        import threading
        async def test():
            b = FixtureBackend()
            rec = await b.create_stateful_session()
            entered, release = asyncio.Event(), threading.Event()
            loop = asyncio.get_running_loop()
            calls = []
            def terminal(rec, trace):
                calls.append('next')
                rec.guard.events.append(Event())
                rec.guard.finished = True
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(5)
                return 1
            original_settle = b._settle
            def settle(rec, trace):
                calls.append('settle')
                original_settle(rec, trace)
                rec.certificate = {'representable': True}
            b._next, b._settle = terminal, settle
            async def worker(fn, *args):
                if fn == b._advance_application:
                    return await InternalMTPQualificationBackend._call(b, fn, *args)
                return fn(*args)
            b._call = worker
            body = b'{"messages": []}'
            response = await b.qualification_response(rec.session_id, request(),
                tokenizer=None, body=body, sequence=1)
            task = asyncio.create_task(anext(response.body_iterator))
            try:
                await asyncio.wait_for(entered.wait(), 5)
                task.cancel()
            finally:
                release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(calls, ['next', 'settle'])
            self.assertEqual(rec.pending_delivery, (Event().to_json(),))
            self.assertTrue(rec.last_turn['application_terminal'])
            self.assertTrue(rec.last_turn['cancelled'])
            self.assertFalse(rec.poisoned)
            self.assertFalse(rec.busy)
            retry = await b.qualification_response(rec.session_id, request(),
                tokenizer=None, body=body, sequence=1, outcome_projection=True)
            self.assertEqual(json.loads(retry.body)['response'], rec.last_turn['response'])
            from ds41f_mlx.serving.server import sse_frame
            for _ in range(2):
                replay = await b.qualification_response(rec.session_id, request(),
                    tokenizer=None, body=body, sequence=1)
                frames = [frame async for frame in replay.body_iterator]
                self.assertEqual(frames, [sse_frame(None, Event().to_json()), sse_frame(None, '[DONE]')])
            self.assertEqual(calls, ['next', 'settle'])
            b.close()
        self.run_async(test)

    def test_native_chunk_transfer_serializes_before_response_append(self):
        import deepseek_recipe as d
        processor = d.StreamProcessor(d.ChatCompletionChunkGenerator('x', 'v41', False, False), d.ParsingOptions())
        events = processor.push(d.InferenceChunk.text('hello', 1))
        self.assertTrue(events)
        class Cache:
            def size(self): return 3
        class Quiet:
            target_cache = [Cache() for _ in range(40)]
            dspark_context = SimpleNamespace(caches=[SimpleNamespace(offset=3) for _ in range(3)])
            canonical_tokens = (1, 2, 3)
            counters = SimpleNamespace(require_m28_zeroes=lambda: None)
            def to_json(self): return {'canonical_frontier': 3}
        class Owner:
            calls = 0
            _bg = SimpleNamespace(_generation_batch=SimpleNamespace())
            history = SimpleNamespace(canonical_generated_tokens=[3], canonical_frontier=3)
            connection = SimpleNamespace(lifetime='fixture', revision=1, disposition='settled',
                consumed_tokens=(1,2,3), queue_ahead=(), pending_prediction=None, rng_draws=0)
            def quiesce(self):
                self.calls += 1
                assert self.calls == 1
                return Quiet()
            def close(self): pass
        b = FixtureBackend()
        owner = Owner()
        rec = SimpleNamespace(owner=owner, guard=SimpleNamespace(events=events, matches=[]), processor=processor, fence=None)
        trace = {'response_id': 'x'}
        InternalMTPQualificationBackend._settle(b, rec, trace)
        self.assertEqual(owner.calls, 1)
        self.assertIsNone(rec.owner)
        self.assertTrue(trace['protocol_events'])
        with self.assertRaisesRegex(RuntimeError, 'consumed'):
            events[0].to_json()
        b.close()

    def test_protocol_failure_after_quiescence_cannot_settle_owner_twice(self):
        async def test():
            b = FixtureBackend()
            rec = await b.create_stateful_session(session_id='one')
            calls = []
            def failure(rec, trace):
                calls.append(1)
                rec.owner = None  # native ownership already retired
                trace['quiescence'] = {'canonical_frontier': 3}
                raise RuntimeError('protocol serialization failure')
            b._settle = failure
            r = await b.qualification_response('one', request(), tokenizer=None)
            await anext(r.body_iterator)
            with self.assertRaisesRegex(RuntimeError, 'protocol serialization'):
                await r.body_iterator.aclose()
            self.assertEqual(len(calls), 1)
            self.assertTrue(rec.poisoned)
            self.assertFalse(b._lock.locked())
            b.close()
        self.run_async(test)

    def test_bad_prefix_is_atomic(self):
        async def test():
            b = FixtureBackend()
            rec = await b.create_stateful_session(session_id='one')
            rec.canonical = [8, 9]
            before = rec.to_json()
            with self.assertRaisesRegex(ValueError, 'exactly extend'):
                await b.qualification_response('one', request(), tokenizer=None)
            self.assertEqual(before, rec.to_json())
            self.assertFalse(b._lock.locked())
            b.close()
        self.run_async(test)
