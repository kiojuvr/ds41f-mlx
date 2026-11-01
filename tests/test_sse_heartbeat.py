"""Real TCP/HTTP liveness over ordinary_response and the Scheduler tool gate.

Only physical inference/retirement is controlled. Native Recipe owns all events.
No model weights are loaded; short read deadlines are inactivity, not total time.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
from threading import Thread
import time
from types import SimpleNamespace as NS

import deepseek_recipe as d
import httpx
import pytest
import uvicorn

from ds41f_mlx.mtp_identity import config
from ds41f_mlx.runtime.recipe_semantic_guard import MessagesSemanticGuard
from ds41f_mlx.serving import production_mtp
from ds41f_mlx.serving.production_mtp import ProductionMTPBackend, ProductionScheduler
from ds41f_mlx.serving.server import create_app, sse_frame
from test_recipe02_semantic_guard import encode, feed, PREFIX, ARG_END, TAIL

HEARTBEAT = b': keepalive\n\n'
INTERVAL = 0.025
READ_TIMEOUT = 0.15
SILENCE = 0.24


@contextmanager
def wire_server(app):
    """A real uvicorn socket, not TestClient's fully buffered ASGI transport."""
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level='critical', lifespan='off'))
    thread = Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert thread.is_alive() and time.monotonic() < deadline
        time.sleep(.01)
    try:
        yield f'http://127.0.0.1:{sock.getsockname()[1]}'
    finally:
        server.should_exit = True
        thread.join(10)
        assert not thread.is_alive(), 'orphaned HTTP stream'
        sock.close()


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.setattr(production_mtp, 'SSE_HEARTBEAT_INTERVAL', INTERVAL)
    backend = ProductionMTPBackend(runtime_config=config(profile='mtp-serving-v1'))
    backend.dependency_identity = 'controlled-heartbeat-test'
    state = NS(mode='success', steps=0, events=[], response=None, settled=False,
               cancelled=False, closed=False, aborts=0, silent_steps=0)
    scheduler = NS(requests={}, running={}, waiting=[], _completed=[], backend=backend)
    backend.scheduler = scheduler

    def enqueue(prepared, tokenizer, rid):
        converted = prepared.conversation_request
        guard = MessagesSemanticGuard(d.MessagesSemanticSession(
            d.MessagesRequest.chunk_generator(converted, rid, 'model'),
            converted.parsing_options, tokenizer))
        guard._push(d.InferenceChunk.ready(prompt_usage=d.PromptUsage(prompt_tokens=len(prepared.token_ids))))
        state.guard, state.tokenizer, state.steps = guard, tokenizer, 0
        state.settled = state.closed = state.cancelled = False
        rec = NS(pending_delivery=[e.to_json() for e in guard.events])
        req = NS(request_id=rid, prepared=prepared, execution=rec, trace={}, delivery_cursor=0)
        scheduler.requests[rid] = req
        scheduler.waiting.append(req)

    def schedule():
        if scheduler.waiting:
            # Real synchronous worker occupation: heartbeat must run independently.
            time.sleep(SILENCE)
            req = scheduler.waiting.pop(0)
            scheduler.running[req.request_id] = req
            return [req], []
        return [], []

    def advance(rec, trace):
        state.steps += 1
        tok, guard = state.tokenizer, state.guard
        if state.steps == 1:
            feed(guard, encode(tok, 'intro '))
        elif state.steps == 2:
            feed(guard, encode(tok, PREFIX + 'assembled'))
        elif state.steps < 26:
            # Many empty broker outputs while the tool suffix stays withheld.
            state.silent_steps += 1
            time.sleep(.01)
        elif state.mode == 'error':
            raise RuntimeError('controlled generation failure')
        else:
            feed(guard, encode(tok, ARG_END + TAIL))
            assert guard.finished and guard.tool_complete
            trace['application_terminal'] = True
        rec.pending_delivery = [e.to_json() for e in guard.events]

    def retire(req, *, error=None, cancelled=False):
        state.cancelled = cancelled
        state.settled = error is None and not cancelled
        state.events = [e.to_json() for e in state.guard.events]
        state.response = json.loads(state.guard.response.to_json())
        scheduler.requests.pop(req.request_id, None)
        scheduler.running.pop(req.request_id, None)
        scheduler.waiting[:] = [r for r in scheduler.waiting if r is not req]
        scheduler._completed.append(NS(request_id=req.request_id, finished=True,
            error=str(error) if error else 'cancelled' if cancelled else None,
            recipe_events=() if error or cancelled else state.events[req.delivery_cursor:],
            recipe_response=state.response))
        state.guard.close()
        state.closed = True

    def abort(rid):
        state.aborts += 1
        if rid in scheduler.requests:
            retire(scheduler.requests[rid], cancelled=True)

    scheduler._process_pending_aborts = lambda: None
    scheduler._schedule_waiting = schedule
    scheduler._retire = retire
    scheduler.abort_request = abort
    scheduler.step = lambda: ProductionScheduler.step(scheduler)
    monkeypatch.setattr(backend, '_enqueue', enqueue)
    monkeypatch.setattr(backend, '_advance_application', advance)
    app = create_app(profile='mtp-serving-v1', backend=backend, runtime_config=backend.runtime_config)
    with wire_server(app) as base:
        yield base, backend, state
        deadline = time.monotonic() + 5
        while backend._deliveries or backend._settled or backend._pump_task is not None:
            assert time.monotonic() < deadline, 'orphaned delivery/pump'
            time.sleep(.01)
        assert not scheduler.requests
    backend.scheduler = None
    backend.close()


def body(stream=True):
    return dict(model='deepseek-flash', stream=stream, max_tokens=256,
        thinking={'type': 'disabled'}, messages=[dict(role='user', content='Call f.')],
        tools=[dict(name='f', input_schema={'type': 'object'})])


def collect(response, state):
    wire, received = bytearray(), []
    started = time.monotonic()
    for chunk in response.iter_raw():
        wire.extend(chunk)
        received.append((time.monotonic()-started, chunk))
        if b'event: content_block_start\n' in chunk and b'"tool_use"' in chunk:
            assert state.settled, 'incomplete tool disclosed before settlement'
    return bytes(wire), received


def test_wire_silence_tool_gate_semantics_and_exact_id(transport):
    base, backend, state = transport
    with httpx.Client(timeout=httpx.Timeout(5, read=READ_TIMEOUT), trust_env=False) as client:
        with client.stream('POST', base+'/v1/messages', json=body()) as response:
            assert response.status_code == 200
            wire, received = collect(response, state)
    assert wire.startswith(HEARTBEAT)  # prefill, before the first semantic event
    assert wire.count(HEARTBEAT) >= 6
    # The request exceeds the read timeout many times, yet each read stays live.
    assert received[-1][0] > 2*READ_TIMEOUT
    assert max(b[0]-a[0] for a, b in zip([(0, b'')]+received, received)) < READ_TIMEOUT
    expected = ''.join(sse_frame(json.loads(e)['type'], e) for e in state.events).encode()
    if directory := os.environ.get('SSE_HEARTBEAT_WIRE_EVIDENCE'):
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        (output/'controlled.sse').write_bytes(wire)
        (output/'controlled-semantic.sse').write_bytes(expected)
        (output/'controlled-timing.json').write_text(json.dumps([
            dict(elapsed_s=t, bytes=len(b), chunk=b.decode()) for t, b in received], indent=2)+'\n')
    assert wire.replace(HEARTBEAT, b'') == expected
    assert wire.endswith(sse_frame('message_stop', state.events[-1]).encode())
    assert state.settled and not state.cancelled and state.silent_steps >= 20
    # A native text delta is followed by comments during the withheld tool suffix.
    text_end = wire.index(b'"text_delta"')
    tool_start = wire.index(b'"type":"tool_use"')
    assert HEARTBEAT in wire[text_end:tool_start]
    response = state.response
    call, = [b for b in response['content'] if b['type'] == 'tool_use']
    assert call['input'] == {'x': 'assembled'}
    continued = body() | {'messages': body()['messages'] + [
        dict(role='assistant', content=response['content']),
        dict(role='user', content=[dict(type='tool_result', tool_use_id=call['id'], content='ok')])]}
    from ds41f_mlx.serving.messages_admission import convert_messages
    converted, _, _ = convert_messages(json.dumps(continued).encode())
    assistant = next(m for m in converted.conversation.messages if m.role == 'assistant')
    assert assistant.tool_calls[0].id == call['id']


def test_without_heartbeat_same_silence_read_times_out(transport, monkeypatch):
    base, _, state = transport
    monkeypatch.setattr(production_mtp, 'SSE_HEARTBEAT_INTERVAL', 60)
    with httpx.Client(timeout=httpx.Timeout(5, read=.08), trust_env=False) as client:
        with pytest.raises(httpx.ReadTimeout):
            with client.stream('POST', base+'/v1/messages', json=body()) as response:
                list(response.iter_raw())
    deadline = time.monotonic()+5
    while not state.closed:
        assert time.monotonic() < deadline
        time.sleep(.01)
    assert state.cancelled and not state.settled


@pytest.mark.parametrize('during_tool', [False, True])
def test_disconnect_stops_heartbeat_and_cleans_native_owner(transport, during_tool):
    base, _, state = transport
    with httpx.Client(timeout=5, trust_env=False) as client:
        with client.stream('POST', base+'/v1/messages', json=body()) as response:
            for chunk in response.iter_raw():
                if HEARTBEAT in chunk and (not during_tool or state.steps >= 3):
                    break
    deadline = time.monotonic()+5
    while not state.closed:
        assert time.monotonic() < deadline
        time.sleep(.01)
    assert state.cancelled and not state.settled and state.aborts >= 1


def test_terminal_error_closes_wire_without_tool_or_message_stop(transport):
    base, _, state = transport
    state.mode = 'error'
    wire = bytearray()
    with httpx.Client(timeout=httpx.Timeout(5, read=READ_TIMEOUT), trust_env=False) as client:
        with pytest.raises(httpx.RemoteProtocolError):
            with client.stream('POST', base+'/v1/messages', json=body()) as response:
                for chunk in response.iter_raw():
                    wire.extend(chunk)
    assert HEARTBEAT in wire
    assert b'"type":"tool_use"' not in wire and b'event: message_stop' not in wire
    assert state.closed and not state.settled


def test_json_never_contains_heartbeats(transport):
    base, _, state = transport
    with httpx.Client(timeout=5, trust_env=False) as client:
        response = client.post(base+'/v1/messages', json=body(stream=False))
    assert response.status_code == 200
    assert response.headers['content-type'] == 'application/json'
    assert HEARTBEAT not in response.content
    assert response.json() == state.response


def test_shared_chat_broker_keeps_done_and_json_events(transport, monkeypatch):
    base, backend, state = transport
    event = json.dumps(dict(id='chat-test', object='chat.completion.chunk',
        choices=[dict(index=0, delta={'content': 'OK'}, finish_reason='stop')]))

    def enqueue(prepared, tokenizer, rid):
        assert prepared.protocol == 'chat_completions'
        backend.scheduler.requests[rid] = True

    def step():
        time.sleep(SILENCE)
        rid, = backend.scheduler.requests
        backend.scheduler.requests.clear()
        state.closed = True
        return NS(outputs=[NS(request_id=rid, finished=True, error=None,
            recipe_events=[event], recipe_response=None)])

    monkeypatch.setattr(backend, '_enqueue', enqueue)
    backend.scheduler.step = step
    with httpx.Client(timeout=httpx.Timeout(5, read=READ_TIMEOUT), trust_env=False) as client:
        with client.stream('POST', base+'/v1/chat/completions', json=dict(
            model='deepseek-flash', stream=True, max_tokens=16,
            messages=[dict(role='user', content='Reply OK.')])) as response:
            assert response.status_code == 200
            wire, _ = collect(response, state)
    assert wire.startswith(HEARTBEAT)
    assert wire.replace(HEARTBEAT, b'') == (sse_frame(None, event) + sse_frame(None, '[DONE]')).encode()
