import asyncio
import curses
import json

import httpx
import pytest

from ds41f_mlx.chat_application import ChatApplication, ApplicationError
from ds41f_mlx.operator_chat import Composer, ChatView, wrap, cells


def run(coro):
    return asyncio.run(coro)


class Backend:
    def __init__(self):
        self.count = 0
        self.turn = None
        self.requests = []
        self.mode = 'text'
        self.tools = None
        self.closed = False
        self.bad_identity = False

    def handle(self, req):
        value = json.loads(req.content or b'{}')
        self.requests.append((req.method, req.url.path, value))
        path = req.url.path
        if path == '/api/status':
            return httpx.Response(200, json={'runtime': {'status': 'ready', 'profile': 'standard-off'}, 'tools': [{'type': 'function', 'function': {'name': 'tool'}}]})
        if path == '/api/session':
            return httpx.Response(200, json={'id': value['id']})
        if path.startswith('/api/session/'):
            if req.method == 'DELETE':
                self.closed = True
            return httpx.Response(200, json={'state': 'idle', 'request_count': self.count, 'last_turn': self.turn})
        if path == '/api/budget':
            return httpx.Response(200, json={'request_count': self.count})
        if path == '/api/stream':
            assert value['expected_count'] == self.count
            assert value['request']['reasoning_effort'] == 'none'
            self.count += 1
            message = {'role': 'assistant', 'content': 'canonical 日本語', 'reasoning_content': 'reason'}
            if self.mode == 'tools' and self.count == 1:
                message['tool_calls'] = [{'id': 'call1', 'type': 'function', 'function': {'name': 'tool', 'arguments': '{}'}}]
            self.turn = {'application_request_id': 'wrong' if self.bad_identity else value['application_id'],
                         'request_id': 'r' + str(self.count), 'response_json': {'choices': [{'message': message, 'finish_reason': 'stop'}]}}
            # Truncated/error stream: visible text must be replaced by canonical GET.
            return httpx.Response(200, text='event: admitted\ndata: ' + json.dumps({'request_id': self.turn['request_id']}) + '\n\ndata: ' + json.dumps({'choices': [{'delta': {'content': 'PARTIAL'}}]}) + '\n\nevent: error\ndata: {"message":"disconnect"}\n\n')
        if path == '/api/tools':
            self.tools = {'messages': [{'role': 'tool', 'tool_call_id': 'call1', 'content': 'result'}], 'results': []}
            if self.mode == 'uncertain':
                self.tools = None
                return httpx.Response(500, json={'error': {'message': 'lost'}})
            return httpx.Response(200, json=self.tools)
        if path == '/api/tools/result':
            return httpx.Response(200, json={'result': self.tools})
        raise AssertionError(path)


@pytest.mark.parametrize('profile', ['mtp-serving-v1', 'mtp-singleton-v1'])
def test_unsupported_profile_is_explicit_without_session_or_sampler_fallback(profile):
    async def check():
        requests = []
        def handler(req):
            requests.append(req.url.path)
            return httpx.Response(200, json={'runtime': {'status': 'ready', 'profile': profile}, 'tools': []})
        app = ChatApplication(transport=httpx.MockTransport(handler))
        with pytest.raises(ApplicationError, match='Unsupported runtime stateful_chat capability'):
            await app.new()
        assert app.id is None and requests == ['/api/status']
        await app.shutdown()
    run(check())


def test_create_failure_retains_known_id_and_reports_actual_error():
    async def check():
        def handler(req):
            if req.url.path == '/api/status':
                return httpx.Response(200, json={'runtime': {'status': 'ready', 'profile': 'standard-off'}, 'tools': []})
            return httpx.Response(404, json={'error': {'message': 'native session endpoint not_found'}})
        app = ChatApplication(transport=httpx.MockTransport(handler))
        with pytest.raises(ApplicationError):
            await app.new()
        assert app.id and 'not_found' in app.problem and 'no retry' in app.problem
        await app.shutdown()
    run(check())


def test_long_paste_crlf_and_actions_remain_text_only():
    editor = Composer()
    text = '日本語 abc\r\n' * 1000
    for char in '\x1b[200~' + text + '\x1b[201~':
        assert editor.decode(char) is None
    assert editor.text == text.replace('\r\n', '\n')
    assert not editor.pasting


def test_production_protocol_multiturn_replaces_partial_and_retires_session():
    async def check():
        backend = Backend()
        app = ChatApplication(transport=httpx.MockTransport(backend.handle))
        await app.new()
        await app.send('こんにちは')
        assert app.messages[-1]['content'] == 'canonical 日本語'
        assert app.live is None and app.pending is None and app.count == 1
        await app.send('次の質問')
        assert app.count == 2 and len(app.messages) == 4
        stream = [v for _, p, v in backend.requests if p == '/api/stream']
        assert len(stream[1]['request']['messages']) == 3
        assert stream[0]['application_id'] != stream[1]['application_id']
        assert not any(p.startswith('/v1/') or p == '/api/chat' for _, p, _ in backend.requests)
        await app.shutdown()
        assert backend.closed
    run(check())


def test_identity_mismatch_never_commits_or_resends():
    async def check():
        backend = Backend(); backend.bad_identity = True
        app = ChatApplication(transport=httpx.MockTransport(backend.handle))
        await app.new()
        with pytest.raises(ApplicationError):
            await app.send('test')
        assert app.pending and app.live and len(app.messages) == 1
        with pytest.raises(ApplicationError):
            await app.send('not retried')
        assert sum(p == '/api/stream' for _, p, _ in backend.requests) == 1
        await app.shutdown()
    run(check())


def test_server_tool_result_and_safe_model_continuation():
    async def check():
        backend = Backend(); backend.mode = 'tools'
        app = ChatApplication(transport=httpx.MockTransport(backend.handle))
        await app.new(); await app.send('use tool')
        assert [m['role'] for m in app.messages] == ['user', 'assistant', 'tool', 'assistant']
        assert app.count == 2 and not app.awaiting_response
        assert sum(p == '/api/tools' for _, p, _ in backend.requests) == 1
        await app.shutdown()
    run(check())


def test_uncertain_effect_is_observed_never_reexecuted():
    async def check():
        backend = Backend(); backend.mode = 'tools'
        app = ChatApplication(transport=httpx.MockTransport(backend.handle))
        await app.new()
        await app.new()
        await app.generate()  # settle tool call without invoking the loop
        backend.mode = 'uncertain'
        with pytest.raises(ApplicationError):
            await app.tool_loop()
        assert app.effects['state'] == 'reserved' and app.problem
        with pytest.raises(ApplicationError):
            await app.resume()
        assert sum(p == '/api/tools' for _, p, _ in backend.requests) == 1
        await app.shutdown()
    # Generate requires a user prefix (mock backend intentionally ignores recipe).
    run(check())


def test_unicode_paste_never_dispatches_actions_and_preserves_resize():
    editor = Composer()
    for char in '\x1b[200~こんにちは\nsecond line\x07\x03\x1b[201~':
        key = editor.decode(char)
        assert key is None
    assert editor.text == 'こんにちは\nsecond line'
    assert not editor.pasting
    editor.edit('\n', curses)
    editor.edit('終', curses)
    editor.edit(curses.KEY_LEFT, curses)
    editor.edit(curses.KEY_BACKSPACE, curses)
    assert editor.text.endswith('line終')
    assert editor.decode('\x07') == '\x07'
    assert wrap('日本語abc', 4) == ['日本', '語ab', 'c']
    app = ChatApplication()
    view = ChatView(); view.editor = editor
    app.messages = [{'role': 'assistant', 'content': '日本語\n' * 100, 'reasoning_content': 'thought'}]
    before = (editor.text, editor.cursor)
    view.render(app, 80, 24, 'READY', 'READY', True)
    view.scroll(-10)
    assert view.top is not None
    lines, cursor = view.render(app, 40, 10, 'READY', 'READY', True)
    assert len(lines) <= 10 and before == (editor.text, editor.cursor)
    assert 'scroll paused' in lines[0].text
    assert all(sum(cells(c) for c in s) <= 4 for s in wrap('が日本', 4))
    run(app.http.aclose())
