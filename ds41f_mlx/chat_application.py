"""Text application frontend for the production Chat Server protocol.

Only ordinary history lives here. Native sessions, admission, canonical responses
and tool-effect reservations remain authoritative on the servers. Never retry a
POST whose outcome is unknown. This intentionally has no persistence/restore.
"""
from __future__ import annotations

import asyncio
import json
from uuid import uuid4
from urllib.parse import quote

import httpx


class ApplicationError(RuntimeError):
    def __init__(self, message, *, status=None, effect_not_started=False):
        super().__init__(message)
        self.status = status
        self.effect_not_started = effect_not_started


class ChatApplication:
    def __init__(self, base_url='http://127.0.0.1:8080', *, transport=None):
        self.http = httpx.AsyncClient(base_url=base_url, timeout=1800, transport=transport, trust_env=False)
        self.id = None
        self.messages = []
        self.count = 0
        self.pending = self.effects = self.live = None
        self.tools = []
        self.defaults = None
        self.settings = None
        self.runtime = None
        self.record = None
        self.problem = ''
        self.notice = 'New chat: Ctrl-N (in-memory only)'
        self.stopped = False
        self.awaiting_response = False
        self.stream_task = None

    @property
    def path(self):
        return '/api/session/' + quote(self.id, safe='')

    async def api(self, path, method='GET', value=None):
        response = await self.http.request(method, path, json=value)
        if not response.is_success:
            try:
                error = response.json()['error']
            except (ValueError, KeyError):
                error = {'message': response.text}
            raise ApplicationError(error['message'], status=response.status_code,
                                   effect_not_started=error.get('effect_not_started', False))
        return response.json()

    async def new(self):
        if self.id:
            await self.close()
        status = await self.api('/api/status')
        runtime = status.get('runtime') or {}
        from .chat_capability import Settings
        try:
            previous = self.settings.values if self.settings else None
            self.settings = Settings(runtime, previous)
            self.defaults = self.settings.defaults
        except ValueError as exc:
            raise ApplicationError(str(exc)) from exc
        self.runtime = runtime
        production = (runtime.get('capabilities') or {}).get('stateful_chat') == 'production-sessions-v1'
        self.tools = [tool for tool in status['tools'] if not production or
                      tool.get('function', {}).get('name') not in ('fetch_image', 'fetch_pdf')]
        # Retain the known ID even if the create acknowledgement is lost.
        self.id = 'tui-' + str(uuid4())
        self.messages = []
        self.count = 0
        self.pending = self.effects = self.live = None
        self.awaiting_response = False
        self.problem = 'Session creation not yet reconciled; no automatic retry'
        try:
            await self.api('/api/session', 'POST', {'id': self.id})
            await self.reconcile()
        except Exception as exc:
            self.problem = 'Session creation unconfirmed: ' + str(exc) + '; reconcile or close, no retry'
            raise

    async def close(self):
        if not self.id:
            return
        try:
            await self.api(self.path, 'DELETE')
        except ApplicationError as exc:
            if exc.status != 404:
                raise
        self.id = None
        self.messages = []
        self.pending = self.effects = self.live = None
        self.problem = ''
        self.awaiting_response = False
        self.notice = 'Chat closed. Ctrl-N: new chat'

    def request(self):
        if self.defaults is None:
            raise ApplicationError('Runtime defaults have not been negotiated')
        return dict(model='deepseek-v4.1-flash', messages=self.messages, stream=True,
                    **self.settings.request(), tools=self.tools, tool_choice='auto')

    def calls(self):
        last = next((m for m in reversed(self.messages) if m['role'] == 'assistant'), {})
        return [c for c in last.get('tool_calls', []) if not any(
            m.get('tool_call_id') == c['id'] for m in self.messages)]

    def append_results(self, result):
        for message in result['messages']:
            if not any(m.get('tool_call_id') == message['tool_call_id'] for m in self.messages):
                self.messages.append(message)
        self.effects = {'count': self.count, 'state': 'completed'}
        self.awaiting_response = True
        self.notice = result.get('budget_error') or next(
            (str(r['error']) for r in result.get('results', []) if r.get('error')), '')

    async def reconcile(self, wait=False):
        if not self.id:
            raise ApplicationError('No chat session. Ctrl-N: new chat')
        self.notice = 'Reconciling canonical outcome…'
        try:
            for i in range(601):
                rec = await self.api(self.path)
                if not wait or rec['state'] != 'busy' or i == 600:
                    break
                await asyncio.sleep(.1)
            self.record = rec
            self.problem = ''
            if rec['state'] not in ('empty', 'idle'):
                self.problem = 'Session ' + rec['state'] + '; continuation disabled'
            elif self.pending and rec['request_count'] == self.pending['base'] + 1:
                turn = rec.get('last_turn') or {}
                if (turn.get('application_request_id') != self.pending['nonce'] or
                    (self.pending.get('request_id') and turn.get('request_id') != self.pending['request_id'])):
                    raise ApplicationError('Canonical request identity mismatch; no replay')
                response = turn.get('response_json')
                if not response:
                    raise ApplicationError('Canonical response missing; no replay')
                self.messages.append(response['choices'][0]['message'])
                self.count = rec['request_count']
                self.pending = self.live = None
                self.awaiting_response = False
                self.notice = 'Stopped; canonical committed response recovered' if turn.get('cancelled') or self.stopped else 'Canonical response settled'
            elif rec['request_count'] != self.count:
                self.problem = 'Application/runtime count mismatch; no replay'
            elif self.pending:
                self.problem = 'No settled outcome for frozen request; no automatic retry. Reconcile or close chat.'
            else:
                self.notice = 'Canonical session ' + rec['state']
            if self.effects and self.effects['state'] == 'reserved' and not self.problem:
                result = await self.api(
                    '/api/tools/result?session_id=' + quote(self.id, safe='') + '&request_count=' + str(self.effects['count']))
                if result['result']:
                    self.append_results(result['result'])
                else:
                    self.problem = 'Tool effect outcome unavailable; NEVER re-executed. Reconcile or close chat.'
            if self.problem:
                self.notice = self.problem
            return rec
        except Exception as exc:
            self.problem = 'Reconciliation unavailable: ' + str(exc) + '; no resend/replay'
            self.notice = self.problem
            raise

    async def send(self, text, on_submitted=None):
        self.stopped = False
        await self.reconcile()
        if self.stopped:
            self.notice = 'Stopped before sending; draft retained'
            return
        self.check_continuation()
        if self.calls() or self.awaiting_response:
            raise ApplicationError('Pending tool continuation: Ctrl-E before another user turn')
        self.settings.begin_turn()
        self.messages.append({'role': 'user', 'content': text})
        self.pending = {'base': self.count, 'nonce': str(uuid4())}
        if on_submitted:
            on_submitted()
        await self.generate()
        await self.tool_loop()

    def check_continuation(self):
        if self.problem or self.pending or (self.effects and self.effects['state'] == 'reserved'):
            raise ApplicationError(self.problem or 'Unresolved request/effect; reconcile or close')

    async def generate(self):
        from .chat_capability import negotiate
        runtime = (await self.api('/api/status')).get('runtime') or {}
        try:
            negotiate(runtime)
            if runtime['generation_policy'] != self.settings.contract:
                raise ValueError('Runtime generation contract changed; close and start a new chat')
        except ValueError as exc:
            raise ApplicationError(str(exc)) from exc
        self.runtime = runtime
        request = self.request()
        await self.api('/api/budget', 'POST', dict(session_id=self.id, expected_count=self.count, request=request))
        if self.stopped:
            return
        self.pending = self.pending or {'base': self.count, 'nonce': str(uuid4())}
        self.live = {'role': 'assistant', 'content': '', 'reasoning_content': ''}
        self.notice = 'Streaming (unsettled)…'
        async def consume():
            self.settings.freeze_reasoning()
            async with self.http.stream('POST', '/api/stream', json=dict(
                session_id=self.id, expected_count=self.count,
                application_id=self.pending['nonce'], request=request)) as response:
                if not response.is_success:
                    raise ApplicationError((await response.aread()).decode())
                frame = []
                async for line in response.aiter_lines():
                    if line:
                        frame.append(line)
                        continue
                    event = next((x[6:].strip() for x in frame if x.startswith('event:')), '')
                    data = '\n'.join(x[5:].lstrip() for x in frame if x.startswith('data:'))
                    frame = []
                    if not data or data == '[DONE]':
                        continue
                    value = json.loads(data)
                    if event == 'admitted':
                        self.pending['request_id'] = value['request_id']
                    elif event == 'error':
                        raise ApplicationError(value['message'])
                    else:
                        delta = (value.get('choices') or [{}])[0].get('delta', {})
                        for field in ('content', 'reasoning_content'):
                            self.live[field] += delta.get(field) or ''
                        for call in delta.get('tool_calls') or []:
                            calls = self.live.setdefault('tool_calls', [])
                            while len(calls) <= call['index']:
                                calls.append({'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                            item = calls[call['index']]
                            if call.get('id'):
                                item['id'] = call['id']
                            for field in ('name', 'arguments'):
                                item['function'][field] += call.get('function', {}).get(field) or ''
                if frame:
                    raise ApplicationError('Incomplete SSE frame')
        self.stream_task = asyncio.create_task(consume())
        try:
            await self.stream_task
        except asyncio.CancelledError:
            self.notice = 'Stream interrupted; canonical reconciliation required'
        except Exception as exc:
            self.notice = str(exc)
        finally:
            self.stream_task = None
            # Live partial output is kept labelled unsettled until GET replaces it.
            await self.reconcile(wait=True)

    async def tool_loop(self):
        for _ in range(128):
            if self.stopped or self.problem or not self.calls():
                return
            self.check_continuation()
            self.notice = 'Using server tool…'
            self.effects = {'count': self.count, 'state': 'reserved'}
            try:
                result = await self.api('/api/tools', 'POST', dict(
                    session_id=self.id, request_count=self.count, request=self.request()))
            except Exception as exc:
                if isinstance(exc, ApplicationError) and exc.effect_not_started:
                    self.effects = None
                await self.reconcile(wait=True)
                raise ApplicationError('Tool request failed; no automatic retry: ' + str(exc)) from exc
            self.append_results(result)
            if self.stopped:
                self.notice = 'Stopped after tool settlement; Ctrl-E to continue'
                return
            if self.notice:
                return
            await self.generate()
        self.notice = 'Tool loop circuit breaker; Ctrl-E to explicitly continue'

    async def resume(self):
        self.stopped = False
        await self.reconcile()
        if self.stopped:
            self.notice = 'Stopped before continuation'
            return
        self.check_continuation()
        if not self.calls():
            if not self.awaiting_response:
                raise ApplicationError('No pending model continuation')
            await self.generate()
        await self.tool_loop()

    async def stop(self):
        self.stopped = True
        self.notice = 'Stopping at safe boundary…'
        try:
            if self.id:
                record = await self.api(self.path)
                request_id = (self.pending or {}).get('request_id') or record.get('active_request_id')
                if request_id:
                    await self.api(self.path + '/cancel', 'POST', {'request_id': request_id})
        finally:
            if self.stream_task:
                self.stream_task.cancel()
            elif self.id:
                await self.reconcile(wait=True)

    async def shutdown(self):
        # DELETE is session retirement, never service shutdown. Keep failures visible.
        try:
            await self.close()
        finally:
            await self.http.aclose()
