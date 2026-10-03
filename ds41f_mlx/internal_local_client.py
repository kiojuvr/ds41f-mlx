"""Experimental living-client M36R workflow over the existing local HTTP boundary.

No parser/token/cache authority, persistence, shared use or automatic replacement
work. SSE is display-only; every settlement uses the same certificate path.
"""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import socket
import time

from .web_client import RuntimeClient, RuntimeHTTPError


class ClientStateError(RuntimeError):
    pass


@dataclass(frozen=True)
class RequestIdentity:
    session_id: str
    sequence: int
    body: bytes

    def to_json(self):
        return dict(session_id=self.session_id, sequence=self.sequence,
                    body_utf8=self.body.decode(), body_sha256=hashlib.sha256(self.body).hexdigest())


class OwnedStream:
    """Close/drop interrupts the real socket; partial SSE/UTF8 is never history."""
    def __init__(self, client, conn, response):
        self.client, self.conn, self.response = client, conn, response
        self.closed = False
        client.state = 'streaming'

    def __iter__(self):
        try:
            while True:
                line = self.response.readline()
                if not line:
                    break
                if line.startswith(b'data: '):
                    # A complete display event only. Never accumulate assistant history.
                    value = line[6:].strip()
                    if value == b'[DONE]':
                        break
                    yield json.loads(value.decode('utf-8'))
        finally:
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        sock = self.conn.sock
        if sock is None and self.response.fp is not None:
            sock = self.response.fp.raw._sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.response.close()
        self.conn.close()
        if self.client.state != 'stopped':
            self.client.state = 'ambiguous'
        self.client._stream = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


class InternalLocalClient:
    """One thread, one live client. Explicit reconcile/retire/create actions only.

    Request bytes are frozen before sending. One current outcome is copied before
    advancing; the bounded effect ledger survives session retirement in this object.
    The transcript is ordinary application messages, never a native history oracle.
    """
    def __init__(self, runtime: RuntimeClient, *, ledger_limit=128):
        self.runtime = runtime
        self.state = 'retired'
        self.session_id = None
        self.next_sequence = None
        self._messages = []
        self._pending = None
        self._outcome = None
        self._stream = None
        self._ledger = {}
        if type(ledger_limit) is not int or not 0 <= ledger_limit <= 128:
            raise ValueError('ledger capacity must be between 0 and 128')
        self.ledger_limit = ledger_limit
        self.timings = deque(maxlen=128)

    @property
    def messages(self):
        return deepcopy(self._messages)

    @property
    def identity(self):
        return self._pending

    @property
    def outcome(self):
        return deepcopy(self._outcome)

    @property
    def ledger(self):
        return deepcopy(self._ledger)

    def _stop(self, message):
        self.state = 'stopped'
        raise ClientStateError(message)

    def create(self, session_id=None):
        if self.state != 'retired':
            self._stop('must explicitly retire before fresh session creation')
        t0 = time.perf_counter()
        try:
            rec = self.runtime.create_session(session_id)
            if not isinstance(rec.get('id'), str) or rec.get('outcome_state') != 'not_admitted' or rec.get('next_sequence') != 1:
                raise ClientStateError('fresh session did not establish the internal fence contract')
        except BaseException:
            self.state = 'stopped'
            raise
        self.session_id = rec['id']
        self.next_sequence = 1
        self._pending = self._outcome = None
        self.state = 'ready'
        self.timings.append(dict(action='create', seconds=time.perf_counter()-t0))
        return dict(id=self.session_id, outcome_state='not_admitted', next_sequence=1)

    def submit(self, additions, *, options):
        if self.state != 'ready':
            self._stop('unresolved request or tool ownership; new work forbidden')
        if 'messages' in options:
            self._stop('options cannot replace owned conversation')
        if not isinstance(additions, list) or any(m.get('role') != 'user' for m in additions):
            self._stop('submit accepts ordinary user additions; tool results are ledger owned')
        body = deepcopy(options)
        body['messages'] = self.messages + deepcopy(additions)
        raw = json.dumps(body, ensure_ascii=False).encode()
        if len(raw) > 1048576:
            self._stop('bounded local request body exceeds 1 MiB')
        self._pending = RequestIdentity(self.session_id, self.next_sequence, raw)
        self._outcome = None
        return self._send()

    def _send(self):
        self.state = 'in_flight'
        identity = self._pending
        t0 = time.perf_counter()
        try:
            conn, resp = self.runtime.internal_fenced_request(identity.session_id, identity.body, identity.sequence)
            if resp.getheader('Content-Type', '').startswith('text/event-stream'):
                stream = OwnedStream(self, conn, resp)
                # Client does not retain the stream: dropping the handle closes it.
                return stream
            try:
                try:
                    value = json.loads(resp.read().decode('utf-8'))
                    if not isinstance(value, dict):
                        raise ValueError('JSON object required')
                except (ValueError, UnicodeError) as exc:
                    self._stop(f'malformed HTTP outcome: {exc}')
            finally:
                resp.close(); conn.close()
            self.state = 'ambiguous'
            # New non-streaming work returns an ordinary response, NOT a certificate.
            if 'outcome_state' in value:
                self._accept(value)
            return value
        except RuntimeHTTPError as exc:
            if exc.status == 409:
                self.state = 'ambiguous'
            else:
                self._stop(f'fenced request rejected; client/admission state error: {exc}')
            raise
        except ClientStateError:
            raise
        except BaseException:
            self.state = 'ambiguous'
            raise
        finally:
            self.timings.append(dict(action='send_headers_or_json', seconds=time.perf_counter()-t0,
                                     sequence=identity.sequence))

    def reconcile(self):
        """One lookup, no hidden polling/generation. Active => caller waits.

        Absence/not_admitted cannot ACK a delayed POST: resend ONLY frozen bytes.
        A settled slot is re-observed by exact-byte POST (JSON, never new work).
        """
        if self.state in ('retired', 'stopped', 'streaming', 'in_flight') or self._pending is None:
            self._stop('cannot reconcile in this state; close the stream first')
        t0 = time.perf_counter()
        try:
            rec = self.runtime.get_session(self.session_id)
            if not isinstance(rec, dict):
                self._stop('malformed session observation')
            slot = rec.get('request_fence')
            if slot is not None and (not isinstance(slot, dict) or type(slot.get('sequence')) is not int):
                self._stop('malformed request fence')
            seq = self._pending.sequence
            if type(rec.get('next_sequence')) is not int:
                self._stop('malformed sequence observation')
            if slot and slot.get('sequence') > seq:
                self.state = 'expired'
                return dict(outcome_state='expired', sequence=seq,
                            action='explicit restart or external canonical reconciliation required; no regeneration')
            if slot and (slot.get('sequence') != seq or slot.get('body_sha256') != hashlib.sha256(self._pending.body).hexdigest()):
                self._stop('sequence/body disagreement')
            state = rec.get('outcome_state')
            if state == 'active':
                if slot is None or slot.get('state') != 'active' or rec['next_sequence'] != seq:
                    self._stop('active identity disagreement')
                self.state = 'ambiguous'
                return dict(outcome_state='active', sequence=seq)
            if state == 'not_admitted':
                if rec['next_sequence'] != seq:
                    self._stop('unstarted sequence disagreement')
                return self._send()
            if state not in ('recoverable', 'unrecoverable', 'poisoned') or rec['next_sequence'] != seq+1 or slot is None or slot.get('state') != state:
                self._stop('settled sequence disagreement')
            return self._send()
        finally:
            self.timings.append(dict(action='reconcile_lookup_and_observe', seconds=time.perf_counter()-t0))

    def observe_identity(self, identity):
        """Explicit older-outcome query. Never regenerate an expired identity."""
        if identity.session_id != self.session_id or self.state == 'retired':
            raise ClientStateError('retired/wrong session identity cannot resume')
        if identity == self._pending:
            return self.reconcile()
        rec = self.runtime.get_session(self.session_id)
        slot = rec.get('request_fence')
        if slot and slot.get('sequence', 0) > identity.sequence:
            return dict(outcome_state='expired', sequence=identity.sequence,
                        action='use current certified conversation or explicitly restart; no regeneration')
        self._stop('wrong request identity; not current or provably expired')

    def _accept(self, out):
        seq = self._pending.sequence
        if out.get('sequence') != seq or out.get('outcome_state') not in ('recoverable', 'unrecoverable', 'poisoned'):
            self._stop('malformed settled outcome identity')
        if self._outcome is not None:
            if out != self._outcome:
                self._stop('settled outcome changed')
            if out['outcome_state'] == 'recoverable':
                calls = out['response']['choices'][0]['message'].get('tool_calls') or []
                completed = calls and all(self._ledger.get((self.session_id, seq, i, c['id']), {}).get('status') == 'completed' for i,c in enumerate(calls))
                self.state = ('tool_completed' if completed else 'tool_pending') if calls else 'ready'
            else:
                self.state = out['outcome_state']
            return
        state = out['outcome_state']
        cert = out.get('certificate')
        if state == 'recoverable':
            try:
                if not (type(cert['version']) is int and cert['version'] == 1 and
                        type(cert['frontier']) is int and cert['frontier'] >= 0 and
                        type(cert['encoded_length']) is int and type(cert['executable_tools']) is bool and
                        cert['representable'] is True and
                        cert['exact_prefix'] is True and cert['semantic_complete'] is True and
                        cert['first_mismatch'] is None and cert['encoded_length'] > cert['frontier'] and
                        isinstance(cert['canonical_sha256'], str) and len(cert['canonical_sha256']) == 64 and
                        all(c in '0123456789abcdef' for c in cert['canonical_sha256'])):
                    raise ValueError('invalid certificate')
                choices = out['response']['choices']
                if len(choices) != 1 or choices[0]['message']['role'] != 'assistant':
                    raise ValueError('invalid ordinary assistant')
                msg = deepcopy(choices[0]['message'])
                calls = msg.get('tool_calls') or []
                if bool(calls) != cert['executable_tools'] or (calls and choices[0]['finish_reason'] != 'tool_calls'):
                    raise ValueError('tool certificate mismatch')
                request = json.loads(self._pending.body)
                # Check ordinary witness binding, not tokens or recipe re-encoding.
                if cert['witness']['messages'][:len(request['messages'])+1] != request['messages']+[msg]:
                    raise ValueError('certificate ordinary message binding mismatch')
                if calls and (len({c['id'] for c in calls}) != len(calls) or
                              any(c['type'] != 'function' or not isinstance(c['function']['arguments'], str) for c in calls)):
                    raise ValueError('invalid completed calls')
            except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
                self._stop(f'malformed recovery outcome: {exc}')
            self._messages = request['messages'] + [msg]  # replace, never concatenate SSE
            self.state = 'tool_pending' if calls else 'ready'
            self.next_sequence = seq+1
        elif state == 'unrecoverable':
            if not isinstance(cert, dict) or cert.get('representable') is not False:
                self._stop('invalid negative certificate')
            # Request messages are legitimate application history; final native
            # assistant has no certified ordinary representation and is NOT inserted.
            self._messages = json.loads(self._pending.body)['messages']
            self.state = 'unrecoverable'
        else:
            self._messages = json.loads(self._pending.body)['messages']
            self.state = 'poisoned'
        self._outcome = deepcopy(out)

    def execute_tools(self, execute):
        if self.state not in ('tool_pending', 'tool_completed') or not self._outcome:
            self._stop('no owned certified completed tool outcome')
        t0 = time.perf_counter()
        out = self._outcome
        identity = self._pending
        if out['sequence'] != identity.sequence or out['outcome_state'] != 'recoverable' or not out['certificate']['executable_tools']:
            self._stop('tool outcome identity mismatch')
        calls = out['response']['choices'][0]['message']['tool_calls']
        missing = sum((identity.session_id, identity.sequence, i, c['id']) not in self._ledger for i,c in enumerate(calls))
        if len(self._ledger) + missing > self.ledger_limit:
            self._stop('bounded tool ledger full; no eviction/re-execution')
        results = []
        for index, call in enumerate(calls):
            key = (identity.session_id, identity.sequence, index, call['id'])
            entry = self._ledger.get(key)
            if entry is not None and entry['call'] != call:
                self._stop('tool ledger identity mismatch')
            if entry is None:
                if len(self._ledger) >= self.ledger_limit:
                    self._stop('bounded tool ledger full; no eviction/re-execution')
                entry = self._ledger[key] = dict(call=deepcopy(call), status='reserved', result=None)
                try:
                    content = execute(deepcopy(call))
                    if not isinstance(content, str) or len(content.encode()) > 65536:
                        raise ClientStateError('tool must return ordinary string content bounded to 64 KiB')
                    entry.update(status='completed', result=dict(role='tool', tool_call_id=call['id'], content=content))
                except BaseException:
                    self.state = 'tool_ambiguous'
                    raise
            if entry['status'] != 'completed':
                self._stop('reserved effect ambiguous; automatic retry forbidden')
            results.append(deepcopy(entry['result']))
        if self.state != 'tool_completed':
            self._messages.extend(results)
        self.state = 'tool_completed'
        self.timings.append(dict(action='tool_ledger_and_effect', seconds=time.perf_counter()-t0))
        return results

    def submit_tool_results(self, *, options):
        if self.state != 'tool_completed':
            self._stop('stored ordinary tool results required')
        self.state = 'ready'
        return self.submit([], options=options)

    def retire(self):
        if self.state in ('streaming', 'in_flight', 'ambiguous'):
            self._stop('resolve/close active request before DELETE')
        t0 = time.perf_counter()
        try:
            rec = self.runtime.close_session(self.session_id)
            if rec.get('state') != 'closed':
                raise ClientStateError('DELETE not confirmed')
        except BaseException:
            self.state = 'stopped'
            raise
        self.state = 'retired'
        self.next_sequence = None
        self._pending = self._outcome = None
        self.timings.append(dict(action='delete', seconds=time.perf_counter()-t0))
        return dict(id=self.session_id, state='closed')

    def retained_state_bytes(self):
        """Serialized payload estimate, not Python allocator/RSS accounting."""
        return len(json.dumps(dict(messages=self._messages, outcome=self._outcome,
                                   ledger=[dict(key=list(k), **v) for k,v in self._ledger.items()])).encode()) + (len(self._pending.body) if self._pending else 0)
