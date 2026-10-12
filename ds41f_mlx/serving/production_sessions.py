"""Logical production sessions: no native cache, parser, or executable owner.

A session retains ordinary canonical history and token identity, not physical
state. ProductionScheduler publishes outcomes only after paired settlement and
native retirement. Its PairedCheckpointAuthority owns the sole opaque pin.
POSTs are never replayed, including identical application IDs.
"""
from copy import deepcopy
from dataclasses import dataclass, field
from time import time
from uuid import UUID, uuid4
import re


@dataclass
class ProductionSession:
    session_id: str
    publication: dict = field(default_factory=lambda: dict(
        state='empty', request_count=0, frontier=0, last_turn=None,
        diagnostics=None, last_error=None))
    history: list = field(default_factory=list)
    canonical: tuple = ()
    envelope: dict | None = None
    active_request_id: str | None = None
    used_applications: set = field(default_factory=set)
    created_at: float = field(default_factory=time)
    closed: bool = False
    closing: bool = False

    @property
    def busy(self):
        return self.closing or self.active_request_id is not None

    @property
    def request_count(self):
        return self.publication['request_count']

    def to_json(self):
        value = deepcopy(self.publication)
        value.update(id=self.session_id, protocol='chat_completions',
                     state='closed' if self.closed else 'busy' if self.busy else value['state'],
                     active_request_id=self.active_request_id, created_at=self.created_at,
                     canonical_frontier=value.pop('frontier'))
        return value


class ProductionSessions:
    MAX_LIFETIMES = 4096  # bounded non-reusable requested-ID tombstones
    MAX_REQUESTS = 65536  # bounded correlation identities; no eviction/retry

    def __init__(self, backend):
        self.backend = backend
        self.records = {}
        self.issued = set()

    def create(self, session_id=None):
        if self.backend.fatal_error:
            raise RuntimeError('production engine failed; restart required')
        if self.records:
            raise RuntimeError('maximum live production sessions is one; close the current conversation')
        if len(self.issued) >= self.MAX_LIFETIMES:
            raise RuntimeError('session lifetime identity budget exhausted; no eviction/reuse')
        sid = session_id if session_id is not None else 'sess_' + uuid4().hex
        if not isinstance(sid, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', sid):
            raise ValueError('invalid session identity')
        if sid in self.issued:
            raise RuntimeError('session identity already issued; no reuse or create retry')
        record = ProductionSession(sid)
        self.issued.add(sid)
        self.records[sid] = record
        return record

    def get(self, sid):
        if sid not in self.records:
            raise KeyError('session absent or retired; no restore/replay')
        return self.records[sid]

    def validate(self, sid, prepared, body, expected=None):
        rec = self.get(sid)
        if self.backend.fatal_error or rec.publication['state'] == 'unrecoverable':
            raise RuntimeError('unrecoverable production session; close required, no replay')
        if rec.busy:
            raise RuntimeError('session has an active request; reconcile before continuation')
        if expected is not None and expected != rec.request_count:
            raise RuntimeError('stale expected request_count; reconcile, no retry')
        if len(rec.used_applications) >= self.MAX_REQUESTS:
            raise RuntimeError('session request identity budget exhausted; close required')
        if prepared.protocol != 'chat_completions':
            raise ValueError('production sessions support Chat Completions only')
        incoming = body['messages']  # recipe schema/conversion has already validated it
        if not rec.request_count and any(m.get('role') in ('assistant', 'tool') for m in incoming):
            raise ValueError('new session cannot reconstruct prior assistant/tool history; no session replay')
        from .generation_policy import reasoning_policy
        envelope = dict(tools=body.get('tools'), reasoning=reasoning_policy(prepared))
        if rec.envelope is not None and envelope != rec.envelope:
            raise ValueError('retained session protocol envelope changed')
        if incoming[:len(rec.history)] != rec.history:
            raise ValueError('history differs from this session canonical outcome')
        C = len(rec.canonical)
        if C and (len(prepared.token_ids) < C+2 or tuple(prepared.token_ids[:C]) != rec.canonical):
            raise ValueError('request must exactly extend this session canonical token frontier')
        additions = incoming[len(rec.history):]
        last = (rec.publication['last_turn'] or {}).get('response_json')
        message = last['choices'][0]['message'] if last else {}
        calls = message.get('tool_calls') or []
        if calls:
            cert = rec.publication['last_turn'].get('reconstruction') or {}
            if not cert.get('executable_tools') or len(additions) < len(calls) or any(
                m.get('role') != 'tool' or m.get('tool_call_id') != c['id']
                for m, c in zip(additions, calls)):
                raise ValueError('tool results require this session completed certified call batch')
        if any(m.get('role') == 'tool' for m in additions[len(calls):]):
            raise ValueError('foreign or duplicate tool result')
        return rec

    def reserve(self, sid, prepared, body, expected, application_id, request_id):
        if type(expected) is not int or expected < 0:
            raise ValueError('expected request_count is required')
        try:
            if str(UUID(application_id)) != application_id:
                raise ValueError()
        except (ValueError, TypeError, AttributeError):
            raise ValueError('application request correlation must be a canonical UUID') from None
        rec = self.validate(sid, prepared, body, expected)
        if application_id in rec.used_applications:
            raise RuntimeError('application request identity already reserved; observe via GET, never retry POST')
        rec.used_applications.add(application_id)
        rec.active_request_id = request_id
        return rec

    def unstarted(self, rec, request_id, error):
        if rec.active_request_id == request_id:
            rec.publication = dict(rec.publication, last_error=str(error))
            rec.active_request_id = None

    def publish(self, request, error):
        """Called by Scheduler after native retirement, never by HTTP delivery."""
        rec = request.logical_session
        trace, execution = request.trace, request.execution
        if error or 'quiescence' not in trace or not trace.get('response'):
            rec.publication = dict(rec.publication, state='unrecoverable',
                                   last_error=str(error or 'no canonical settlement'))
            return
        certificate = deepcopy(execution.certificate or {})
        certificate.pop('witness', None)
        usable = certificate.get('representable', False) and request.session_pair_retained
        response = deepcopy(trace['response'])
        finish = response['choices'][0]['finish_reason']
        cancelled = trace['cancelled']
        turn = dict(request_id=request.request_id, application_request_id=request.application_id,
                    response_json=response, cancelled=cancelled, finish_reason=finish,
                    termination_reason='user_stop' if cancelled else finish,
                    reconstruction=certificate, capacity=dict(request.prepared.capacity or {}, binding=True))
        frontier = trace['canonical_frontier']
        target, dspark = trace['target_offsets'], trace['dspark_offsets']
        diag = dict(frontier=frontier, cache_offsets_all=list(target), dspark_offsets=list(dspark),
                    all_cache_offsets_equal_frontier=set(target+dspark) == {frontier},
                    total_prompt_replay_count=trace['prompt_replay'],
                    total_full_cache_repack_count=trace['full_cache_repack'])
        turn['generation_policy'] = trace.get('generation_policy')
        turn['metrics'] = dict(diag, cached_tokens=trace.get('cached_tokens'),
                               generated=trace.get('generated'), cache_published=trace['cache_published'])
        rec.history = deepcopy(request.session_body['messages']) + [response['choices'][0]['message']]
        rec.canonical = tuple(execution.canonical)
        from .generation_policy import reasoning_policy
        rec.envelope = dict(tools=deepcopy(request.session_body.get('tools')),
                            reasoning=reasoning_policy(request.prepared))
        # Atomic public snapshot replacement: count/frontier/outcome move together.
        rec.publication = dict(state='idle' if usable else 'unrecoverable',
            request_count=rec.request_count+1, frontier=frontier, last_turn=turn,
            diagnostics={'m8': diag}, last_error=None if usable else
            'canonical outcome not reconstructible or extendable pair unavailable; no replay')
