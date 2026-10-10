"""Disposable, content-free diagnostic projection; never execution authority."""
from __future__ import annotations

from copy import deepcopy
from threading import Lock
from time import time, monotonic

# Explicit allowlist: never publish a native object, response, or arbitrary trace.
TRACE_FIELDS = ('cached_tokens', 'remaining_suffix_tokens', 'generated', 'decode_s',
                'recipe_convert_s', 'recipe_render_s', 'recipe_encode_s',
                'request_prepare_s', 'cache_lookup_s', 'paired_restore_s',
                'prompt_checkpoint_capture_s', 'suffix_append_s', 'p5_handoff_s',
                'first_decode_call_s', 'first_native_decode_s', 'total_ttft_s',
                'prompt_replay', 'full_cache_repack', 'canonical_frontier')


class Projection:
    def __init__(self):
        self._lock = Lock()
        self.state = 'STARTING'
        self.requests = {}
        self.last = None
        self.cache = None
        self.endpoint = None

    def lifecycle(self, state, endpoint=None):
        try:
            with self._lock:
                self.state = state
                if endpoint is not None:
                    self.endpoint = endpoint
        except Exception:
            pass

    def begin(self, rid):
        try:
            with self._lock:
                now, mono = time(), monotonic()
                self.requests[rid] = dict(request_id=rid, request_received_at=now,
                    request_started_at=now, request_started_mono=mono,
                    current_phase='RECEIVED', phase_started_at=now, phase_started_mono=mono,
                    last_progress_at=now, completed_phase_durations={}, metrics={})
        except Exception:
            pass

    def phase(self, rid, phase, queue_reason=None):
        try:
            with self._lock:
                r = self.requests.get(rid)
                if r is None or r['current_phase'] == phase:
                    return
                now, mono = time(), monotonic()
                durations = r['completed_phase_durations']
                old = r['current_phase'].lower() + '_ms'
                durations[old] = durations.get(old, 0.) + (mono-r['phase_started_mono'])*1000
                r.update(current_phase=phase, phase_started_at=now,
                         phase_started_mono=mono, last_progress_at=now)
                if phase == 'QUEUED':
                    r['queue_entered_at'], r['queue_entered_mono'] = now, mono
                    r['queue_reason'] = queue_reason or 'scheduler_worker'
        except Exception:
            pass

    def metrics(self, rid, trace=None, prompt_tokens=None):
        try:
            values = {k: trace[k] for k in TRACE_FIELDS if k in (trace or {})
                      and isinstance(trace[k], (int, float, bool))}
            if prompt_tokens is not None:
                values['prompt_tokens'] = int(prompt_tokens)
            if trace and 'target_offsets' in trace and 'dspark_offsets' in trace:
                offsets = trace['target_offsets'] + trace['dspark_offsets']
                values['aligned'] = bool(offsets) and set(offsets) == {trace.get('canonical_frontier')}
            stats = (trace or {}).get('mtp_stats', {})
            if stats:
                values['accepted_drafts'] = sum(stats.get('depth_accepted', []))
                values['proposed_drafts'] = sum(stats.get('depth_drafted', []))
            with self._lock:
                r = self.requests.get(rid)
                if r is not None:
                    previous_generated = r['metrics'].get('generated', 0)
                    r['metrics'].update(values)
                    r['last_progress_at'] = time()
                    if values.get('generated', 0) > previous_generated:
                        r.setdefault('first_token_at', time())
                        r['last_token_at'] = time()
        except Exception:
            pass

    def finish(self, rid, finish_reason='error'):
        self.phase(rid, 'ERROR' if finish_reason == 'error' else 'DONE')
        try:
            with self._lock:
                r = self.requests.pop(rid, None)
                if r is not None:
                    r.update(finish_reason=finish_reason, finished_at=time(), finished_mono=monotonic())
                    self.last = r
        except Exception:
            pass

    def cache_summary(self, summary):
        try:
            # Scalar stats only, copied by the authority's worker.
            values = {k: v for k, v in summary.items() if isinstance(v, (int, float, bool))}
            with self._lock:
                self.cache = values
        except Exception:
            pass

    def snapshot(self):
        with self._lock:
            requests = list(self.requests.values())
            queued = [r for r in requests if r['current_phase'] == 'QUEUED']
            active = [r for r in requests if r['current_phase'] != 'QUEUED']
            current = next((r for r in active if r['current_phase'] not in ('RECEIVED', 'ENCODING')), None)
            current = current or (active[0] if active else queued[0] if queued else None)
            return deepcopy(dict(schema='ds41f.live.v1', sampled_at=time(), sampled_mono=monotonic(),
                state=self.state, endpoint=self.endpoint, current_request=current,
                requests=requests, queued_requests=len(queued), active_requests=len(active),
                last_request=self.last, cache=self.cache))


def phase(backend, trace, name):
    projection = getattr(backend, 'telemetry', None)
    if projection is not None:
        projection.phase(trace.get('diagnostic_id'), name)


def metrics(backend, trace):
    projection = getattr(backend, 'telemetry', None)
    if projection is not None:
        projection.metrics(trace.get('diagnostic_id'), trace)
