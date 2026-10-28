"""Disposable, content-free diagnostic projection; never execution authority."""
from __future__ import annotations

import logging
from copy import deepcopy
from collections import deque
import json
import re
from threading import Lock
from time import time, monotonic

logger = logging.getLogger('ds41f.operator.runtime')

# Explicit allowlist: never publish a native object, response, or arbitrary trace.
TRACE_FIELDS = ('cached_tokens', 'remaining_suffix_tokens', 'generated', 'decode_s',
                'recipe_convert_s', 'recipe_render_s', 'recipe_encode_s',
                'request_prepare_s', 'cache_lookup_s', 'paired_restore_s',
                'prompt_checkpoint_capture_s', 'suffix_append_s', 'p5_handoff_s',
                'first_decode_call_s', 'first_native_decode_s', 'total_ttft_s',
                'prompt_replay', 'full_cache_repack', 'canonical_frontier',
                'historical_patch_discount', 'historical_planned_patches', 'fresh_encoder_patches',
                'selected_pair_frontier', 'selected_pair_bytes', 'retained_pair_bytes',
                'projected_pair_bytes', 'requested_pair_frontier', 'retained_pair_roots',
                'paired_free_blocks', 'paired_evictable_blocks',
                'legacy_pair_bytes_exceeded', 'legacy_root_bytes_exceeded')


class Projection:
    def __init__(self):
        self._lock = Lock()
        self.state = 'STARTING'
        self.requests = {}
        self.last = None
        self.cache = None
        self.endpoint = None
        self.failures = deque(maxlen=128)
        self.health = None  # read-only backend readiness/fatal projection

    def lifecycle(self, state, endpoint=None):
        try:
            with self._lock:
                changed = self.state != state
                self.state = state
                if endpoint is not None:
                    self.endpoint = endpoint
            if changed:
                logger.log(logging.ERROR if state == 'FAILED' else logging.INFO, 'lifecycle %s', state)
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
            logger.info('request received id=%s', rid)
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

    def correlate(self, rid, scheduler_id):
        try:
            with self._lock:
                r = self.requests.get(rid)
                if r is not None:
                    r['scheduler_request_id'] = scheduler_id
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
                    if trace:
                        for key in ('native_started', 'canonical_settled', 'cache_published'):
                            if key in trace:
                                r[key] = bool(trace[key])
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
            if r is not None:
                logger.log(logging.ERROR if finish_reason == 'error' else logging.INFO,
                           'request finished id=%s reason=%s', rid, finish_reason)
        except Exception:
            pass

    def failure(self, rid, exc, *, stage='unexpected', scheduler_id=None,
                native_started=None, settled=None, published=None, fatal=False,
                secondary=False):
        """Copy only bounded scalars, never exception text/objects/locals."""
        try:
            with self._lock:
                if not secondary and any(f['request_id'] == rid and not f['secondary']
                                         for f in self.failures):
                    return  # outer HTTP/delivery wrappers do not replace a primary
                r = self.requests.get(rid) or (self.last if self.last and self.last['request_id'] == rid else {})
                scheduler_id = scheduler_id or r.get('scheduler_request_id')
                native_started = r.get('native_started', False) if native_started is None else native_started
                settled = r.get('canonical_settled', False) if settled is None else settled
                published = r.get('cache_published', False) if published is None else published
            source = None
            reason = getattr(exc, 'code', None)
            # Wrapping at a first-party boundary must not hide the operation.
            seen = set()
            while exc.__cause__ is not None and id(exc) not in seen:
                seen.add(id(exc))
                exc = exc.__cause__
            tb = exc.__traceback__
            while tb is not None:
                frame = tb.tb_frame
                module = frame.f_globals.get('__name__', '')
                function = frame.f_code.co_name
                if (module.startswith('ds41f_mlx.') and
                        re.fullmatch(r'[\w.]+', module) and
                        re.fullmatch(r'[\w<>]+', function)):
                    source = dict(module=module, function=function[:80], line=tb.tb_lineno)
                tb = tb.tb_next
            category = reason if reason in ('context_capacity_exhausted', 'body_limit',
                'body_timeout', 'preparation_busy') else 'unexpected_exception'
            kind = type(exc).__name__
            kind = kind[:80] if re.fullmatch(r'\w+', kind) else 'Exception'
            item = dict(request_id=rid, scheduler_request_id=scheduler_id,
                stage=stage if stage in FAILURE_STAGES else 'unexpected',
                exception_type=kind, category=category, source=source,
                native_started=bool(native_started), canonical_settled=bool(settled),
                cache_published=bool(published), runtime_fatal=bool(fatal),
                secondary=bool(secondary), timestamp=time())
            with self._lock:
                self.failures.append(item)
                r = self.requests.get(rid)
                if r is not None and not secondary:
                    r['failure'] = item
            logger.error('request failure %s', json.dumps(item, separators=(',', ':')))
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
        health = None
        try:
            health = self.health() if self.health is not None else None
        except Exception:
            pass
        with self._lock:
            state = self.state
            if health is not None and state not in ('STOPPING', 'STOPPED'):
                ready, fatal = health
                if fatal:
                    state = 'FAILED'
                elif state == 'READY' and not ready:
                    state = 'LOADING'
            requests = list(self.requests.values())
            queued = [r for r in requests if r['current_phase'] == 'QUEUED']
            active = [r for r in requests if r['current_phase'] != 'QUEUED']
            current = next((r for r in active if r['current_phase'] not in ('RECEIVED', 'ENCODING')), None)
            current = current or (active[0] if active else queued[0] if queued else None)
            return deepcopy(dict(schema='ds41f.live.v1', sampled_at=time(), sampled_mono=monotonic(),
                state=state, endpoint=self.endpoint, current_request=current,
                failures=list(self.failures),
                requests=requests, queued_requests=len(queued), active_requests=len(active),
                last_request=self.last, cache=self.cache))


FAILURE_STAGES = frozenset(('preparation', 'admission', 'cache_lookup', 'paired_restore',
    'historical_image', 'vision_preparation', 'native_start', 'native_execution',
    'settlement', 'publication', 'delivery', 'cancellation', 'cleanup', 'unexpected'))


def failure(backend, trace, exc, **kwargs):
    try:
        projection = getattr(backend, 'telemetry', None)
        if projection is not None:
            projection.failure(trace.get('diagnostic_id'), exc,
                stage=trace.get('failure_stage', 'unexpected'), **kwargs)
    except Exception:
        pass


def phase(backend, trace, name):
    stage = {'CACHE_LOOKUP': 'cache_lookup', 'RESTORING': 'paired_restore',
             'SETTLING': 'settlement', 'DECODING': 'native_execution',
             'SUFFIX_APPEND': 'native_start', 'P5_HANDOFF': 'native_start',
             'CHECKPOINT_CAPTURE': 'publication'}.get(name)
    if stage:
        trace['failure_stage'] = stage
    projection = getattr(backend, 'telemetry', None)
    if projection is not None:
        projection.phase(trace.get('diagnostic_id'), name)


def metrics(backend, trace):
    projection = getattr(backend, 'telemetry', None)
    if projection is not None:
        projection.metrics(trace.get('diagnostic_id'), trace)
