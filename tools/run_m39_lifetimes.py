"""M39 actual issuance/DELETE stress; no checkpoint authority allocated.
Only native retirement is replaced with an assertion that no native owner exists.
"""
import asyncio
import gc
import hashlib
import json
from pathlib import Path
import resource
import statistics
import sys
import time
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ds41f_mlx.serving.internal_mtp import InternalMTPQualificationBackend


def structures(b):
    return dict(live_records=len(b.sessions), live_dictionary_storage_bytes=sys.getsizeof(b.sessions),
                native_owners=sum(s.owner is not None for s in b.sessions.values()),
                identity_scalars=2, namespace_bytes=len(b._namespace), counter_bits=b._issued.bit_length(),
                counter_max_bits=128,
                identity_storage_bytes=sys.getsizeof(b._namespace)+sys.getsizeof(b._issued),
                retired_authority_records=sum(s.closed for s in b.sessions.values()),
                diagnostic_records=len(b.retired_diagnostics),
                diagnostic_payload_bytes=len(json.dumps(list(b.retired_diagnostics)).encode()),
                closed_outcome_payload_bytes=sum(len(json.dumps(s.last_turn).encode())
                    for s in b.sessions.values() if s.closed and s.last_turn is not None),
                diagnostic_deque_storage_bytes=sys.getsizeof(b.retired_diagnostics),
                session_trace_slots=len(b.session_traces), session_trace_limit=b.session_traces.maxlen,
                trace_slots=len(b.traces), trace_limit=b.traces.maxlen, process_maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


async def stress(n=20000):
    b = InternalMTPQualificationBackend()
    def retire(rec):
        assert rec.owner is rec.cache is rec.rings is rec.processor is None
    b._retire = retire
    samples = []
    cohorts = {k: {a: [] for a in ('create', 'delete', 'stale_reject', 'get', 'outcome_retry', 'capacity_deny')}
               for k in ('early', 'mid', 'late')}
    oldest = None
    def timed(rows, name, fn):
        t = time.perf_counter_ns(); value = fn()
        if rows is not None: rows[name].append(time.perf_counter_ns()-t)
        return value
    def rejected_get(sid):
        try: b.get_stateful_session(sid)
        except KeyError: return
        raise AssertionError('stale identity became live')
    try:
        for i in range(n):
            cohort = 'early' if i < 200 else 'mid' if n//2 <= i < n//2+200 else 'late' if i >= n-200 else None
            rows = cohorts.get(cohort)
            t = time.perf_counter_ns(); rec = await b.create_stateful_session()
            if rows is not None: rows['create'].append(time.perf_counter_ns()-t)
            if oldest is None: oldest = rec.session_id
            sid = rec.session_id
            ref = weakref.ref(rec)
            # Fixed, cheap contents expose payload disposal independently of RSS.
            rec.canonical = list(range(8192))
            rec.reconstruction_body = {'diagnostic': 'x'*65536}
            rec.certificate = {'representable': True, 'payload': 'x'*65536}
            rec.last_turn = {'response': {'choices': []}, 'payload': 'x'*65536}
            rec.request_count = 1
            from ds41f_mlx.serving.request_fence import reserve, finish, observe_retry
            reserve(rec, 1, b'{}'); finish(rec)
            b.session_traces.append(rec.last_turn); b.progress = rec.last_turn
            timed(rows, 'get', lambda: b.get_stateful_session(sid).to_json())
            timed(rows, 'outcome_retry', lambda: observe_retry(rec, 1, b'{}'))
            issued = b._issued
            t = time.perf_counter_ns()
            try: await b.create_stateful_session()
            except RuntimeError: pass
            else: raise AssertionError('capacity admitted')
            if rows is not None: rows['capacity_deny'].append(time.perf_counter_ns()-t)
            assert b._issued == issued and len(b.sessions) == 1
            t = time.perf_counter_ns(); closed = await b.close_stateful_session(sid)
            if rows is not None: rows['delete'].append(time.perf_counter_ns()-t)
            assert closed['state'] == 'closed' and b.identity_state(sid) == 'retired'
            assert not rec.canonical and rec.last_turn is rec.guard is rec.fence is rec.certificate is None
            del rec
            # Executor result delivery callbacks can retain call arguments until
            # the next event-loop turn; they are not backend lifecycle storage.
            await asyncio.sleep(0)
            assert ref() is None, 'server retained closed record'
            timed(rows, 'stale_reject', lambda: rejected_get(oldest))
            assert len(b.retired_diagnostics) <= 16 and not b.sessions
            if i+1 in (1,16,32,200,n//2,n):
                gc.collect(); samples.append(dict(cycles=i+1, **structures(b)))
        assert oldest not in {r['id'] for r in b.retired_diagnostics}
        assert b.identity_state(oldest) == 'retired'
        for sid in (oldest, f'mtp_{b._namespace}_{0:032x}', f'mtp_{b._namespace}_{n+1:032x}',
                    'mtp_'+'0'*32+'_'+f'{1:032x}', 'arbitrary'):
            before = (b._issued, list(b.retired_diagnostics))
            rejected_get(sid)
            try: await b.close_stateful_session(sid)
            except KeyError: pass
            else: raise AssertionError('stale DELETE mutated')
            try: await b.create_stateful_session(session_id=sid)
            except ValueError: pass
            else: raise AssertionError('client-selected ID admitted')
            assert before == (b._issued, list(b.retired_diagnostics))
        # Fixed-width namespace exhaustion has no wrap and no partial creation.
        b._issued = b.MAX_LIFETIMES
        try: await b.create_stateful_session()
        except RuntimeError: pass
        else: raise AssertionError('counter wrapped')
        assert not b.sessions and b._issued == b.MAX_LIFETIMES
        perf = {k: {a: dict(samples=len(v), median_ns=statistics.median(v), p95_ns=sorted(v)[int(.95*len(v))])
                    for a,v in rows.items()} for k,rows in cohorts.items()}
        import mlx.core as mx
        return dict(schema='ds41f.m39.lifetimes.v1', status='PASS', lifetimes=n, diagnostic_budget=16,
                    oldest_identity=oldest, diagnostic_eviction_preserves_retirement=True,
                    stale_operations_nonmutating=True, exhaustion_fail_closed=True,
                    samples=samples, performance=perf,
                    mlx_active_bytes=mx.get_active_memory(), mlx_cache_bytes=mx.get_cache_memory(),
                    scope='actual backend issuance/DELETE; synthetic no-native retirement; bounded synthetic payloads')
    finally: b.close()

if __name__ == '__main__':
    out = ROOT/'artifacts/m39/lifetimes.json'; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asyncio.run(stress()), indent=2)+'\n')
