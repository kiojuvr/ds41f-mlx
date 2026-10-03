"""Synthetic closed-record retention probe: actual create implementation, no model.
Closed flags substitute completed DELETE; real retirement/resource evidence is soak.json.
This records a blocker, not a passing bounded-retention claim.
"""
import asyncio
import json
from pathlib import Path
from ds41f_mlx.serving.internal_mtp import InternalMTPQualificationBackend

async def main():
    b=object.__new__(InternalMTPQualificationBackend);b.sessions={}
    samples=[]
    for i in range(256):
        rec=await b.create_stateful_session(session_id=f'retention-{i}')
        rec.closed=True  # synthetic post-DELETE boundary; no native caches allocated
        if (i+1)%32==0:samples.append(dict(cycles=i+1,retained_sessions=len(b.sessions),
            serialized_record_bytes=sum(len(json.dumps(s.to_json()).encode()) for s in b.sessions.values())))
    try:await b.create_stateful_session(session_id='retention-0')
    except ValueError:pass
    else:raise AssertionError('retired session ID reused')
    assert len(b.sessions)==256
    Path('artifacts/m38/retention-probe.json').write_text(json.dumps(dict(
        schema='ds41f.m38.retention-probe.v1',scope='synthetic closed flags; actual create method; no checkpoint execution',
        finding='NO_LIFETIME_SESSION_CAP',samples=samples,old_ids_denied=True,
        explanation='max_live_sessions excludes closed records; sessions dict retains every closed identity and its diagnostics indefinitely'),indent=2)+'\n')

if __name__=='__main__':asyncio.run(main())
