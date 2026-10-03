"""Checkpoint integration of server-issued lifetimes, retained fences and DELETE.
Not another recovery soak. Detailed evidence is copied outside backend ownership.
"""
import asyncio
import copy
import json
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import run_m35_http_qualification as q  # pinned native/replay instrumentation
from ds41f_mlx.internal_local_client import InternalLocalClient, OwnedStream, RequestIdentity
from ds41f_mlx.web_client import RuntimeClient, RuntimeHTTPError
from tools.run_m39_lifetimes import structures
import uvicorn

OUT=ROOT/'artifacts/m39/integration.json'
result=dict(schema='ds41f.m39.integration.v1',status='RUNNING',phases=[],resources=[],admission=[])
result['sources']={p:q.sha(ROOT/p) for p in ('ds41f_mlx/serving/internal_mtp.py',
    'ds41f_mlx/serving/server.py','ds41f_mlx/internal_local_client.py',
    'tools/run_m39_integration.py','tools/run_m39_lifetimes.py')}
def save(): OUT.write_text(json.dumps(result,indent=2)+'\n')

async def main():
    sock=socket.socket();sock.bind(('127.0.0.1',0));q.port=sock.getsockname()[1]
    app=q.create_app(backend=q.backend,recipe_path=q.backend.recipe_path)
    server=uvicorn.Server(uvicorn.Config(app,log_level='warning',lifespan='off',http='h11'))
    task=asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:await asyncio.sleep(.01)
    runtime=RuntimeClient(f'http://127.0.0.1:{q.port}');client=InternalLocalClient(runtime)
    async def call(fn,*a,**kw):return await asyncio.to_thread(fn,*a,**kw)
    async def resolve():
        for _ in range(10000):
            out=await call(client.reconcile)
            if isinstance(out,OwnedStream): await call(lambda:list(out))
            elif out.get('outcome_state')!='active':return out
            await asyncio.sleep(.01)
        raise AssertionError('settlement timeout')
    options=dict(model='deepseek-v4.1-flash',temperature=0,max_tokens=64,reasoning_effort='none',stream=True)
    oldest=None
    traces=[]
    async def turn(text):
        value=await call(client.submit,[dict(role='user',content=text)],options=options)
        if isinstance(value,OwnedStream):await call(lambda:list(value))
        identity=client.identity
        out=await resolve();assert out['outcome_state']=='recoverable',out
        before=len(q.backend.session_traces)
        for _ in range(2):assert await resolve()==out
        assert len(q.backend.session_traces)==before
        trace=copy.deepcopy(q.backend.progress);traces.append(trace)
        assert trace['prompt_replay']==trace['full_cache_repack']==0
        assert set(trace['target_offsets']+trace['dspark_offsets'])=={trace['canonical_frontier']}
        return dict(identity=identity.to_json(),outcome=out,trace=trace)
    def resource_row():
        import mlx.core as mx
        import subprocess
        return dict(**structures(q.backend),mlx_active_bytes=mx.get_active_memory(),
                    mlx_cache_bytes=mx.get_cache_memory(),
                    process_rss_bytes=int(subprocess.check_output(['ps','-o','rss=','-p',str(__import__('os').getpid())]))*1024)
    try:
        for phase in range(3):
            await call(client.create)
            sid=client.session_id;rec=q.backend.sessions[sid]
            if phase==0:
                before=(q.backend._issued,copy.deepcopy(rec.to_json()),len(q.backend.session_traces),dict(q.counts))
                for name,fn,args in [
                    ('live_capacity',runtime.create_session,()),
                    ('body_capacity',runtime.internal_fenced_request,(sid,b'x'*1048577,1)),
                    ('sequence_capacity',runtime.internal_fenced_request,(sid,json.dumps(dict(options,messages=[dict(role='user',content='Hello')])).encode(),2**64))]:
                    try:await call(fn,*args)
                    except RuntimeHTTPError as exc:result['admission'].append(dict(name=name,status=exc.status))
                    else:raise AssertionError('capacity admitted')
                    assert before==(q.backend._issued,rec.to_json(),len(q.backend.session_traces),dict(q.counts))
            rows=[await turn('Say hello briefly. Do not use tools.')]
            oldest=oldest or client.identity
            rows.append(await turn('Say goodbye briefly. Do not use tools.'))
            expired=await call(client.observe_identity,RequestIdentity(sid,1,rows[0]['identity']['body_utf8'].encode()))
            assert expired['outcome_state']=='expired'
            await call(client.retire)
            assert rec.closed and all(getattr(rec,k) is None for k in ('owner','processor','guard','cache','rings','last_turn','certificate','fence','reconstruction_body','reconstruction_tokenizer'))
            assert not rec.canonical and not q.backend.sessions and not q.backend.session_traces
            for fn,args in [(runtime.get_session,(sid,)),(runtime.close_session,(sid,)),
                            (runtime.internal_fenced_request,(sid,rows[0]['identity']['body_utf8'].encode(),1)),
                            (runtime.create_session,(sid,))]:
                try:await call(fn,*args)
                except RuntimeHTTPError:pass
                else:raise AssertionError('retired identity admitted')
            result['phases'].append(dict(session_id=sid,turns=rows,delete_payloads_removed=True))
            result['resources'].append(resource_row());save()
        # Empty real HTTP lifetimes drive diagnostic eviction without model work.
        for _ in range(48):
            await call(client.create);await call(client.retire)
        assert len(q.backend.retired_diagnostics)==16
        assert oldest.session_id not in [d['id'] for d in q.backend.retired_diagnostics]
        assert q.backend.identity_state(oldest.session_id)=='retired'
        for fn,args in [(runtime.get_session,(oldest.session_id,)),(runtime.close_session,(oldest.session_id,)),
                        (runtime.internal_fenced_request,(oldest.session_id,oldest.body,oldest.sequence))]:
            try:await call(fn,*args)
            except RuntimeHTTPError as exc:assert exc.status==404
            else:raise AssertionError('evicted old authority resumed')
        await call(client.create)
        rows=[await turn('Reply with one short greeting. Do not use tools.')]
        await call(client.retire)
        result['phases'].append(dict(session_id=rows[0]['identity']['session_id'],turns=rows,post_eviction=True))
        result['resources'].append(resource_row())
        assert q.counts['replay']==q.counts['repack']==0 and q.counts['fresh_target_allocations']==4
        result.update(status='PASS',counts=q.counts,empty_lifetimes=48,
                      evicted_session_and_request_rejected=True,trace_count=len(traces))
    except BaseException as exc:result.update(status='FAIL',error=repr(exc));raise
    finally:
        save();server.should_exit=True;await task
        if not any(s.busy for s in q.backend.sessions.values()):q.backend.close()

if __name__=='__main__':asyncio.run(main())
