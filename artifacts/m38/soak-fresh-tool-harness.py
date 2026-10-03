"""M38 bounded living-client checkpoint soak; real loopback HTTP, explicit injection."""
import asyncio
import copy
import gc
import hashlib
import json
from pathlib import Path
import resource
import socket
import subprocess
import time
import uvicorn
from tools import run_m35_http_qualification as q
from tools.run_m11_tool_boundary_qualification import tool_def
from ds41f_mlx.internal_local_client import InternalLocalClient, OwnedStream, ClientStateError
from ds41f_mlx.web_client import RuntimeClient, RuntimeHTTPError

OUT=q.ROOT/'artifacts/m38/soak.json'
result=dict(schema='ds41f.m38.soak.v1',status='RUNNING',turns=[],phases=[],effects=[],actions=[])
def save():
    OUT.write_text(json.dumps(result,indent=2)+'\n')

async def main():
    sock=socket.socket(); sock.bind(('127.0.0.1',0)); q.port=sock.getsockname()[1]
    app=q.create_app(backend=q.backend,recipe_path=q.backend.recipe_path)
    fault={'method':None}
    async def wrapped(scope,receive,send):
        async def faulty_send(message):
            if scope['type']=='http' and scope['method']==fault['method'] and message['type']=='http.response.start':
                fault['method']=None
                raise OSError('M38 synthetic response-header loss AFTER real lifecycle mutation')
            await send(message)
        await app(scope,receive,faulty_send)
    server=uvicorn.Server(uvicorn.Config(wrapped,log_level='warning',lifespan='off',http='h11'))
    task=asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started: await asyncio.sleep(.01)
    runtime=RuntimeClient(f'http://127.0.0.1:{q.port}')
    client=InternalLocalClient(runtime)
    async def call(fn,*a,**kw):return await asyncio.to_thread(fn,*a,**kw)
    async def resolve(c=client):
        t=time.perf_counter()
        for _ in range(4000):
            out=await call(c.reconcile)
            if isinstance(out,OwnedStream):await call(lambda:list(out))
            elif out.get('outcome_state')!='active':return out,time.perf_counter()-t
            await asyncio.sleep(.01)
        raise AssertionError('bounded settlement deadline')
    def snapshot():
        import mlx.core as mx
        sessions=q.backend.sessions
        return dict(client_bytes=client.retained_state_bytes(),ledger_entries=len(client.ledger),
            ledger_json_bytes=len(json.dumps([dict(key=list(k),**v) for k,v in client.ledger.items()]).encode()),
            timings=len(client.timings),request_bytes=len(client.identity.body) if client.identity else 0,
            outcome_bytes=len(json.dumps(client.outcome).encode()),server_sessions=len(sessions),
            server_trace_length=len(q.backend.session_traces),server_trace_limit=q.backend.session_traces.maxlen,
            closed_sessions=sum(s.closed for s in sessions.values()),
            server_session_json_bytes=sum(len(json.dumps(s.to_json()).encode()) for s in sessions.values()),
            retained_canonical_ids=sum(len(s.canonical) for s in sessions.values()),
            mlx_active_bytes=mx.get_active_memory(),mlx_cache_bytes=mx.get_cache_memory(),
            process_rss_bytes=int(subprocess.check_output(['ps','-o','rss=','-p',str(__import__('os').getpid())]))*1024,
            maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    base=dict(model='deepseek-v4.1-flash',tools=[tool_def()],tool_choice='auto',reasoning_effort='none',temperature=0,max_tokens=128,stream=True)
    def effect(call):
        result['effects'].append(dict(identity=client.identity.to_json(),call=copy.deepcopy(call)))
        return q.tool_stub(call['function']['name'],call['function']['arguments'])
    async def turn(phase,name,text=None,mode='normal',required=False,continuation=False):
        opts=copy.deepcopy(base); opts['stream']=mode!='json'
        if required:opts['tool_choice']={'type':'function','function':{'name':'lookup_weather'}};opts['max_tokens']=96
        q.backend.delivery_delay_s=.08 if mode=='partial' else .04 if mode in ('drop','zero','iterator-drop') else .2 if mode=='terminal' else 0
        t=time.perf_counter()
        stream=await call(client.submit_tool_results,options=opts) if continuation else await call(client.submit,[dict(role='user',content=text)],options=opts)
        identity=client.identity; before=client.messages; observed=[]
        assert client.next_sequence==identity.sequence and identity.session_id==client.session_id
        if isinstance(stream,OwnedStream):
            if mode=='zero':await call(stream.close)
            elif mode=='terminal':
                for _ in range(4000):
                    if q.backend.progress.get('certificate'):break
                    await asyncio.sleep(.01)
                else:raise AssertionError('terminal deadline')
                await call(stream.close)
            elif mode=='iterator-drop':
                def drop(s):
                    it=iter(s);observed.append(next(it));it.close()
                await call(drop,stream)
            elif mode in ('drop','partial'):
                def cut():
                    it=iter(stream)
                    for e in it:
                        observed.append(e);d=e['choices'][0].get('delta',{})
                        if (mode=='drop' and d.get('content')) or (mode=='partial' and any(c.get('function',{}).get('arguments') for c in d.get('tool_calls',[]))):break
                    stream.close();it.close()
                await call(cut)
            else:
                def consume():
                    for e in stream:
                        observed.append(e)
                        if mode=='slow' and len(observed)%4==0:time.sleep(.04)
                await call(consume)
        else:observed=[dict(ordinary_response=stream)]
        assert client.messages==before  # no SSE/ordinary JSON enters history
        out,wait=await resolve()
        transcript=client.messages
        expected=json.loads(identity.body)['messages']
        if out['outcome_state']=='recoverable':
            expected=expected+[out['response']['choices'][0]['message']]
            assert client.next_sequence==identity.sequence+1
        else:assert client.next_sequence==identity.sequence
        assert transcript==expected
        for _ in range(2):
            again,_=await resolve();assert again==out and client.messages==transcript
        trace=copy.deepcopy(q.backend.progress)
        assert trace['session_id']==identity.session_id
        assert trace['prompt_replay']==trace['full_cache_repack']==0
        assert set(trace['target_offsets']+trace['dspark_offsets'])=={trace['canonical_frontier']}
        assert trace['queue_empty'] and trace['prediction_retired']
        assert all(trace['quiescence']['counters'][k]==0 for k in ['history_replay','full_cache_repack','new_verify_cycles','new_proposals'])
        entry=dict(phase=phase,name=name,mode=mode,identity=identity.to_json(),observed=observed,outcome=out,transcript=transcript,
            state=client.state,wall_s=time.perf_counter()-t,reconcile_wait_s=wait,trace=trace,resources=snapshot())
        result['turns'].append(entry);save()
        if client.state=='tool_pending' and mode!='partial':
            n=len(result['effects']); stored=await call(client.execute_tools,effect)
            for _ in range(2):
                assert (await resolve())[0]==out
                assert await call(client.execute_tools,effect)==stored
            assert len(result['effects'])==n+len(stored)
        return identity,out
    try:
        for phase in range(3):
            await call(client.create,f'm38-agent-{phase}')
            assert client.next_sequence==1 and client.identity is None
            first=None
            for round in range(4):
                ident,_=await turn(phase,f'{round}-plan','Do not finish any previous lookup. Without tools, explain one practical step for planning a weather-aware day in two sentences.')
                if first is None:first=ident
                await turn(phase,f'{round}-weather',f'Use lookup_weather for {"Paris" if round%2==0 else "Berlin"}.',required=True,mode='terminal' if round==2 else 'normal')
                assert client.state=='tool_completed'
                await turn(phase,f'{round}-stored-result',continuation=True)
                await turn(phase,f'{round}-note','Without tools, summarize the day plan briefly.',mode='json')
                await turn(phase,f'{round}-interrupted','Without tools, list three short travel tips.',mode=['drop','zero','iterator-drop','drop'][round])
                await turn(phase,f'{round}-review','Without tools, explain why a flexible schedule helps in three sentences.',mode='slow' if round==1 else 'normal')
            before=q.counts.copy()
            assert (await call(client.observe_identity,first))['outcome_state']=='expired'
            try:await call(runtime.internal_fenced_request,first.session_id,first.body,first.sequence)
            except RuntimeHTTPError as exc:assert exc.status==400
            else:raise AssertionError('expired regenerated')
            assert q.counts==before
            if phase<2:
                old,out=await turn(phase,'partial-tool','Use lookup_weather for Paris again.',required=True,mode='partial')
                assert out['outcome_state']=='unrecoverable'
                assert client.messages==json.loads(old.body)['messages']
            else:old=client.identity
            preserved=client.messages; closed=await call(client.retire)
            assert closed['state']=='closed' and client.identity is None
            try:await call(client.observe_identity,old)
            except ClientStateError:pass
            else:raise AssertionError('retired identity resumed')
            for fn,args in [(runtime.create_session,(old.session_id,)),(runtime.internal_fenced_request,(old.session_id,old.body,old.sequence))]:
                try:await call(fn,*args)
                except RuntimeHTTPError:pass
                else:raise AssertionError('retired server identity resumed')
            assert client.messages==preserved
            assert all(s.owner is None and s.cache is None and s.rings is None and not s.busy for s in q.backend.sessions.values())
            result['phases'].append(dict(phase=phase,resources=snapshot(),ledger=client.ledger and [dict(key=list(k),**v) for k,v in client.ledger.items()],counts=q.counts.copy()))
            save()
        # Real HTTP mutation, targeted synthetic ASGI header-send loss; no outcome guessing.
        await call(client.create,'m38-delete-loss');fault['method']='DELETE'
        try:await call(client.retire)
        except Exception as exc: result['actions'].append(dict(fault='DELETE response loss',error=repr(exc)))
        else:raise AssertionError('DELETE loss not injected')
        assert q.backend.sessions['m38-delete-loss'].closed and client.lifecycle_uncertain['action']=='delete'
        for fn in [client.retire,lambda:client.create('forbidden')]:
            try:await call(fn)
            except ClientStateError:pass
            else:raise AssertionError('uncertain DELETE bypass')
        fresh=InternalLocalClient(runtime);fault['method']='POST'
        try:await call(fresh.create,'m38-create-loss')
        except Exception as exc:result['actions'].append(dict(fault='create response loss',error=repr(exc)))
        else:raise AssertionError('create loss not injected')
        assert not q.backend.sessions['m38-create-loss'].closed and fresh.lifecycle_uncertain['action']=='create'
        for fn in [fresh.retire,lambda:fresh.create('forbidden')]:
            try:await call(fn)
            except ClientStateError:pass
            else:raise AssertionError('uncertain create bypass')
        # Explicit external test-controller cleanup, NOT client-authorized reconciliation.
        await call(runtime.close_session,'m38-create-loss')
        assert q.counts['replay']==q.counts['repack']==0
        result.update(status='PASS_BOUNDED_WORKLOAD',counts=q.counts,final_resources=snapshot(),timings=list(client.timings),
            ledger=[dict(key=list(k),**v) for k,v in client.ledger.items()],
            sources={p:q.sha(q.ROOT/p) for p in ['tools/run_m38_soak.py','ds41f_mlx/internal_local_client.py','ds41f_mlx/serving/internal_mtp.py','ds41f_mlx/serving/server.py']})
    except BaseException as exc:
        result.update(status='FAIL',error=repr(exc),client_state=client.state);raise
    finally:
        save();server.should_exit=True;await task
        if not any(s.busy for s in q.backend.sessions.values()):q.backend.close()

if __name__=='__main__':asyncio.run(main())
