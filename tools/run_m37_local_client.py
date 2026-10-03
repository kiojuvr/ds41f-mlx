"""Real checkpoint coherent workload through the reusable internal local client."""
import asyncio
import copy
import json
from pathlib import Path
import socket
import time
import uvicorn
from tools import run_m35_http_qualification as q
from tools.run_m11_tool_boundary_qualification import tool_def
from ds41f_mlx.web_client import RuntimeClient, RuntimeHTTPError
from ds41f_mlx.internal_local_client import InternalLocalClient, OwnedStream, ClientStateError

SOURCES={p:q.sha(q.ROOT/p) for p in ['ds41f_mlx/internal_local_client.py','ds41f_mlx/web_client.py','ds41f_mlx/serving/internal_mtp.py','ds41f_mlx/serving/server.py','tools/run_m37_local_client.py']}
OUT = q.ROOT/'artifacts/m37/workflow.json'
result = dict(schema='ds41f.m37.workflow.v1', status='RUNNING', turns=[], effects=[], actions=[])
def save():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2)+'\n')

async def main():
    sock=socket.socket(); sock.bind(('127.0.0.1',0)); q.port=sock.getsockname()[1]
    app=q.create_app(backend=q.backend, recipe_path=q.backend.recipe_path)
    server=uvicorn.Server(uvicorn.Config(app,log_level='warning',lifespan='off',http='h11'))
    task=asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started: await asyncio.sleep(.01)
    runtime=RuntimeClient(f'http://127.0.0.1:{q.port}')
    client=InternalLocalClient(runtime)
    async def call(fn,*args,**kw): return await asyncio.to_thread(fn,*args,**kw)
    async def resolve():
        t0=time.perf_counter()
        for _ in range(4000):
            out=await call(client.reconcile)
            if isinstance(out,OwnedStream):
                await call(lambda: list(out))
            elif out.get('outcome_state')!='active':
                return out,time.perf_counter()-t0
            await asyncio.sleep(.01)
        raise AssertionError('settlement timeout')
    base=dict(model='deepseek-v4.1-flash',tools=[tool_def()],tool_choice='auto',reasoning_effort='none',temperature=0,max_tokens=64,stream=True)
    def effect(c):
        result['effects'].append(dict(identity=client.identity.to_json(),call=copy.deepcopy(c)))
        return q.tool_stub(c['function']['name'],c['function']['arguments'])
    async def turn(name,text=None,mode='normal',required=False,tool_result=False,tools_enabled=True):
        options=copy.deepcopy(base)
        if not tools_enabled:
            options.pop('tools');options.pop('tool_choice');options['stream']=False
        if required: options['tool_choice']={'type':'function','function':{'name':'lookup_weather'}};options['max_tokens']=96
        q.backend.delivery_delay_s=.08 if mode=='partial' else 1.0 if mode=='terminal-drop' else .04 if mode=='drop-text' else 0
        t0=time.perf_counter()
        stream=await call(client.submit_tool_results,options=options) if tool_result else await call(client.submit,[dict(role='user',content=text)],options=options)
        identity=client.identity; observed=[]
        if not isinstance(stream,OwnedStream):
            observed=[dict(ordinary_response=stream)]
        elif mode=='terminal-drop':
            while not q.backend.progress.get('certificate'):await asyncio.sleep(.01)
            assert q.backend.progress['certificate']['representable']
            await call(stream.close)
        elif mode=='drop-text':
            # Drop the actual abstraction (no raw Python-controller HTTP socket).
            def cut():
                it=iter(stream)
                for e in it:
                    observed.append(e)
                    if e['choices'][0].get('delta',{}).get('content'):break
                stream.close();it.close()
            await call(cut)
        elif mode=='partial':
            def cut_tool():
                it=iter(stream)
                for e in it:
                    observed.append(e)
                    if any(c.get('function',{}).get('arguments') for c in e['choices'][0].get('delta',{}).get('tool_calls',[])):break
                stream.close();it.close()
            await call(cut_tool)
        else: observed=await call(lambda:list(stream))
        out,wait=await resolve()
        entry=dict(name=name,identity=identity.to_json(),observed=observed,outcome=out,
                   transcript=client.messages,state=client.state,wall_s=time.perf_counter()-t0,
                   recovery_wait_and_lookup_s=wait,retained_state_bytes=client.retained_state_bytes())
        result['turns'].append(entry);save()
        return identity,out
    try:
        await call(client.create,'m37-agent')
        first,_=await turn('normal-streamed-text','Say hello briefly. Do not use tools.')
        _,out=await turn('normal-tool','Use lookup_weather for Paris.',required=True)
        assert out['certificate']['executable_tools']
        stored=await call(client.execute_tools,effect)
        for _ in range(3):
            again,_=await resolve();assert again==out
            assert await call(client.execute_tools,effect)==stored
        assert len(result['effects'])==1
        result['actions'].append(dict(name='normal-tool-repeated-observation',observations=4,results=stored,ledger=[dict(key=list(k),**v) for k,v in client.ledger.items()]))
        await turn('normal-tool-result',tool_result=True)
        _,out=await turn('ambiguous-stream-text','Output exactly: café 🙂 漢字. Nothing else.',mode='drop-text')
        assert out['outcome_state']=='recoverable'
        # Complete tool, but transport loses ALL events in controlled terminal window.
        _,out=await turn('ambiguous-complete-tool','Use lookup_weather for Berlin.',mode='terminal-drop',required=True)
        stored=await call(client.execute_tools,effect)
        for _ in range(3):
            again,_=await resolve();assert again==out
            assert await call(client.execute_tools,effect)==stored
        assert len(result['effects'])==2
        result['actions'].append(dict(name='ambiguous-tool-repeated-observation',observations=4,results=stored))
        await turn('ambiguous-tool-result',tool_result=True)
        before=len(q.backend.session_traces)
        expired=await call(client.observe_identity,first)
        assert expired['outcome_state']=='expired' and len(q.backend.session_traces)==before
        try:await call(runtime.internal_fenced_request,first.session_id,first.body,first.sequence)
        except RuntimeHTTPError as exc:assert exc.status==400
        else:raise AssertionError('expired work regenerated')
        assert len(q.backend.session_traces)==before
        status,_=await asyncio.to_thread(q.http_request,'POST',f'/v1/sessions/{first.session_id}/chat/completions',json.loads(first.body))
        assert status==400  # no-header mixing cannot bypass fence
        result['actions'].append(dict(name='expired-outcome',identity=first.to_json(),result=expired,no_generation=True))
        old,out=await turn('incomplete-tool','Use lookup_weather for Paris again.',mode='partial',required=True)
        assert out['outcome_state']=='unrecoverable' and out['certificate']['representable'] is False
        preserved=client.messages
        assert preserved==json.loads(old.body)['messages']
        # Public operation denial must not cause an effect. Use the actual client;
        # fail-closed stop remains explicitly deletable.
        try:await call(client.execute_tools,effect)
        except ClientStateError:pass
        else:raise AssertionError('incomplete tool executable')
        assert len(result['effects'])==2
        closed=await call(client.retire);assert closed['state']=='closed'
        try:await call(client.observe_identity,old)
        except ClientStateError:pass
        else:raise AssertionError('retired identity resumed')
        # Reusing old session ID and direct old POST must also fail on the server.
        try:await call(runtime.create_session,old.session_id)
        except RuntimeHTTPError:pass
        else:raise AssertionError('old session ID reused')
        try:await call(runtime.internal_fenced_request,old.session_id,old.body,old.sequence)
        except RuntimeHTTPError:pass
        else:raise AssertionError('old request resumed')
        await call(client.create,'m37-fresh')
        assert client.messages==preserved
        result['actions'].append(dict(name='delete-fresh',old_identity=old.to_json(),delete=closed,
            preserved_application_messages=preserved,unrepresentable_assistant_omitted=True,
            native_state_carried=False,old_id_and_request_denied=True))
        await turn('fresh-session-ordinary-prefill','Ignore the unfinished last lookup. Say goodbye briefly without a tool.',tools_enabled=False)
        assert client.state=='ready'
        result['final_transcript']=client.messages
        result['ledger']=[dict(key=list(k),**v) for k,v in client.ledger.items()]
        result['timings']=list(client.timings)
        result['client_payload_bytes']=client.retained_state_bytes()
        await call(client.retire)
        traces=copy.deepcopy(list(q.backend.session_traces))
        assert q.counts['replay']==q.counts['repack']==0
        for t in traces:
            assert t['prompt_replay']==t['full_cache_repack']==0
            assert set(t['target_offsets']+t['dspark_offsets'])=={t['canonical_frontier']}
        import resource
        import mlx.core as mx
        result.update(status='PASS',traces=traces,counts=q.counts,
                      resources=dict(process_maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                     mlx_active_bytes=mx.get_active_memory(),mlx_cache_bytes=mx.get_cache_memory()),
                      sources=SOURCES)
    except BaseException as exc:
        result.update(status='FAIL',error=repr(exc),client_state=client.state);raise
    finally:
        save();server.should_exit=True;await task
        if not any(s.busy for s in q.backend.sessions.values()):q.backend.close()

if __name__=='__main__':asyncio.run(main())
