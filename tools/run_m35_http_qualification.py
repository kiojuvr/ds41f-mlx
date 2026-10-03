"""Bounded real loopback HTTP/SSE M35 experiment (explicit object injection).
Run: DS41F_OMLX_PATH=/tmp/ds41f-m33-omlx /tmp/ds41f-m32-qual/bin/python tools/run_m35_http_qualification.py
"""
import asyncio
import copy
import hashlib
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), '/tmp/ds41f-m33-omlx']
os.environ['DS41F_OMLX_PATH'] = '/tmp/ds41f-m33-omlx'
os.environ['DS41F_RECIPE_PATH'] = '/tmp/ds41f-m32-recipe'
import uvicorn
from ds41f_mlx.serving.internal_mtp import InternalMTPQualificationBackend
from ds41f_mlx.serving.server import create_app, prepare_request
from tools.run_m11_tool_boundary_qualification import initial_body, tool_stub

OUT = ROOT/'artifacts/m35/http.json'
OUT.parent.mkdir(parents=True, exist_ok=True)
result = dict(schema='ds41f.m35.http.v1', status='RUNNING', cases=[], admission=[],
              base_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip())
def save():
    tmp=OUT.with_suffix('.tmp'); tmp.write_text(json.dumps(result, indent=2)+'\n'); tmp.replace(OUT)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ids=json.loads((ROOT/'artifacts/m34/identities.json').read_text())
import deepseek_recipe._native as native
assert sha(native.__file__)==json.loads((ROOT/'artifacts/m33/runtime-identities.json').read_text())['recipe_native_sha256']
for p,h in ids['current_runtime_sources'].items(): assert sha(ROOT/p)==h
result['identities']=ids
result['sources']={p:sha(ROOT/p) for p in ['ds41f_mlx/serving/internal_mtp.py','ds41f_mlx/serving/server.py','tools/run_m35_http_qualification.py']}
counts=dict(replay=0,repack=0,verify=0,proposal=0,fresh_target_allocations=0)
from ds41f_mlx.runtime.omlx_decode import OMLXDecodeStateAdapter
from omlx.patches.mlx_lm_mtp import batch_generator as mtp

def replay(*a,**kw): counts['replay']+=1; raise AssertionError('replay forbidden')
def repack(*a,**kw): counts['repack']+=1; raise AssertionError('repack forbidden')
mtp._reconcile_mtp_to_standard=replay
OMLXDecodeStateAdapter.admit=repack
OMLXDecodeStateAdapter._pack_cache_array=repack
old_cycle=mtp._run_verify_cycle_chain
def cycle(gb,state,*a,**kw):
    counts['verify']+=1
    ret=old_cycle(gb,state,*a,**kw)
    offsets=[int(c.size()) for c in gb.prompt_cache]+[int(c.offset) for c in state.mtp_cache]
    assert len(set(offsets))==1
    return ret
mtp._run_verify_cycle_chain=cycle
backend=InternalMTPQualificationBackend(omlx_path=Path('/tmp/ds41f-m33-omlx'),recipe_path=Path('/tmp/ds41f-m32-recipe'))
original_load=backend.load
instrumented=False
def checked_load():
    global instrumented
    original_load()
    if instrumented: return
    instrumented=True
    lm=backend._model.language_model
    make_cache=lm.make_cache
    def fresh_cache(*a,**kw):
        active=[r for r in backend.sessions.values() if r.busy]
        if len(active)!=1 or active[0].cache is not None:
            return repack(*a,**kw)
        counts['fresh_target_allocations']+=1
        return make_cache(*a,**kw)
    lm.make_cache=fresh_cache
    old_proposal=lm.dspark_forward
    def proposal(*a,**kw): counts['proposal']+=1;return old_proposal(*a,**kw)
    lm.dspark_forward=proposal
backend.load=checked_load
port=0

def http_request(method,path,body=None):
    c=http.client.HTTPConnection('127.0.0.1',port,timeout=1200)
    c.request(method,path,None if body is None else json.dumps(body),{'Content-Type':'application/json'})
    r=c.getresponse(); raw=r.read(); status=r.status; c.close()
    return status,json.loads(raw)

def open_sse(sid,body):
    c=http.client.HTTPConnection('127.0.0.1',port,timeout=1200)
    c.request('POST',f'/v1/sessions/{sid}/chat/completions',json.dumps(body),{'Content-Type':'application/json'})
    r=c.getresponse(); assert r.status==200,(r.status,r.read())
    return c,r

def consume(sid,body,cut=None,slow=False):
    t0=time.perf_counter(); c,r=open_sse(sid,body); rows=[]; done=False
    while True:
        line=r.readline()
        if not line: break
        if not line.startswith(b'data: '): continue
        data=line[6:].strip().decode()
        rows.append(dict(data=data,received_s=time.perf_counter()-t0,
                         server_canonical_at_receive=backend.progress['canonical_emitted']))
        if data=='[DONE]': done=True; break
        if slow and len(rows)%7==0: time.sleep(.6)
        if cut and len(rows)>=cut: break
    # shutdown actually closes the HTTPResponse-owned descriptor too
    if not done:
        c.sock.shutdown(socket.SHUT_RDWR) if c.sock else r.fp.raw._sock.shutdown(socket.SHUT_RDWR)
    r.close();c.close()
    return dict(rows=rows,done=done,client_wall_s=time.perf_counter()-t0)

async def call(*a): return await asyncio.to_thread(http_request,*a)
async def idle(sid):
    t0=time.perf_counter()
    for _ in range(10000):
        status,rec=await call('GET',f'/v1/sessions/{sid}')
        assert status==200
        if rec['state']!='busy':
            assert rec['state']=='idle',rec
            return rec,time.perf_counter()-t0
        await asyncio.sleep(.01)
    raise AssertionError('did not quiesce')

def body(text,limit=128):
    return dict(model='deepseek-v4.1-flash',messages=[dict(role='user',content=text)],reasoning_effort='none',temperature=0,max_tokens=limit)
LONG='Write a detailed practical guide to designing a reliable local weather assistant. Cover tool errors, caching, testing and user experience in eight numbered sections, with examples. Aim for about 450 words. Do not call a tool.'

async def run_case(name,sid,b,*,stream=True,cut=None,slow=False,delay=0.,boundary=None):
    b['stream']=stream;backend.delivery_delay_s=delay
    start=time.perf_counter()
    if not stream:
        status,out=await call('POST',f'/v1/sessions/{sid}/chat/completions',b)
        assert status==200,out
        delivered=dict(response=out,client_wall_s=time.perf_counter()-start,done=True,rows=[])
    elif boundary:
        c,r=await asyncio.to_thread(open_sse,sid,b)
        for _ in range(100000):
            p=backend.progress
            if (boundary=='terminal' and 'quiescence' in p) or (boundary=='suffix' and p['canonical_emitted']>=1 and p['iterator_yields']==0): break
            await asyncio.sleep(.001)
        else: raise AssertionError('boundary not reached')
        at=copy.deepcopy(backend.progress)
        sock=c.sock or r.fp.raw._sock
        sock.shutdown(socket.SHUT_RDWR);r.close();c.close()
        delivered=dict(rows=[],done=False,disconnect_snapshot=at,client_wall_s=time.perf_counter()-start)
    else:
        delivered=await asyncio.to_thread(consume,sid,b,cut,slow)
    rec,wait=await idle(sid)
    trace=rec['last_turn']
    rows=[row for row in delivered['rows'] if row['data']!='[DONE]']
    for i,row in enumerate(rows):
        assert i<len(trace['rows'])
        row['canonical_input_upper_bound']=trace['rows'][i]['canonical_ordinal']
        assert json.loads(row['data']) == trace['protocol_events'][i]
    assert trace['client_acknowledged_ordinal'] == 0  # sends are not acknowledgements
    if cut:
        assert trace['cancelled'] and trace['generated'] < 100, 'disconnect must be admitted at a response boundary, not ignored to completion'
    if slow:
        assert max(row['server_canonical_at_receive']-row['canonical_input_upper_bound'] for row in rows)>20
    if boundary=='suffix':
        assert trace['cancelled'] and trace['canonical_generated']
    if boundary=='terminal':
        assert 'quiescence' in delivered['disconnect_snapshot']
    entry=dict(name=name,request=copy.deepcopy(b),transport=delivered,runtime=trace,post_disconnect_idle_wait_s=wait)
    result['cases'].append(entry);save()
    assert trace['prompt_replay']==trace['full_cache_repack']==0
    assert set(trace['target_offsets']+trace['dspark_offsets'])=={trace['canonical_frontier']}
    assert trace['prediction_retired'] and trace['queue_empty']
    return trace['response']['choices'][0]['message']

async def main():
    global port
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    app=create_app(backend=backend,recipe_path=backend.recipe_path)
    server=uvicorn.Server(uvicorn.Config(app,log_level='warning',lifespan='off'))
    task=asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started: await asyncio.sleep(.01)
    result['server']=dict(host='127.0.0.1',port=port,uvicorn=uvicorn.__version__,admission='explicit Python backend object injection; no public selector',single_flight=True,max_live_sessions=1,max_response_tokens=768,max_total_tokens=8192)
    async def create(sid):
        status,rec=await call('POST','/v1/sessions',{'id':sid});assert status==200,rec
    async def close(sid):
        status,rec=await call('DELETE',f'/v1/sessions/{sid}');assert status==200 and rec['state']=='closed'
    try:
        await create('ordinary')
        b=body('Say hello in one short sentence.')
        msg=await run_case('nonstream','ordinary',b,stream=False)
        b['messages'] += [msg,dict(role='user',content='Now say goodbye in one short sentence.')]
        await run_case('nonstream_continuation','ordinary',b,stream=False)
        await close('ordinary')
        await create('tool')
        b=initial_body('Paris');b['stop']=None
        msg=await run_case('tool_sse','tool',b)
        call0=msg['tool_calls'][0]
        b['messages'] += [msg,dict(role='tool',tool_call_id=call0['id'],content=tool_stub(call0['function']['name'],call0['function']['arguments']))]
        b['tool_choice']='auto'
        await run_case('tool_result_reentry','tool',b)
        await close('tool')
        await create('toolterminal')
        b=initial_body('Berlin');b['stop']=None
        msg=await run_case('semantic_terminal_disconnect','toolterminal',b,delay=.03,boundary='terminal')
        assert result['cases'][-1]['runtime']['terminal_matches']
        call0=msg['tool_calls'][0]
        b['messages'] += [msg,dict(role='tool',tool_call_id=call0['id'],content=tool_stub(call0['function']['name'],call0['function']['arguments']))]
        b['tool_choice']='auto'
        await run_case('terminal_tool_result_reentry','toolterminal',b,stream=False)
        await close('toolterminal')
        for name,kw in [('normal_sse',{}),('slow_sse',dict(slow=True)),('disconnect_long',dict(cut=40)),('undelivered_suffix',dict(delay=.8,boundary='suffix')),('terminal_disconnect',dict(delay=.03,boundary='terminal'))]:
            await create(name)
            b=body('Say exactly Hello there.' if name=='terminal_disconnect' else LONG,768)
            msg=await run_case(name,name,b,**kw)
            # GET yields canonical response, including undelivered drain, from
            # Python authority. Re-entry uses ordinary recipe encoding only.
            b['messages'] += [msg,dict(role='user',content='Acknowledge in one short sentence. Do not call tools.')]
            b['max_tokens']=96
            await run_case(name+'_continuation',name,b,stream=False)
            await close(name)
        await create('rustdrop')
        b=body(LONG,768);b['stream']=True;backend.delivery_delay_s=0.
        request_file=OUT.parent/'rust-request.json';request_file.write_text(json.dumps(b))
        tick=time.perf_counter()
        proc=await asyncio.to_thread(subprocess.run,[str(ROOT/'target/debug/m35_internal_transport'),str(port),'rustdrop',str(request_file)],capture_output=True,text=True)
        assert proc.returncode==0,proc.stderr
        rec,wait=await idle('rustdrop');trace=rec['last_turn']
        delivered=[json.loads(line) for line in proc.stdout.splitlines()[:-1]]
        assert delivered==trace['protocol_events'][:len(delivered)]
        assert trace['cancelled'] and trace['generated']<100
        result['cases'].append(dict(name='rust_iterator_drop',request=copy.deepcopy(b),runtime=trace,
            transport=dict(received_events=delivered,stdout=proc.stdout,client_wall_s=time.perf_counter()-tick),post_disconnect_idle_wait_s=wait))
        b['messages'] += [trace['response']['choices'][0]['message'],dict(role='user',content='Acknowledge in one short sentence.')]
        b['max_tokens']=96
        await run_case('rust_drop_continuation','rustdrop',b,stream=False)
        await close('rustdrop')
        await create('admission')
        b=body(LONG,768);b['stream']=True;backend.delivery_delay_s=.8
        c,r=await asyncio.to_thread(open_sse,'admission',b)
        while not backend.progress['canonical_emitted']: await asyncio.sleep(.01)
        before=list(backend.sessions['admission'].canonical)
        for method,path,payload,expected in [
            ('POST','/v1/sessions/admission/chat/completions',b,409),
            ('POST','/v1/sessions',{'id':'overlap'},409),
            ('DELETE','/v1/sessions/admission',None,409),
            ('POST','/v1/sessions/admission/persist',{'artifact_root':'/tmp/m35-forbidden'},409),
            ('POST','/v1/sessions/restore',{'artifact_path':'/tmp/m35-forbidden'},409),
            ('POST','/v1/sessions/admission/chat/completions',dict(b,stop=['halt']),400),
            ('POST','/v1/sessions/admission/chat/completions',dict(b,model='bad'),400),
            ('POST','/v1/sessions/admission/chat/completions',dict(b,max_tokens=-1),400),
            ('POST','/v1/sessions/admission/chat/completions',dict(b,max_tokens=769),400),
            ('POST','/v1/sessions/admission/chat/completions',body('hello '*200000),400),
            ('POST','/v1/sessions/admission/chat/completions',[],400),
            ('POST','/v1/chat/completions',body('Unsupported stateless MTP'),400),
        ]:
            status,reply=await call(method,path,payload);assert status==expected,(status,reply)
            assert backend.sessions['admission'].canonical==before
            result['admission'].append(dict(method=method,path=path,status=status,response=reply,retained_history_unchanged=True))
        (c.sock or r.fp.raw._sock).shutdown(socket.SHUT_RDWR);r.close();c.close()
        await idle('admission')
        before=backend.sessions['admission'].to_json()
        status,reply=await call('POST','/v1/sessions/admission/chat/completions',body('Unrelated history'))
        assert status==400 and backend.sessions['admission'].to_json()==before
        result['admission'].append(dict(case='nonprefix_idle',status=status,mutation_atomic=True))
        await close('admission')
        assert counts['replay']==counts['repack']==0
        # Same checkpoint, prompt, greedy sampler and native guarded runtime,
        # synchronous direct loop: separate HTTP/event-loop overhead from model
        # execution without borrowing a differently predictable 4K benchmark.
        await create('directcontrol')
        rec=backend.sessions['directcontrol'];rec.busy=True
        prepared=prepare_request('chat_completions',json.dumps(dict(body(LONG,768),stream=True)).encode(),tokenizer=app.state.recipe_tokenizer,recipe_path=backend.recipe_path)
        trace=dict(response_id='directcontrol',t0=time.perf_counter(),generated=0,decode_s=0.,first_canonical_s=None)
        def direct():
            backend._start(rec,prepared,app.state.recipe_tokenizer,trace)
            while not rec.guard.finished:
                if backend._next(rec,trace) is None: break
            backend._settle(rec,trace)
            trace['elapsed_s']=time.perf_counter()-trace['t0']
        try: await backend._call(direct)
        finally: rec.busy=False
        normal=next(c for c in result['cases'] if c['name']=='normal_sse')['runtime']
        assert trace['canonical_generated']==normal['canonical_generated']
        assert trace['response']['choices']==normal['response']['choices']
        result['direct_control']=trace
        await close('directcontrol')
        assert counts['fresh_target_allocations']==11
        assert counts['replay']==counts['repack']==0
        result.update(status='PASS',counts=counts)
    except BaseException as exc:
        result.update(status='FAIL',error=repr(exc),counts=counts)
        raise
    finally:
        save();server.should_exit=True;await task
        if not any(s.busy for s in backend.sessions.values()): backend.close()

if __name__=='__main__': asyncio.run(main())
