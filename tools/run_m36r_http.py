"""Real checkpoint loopback request fence / certificate / single-client tools."""
import asyncio
import copy
import http.client
import json
from pathlib import Path
import socket
import time
import uvicorn
from tools import run_m35_http_qualification as q
from ds41f_mlx.serving.recovery_certificate import LocalToolLedger
OUT=q.ROOT/'artifacts/m36r/http.json'
result=dict(schema='ds41f.m36r.http.v1',status='RUNNING',cases=[],counts=q.counts)
def save(): OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(result,indent=2)+'\n')
def request(method,sid,b=None,seq=None):
    c=http.client.HTTPConnection('127.0.0.1',q.port,timeout=1200)
    headers={'Content-Type':'application/json'}
    if seq is not None: headers['X-DS41F-Request-Sequence']=str(seq)
    c.request(method,'/v1/sessions'+(f'/{sid}' if sid else '')+('/chat/completions' if b is not None and sid else ''),None if b is None else json.dumps(b),headers)
    r=c.getresponse();data=r.read();status=r.status;c.close();return status,json.loads(data)
def open_stream(sid,b,seq):
    c=http.client.HTTPConnection('127.0.0.1',q.port,timeout=1200)
    c.request('POST',f'/v1/sessions/{sid}/chat/completions',json.dumps(b),{'Content-Type':'application/json','X-DS41F-Request-Sequence':str(seq)})
    r=c.getresponse();assert r.status==200,(r.status,r.read());return c,r
def drop(c,r): (c.sock or r.fp.raw._sock).shutdown(socket.SHUT_RDWR);r.close();c.close()
async def call(*a):return await asyncio.to_thread(request,*a)
async def settled(sid):
    for _ in range(2000):
        status,rec=await call('GET',sid);assert status==200
        if rec['outcome_state']!='active':return rec
        await asyncio.sleep(.01)
    raise AssertionError('settlement timeout')
async def observe(sid,b):
    rec=await settled(sid)
    for _ in range(3):assert (await call('GET',sid))[1]==rec
    before=copy.deepcopy(rec);runs=len(q.backend.session_traces)
    for _ in range(3):
        status,out=await call('POST',sid,b,1);assert status==200 and out['outcome_state']==rec['outcome_state']
    assert len(q.backend.session_traces)==runs and (await call('GET',sid))[1]==before
    bad=copy.deepcopy(b);bad['max_tokens']=1
    status,_=await call('POST',sid,bad,1);assert status==400
    assert (await call('GET',sid))[1]==before
    return rec,out
async def main():
    sock=socket.socket();sock.bind(('127.0.0.1',0));q.port=sock.getsockname()[1]
    app=q.create_app(backend=q.backend,recipe_path=q.backend.recipe_path)
    server=uvicorn.Server(uvicorn.Config(app,log_level='warning',lifespan='off',http='h11'))
    task=asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:await asyncio.sleep(.01)
    async def create(sid):assert (await call('POST','',dict(id=sid)))[0]==200
    async def close(sid):assert (await call('DELETE',sid))[0]==200
    try:
        for sid in ['unicode','partial-tool','completed-tool']:
            await create(sid)
            initial=(await call('GET',sid))[1];assert initial['outcome_state']=='not_admitted' and initial['next_sequence']==1
            b=q.body('Output exactly: café 🙂 漢字. Nothing else.',64) if sid=='unicode' else q.initial_body('Berlin')
            b['stream']=True;b['stop']=None
            status,_=await call('POST',sid,b,2);assert status==400
            assert (await call('GET',sid))[1]==initial
            q.backend.delivery_delay_s=.04 if sid!='completed-tool' else .8
            if sid=='completed-tool':
                # Controlled delivery loss at canonical semantic terminal, before transport.
                c,r=await asyncio.to_thread(open_stream,sid,b,1)
                while not q.backend.progress.get('certificate'):await asyncio.sleep(.01)
                assert q.backend.progress['certificate']['representable']
                drop(c,r);observed=dict(controlled_terminal_window=True,received_body_bytes=0)
            else:
                def cut():
                    c,r=open_stream(sid,b,1);raw=bytearray();events=[]
                    if sid=='unicode':
                        while True:
                            byte=r.read(1);assert byte;raw.extend(byte)
                            if byte[0]>=0xc0:break
                        try:raw.decode();raise AssertionError('expected incomplete UTF8')
                        except UnicodeDecodeError:pass
                    else:
                        while True:
                            line=r.readline();assert line
                            if not line.startswith(b'data: '):continue
                            event=json.loads(line[6:]);events.append(event)
                            if any(x.get('function',{}).get('arguments') for x in event.get('choices',[{}])[0].get('delta',{}).get('tool_calls',[])):break
                    drop(c,r);return dict(bytes_hex=raw.hex(),events=events)
                observed=await asyncio.to_thread(cut)
            rec,out=await observe(sid,b)
            entry=dict(name=sid,request=b,observed=observed,recovered=rec,reobserved=out,duplicate_nonmutating=True,body_mismatch_atomic=True)
            if sid=='partial-tool':
                assert rec['outcome_state']=='unrecoverable' and not out['certificate']['representable']
                ledger=LocalToolLedger()
                try:ledger.execute(sid,1,out,lambda call:(_ for _ in ()).throw(AssertionError('executed incomplete')))
                except ValueError:pass
                else:raise AssertionError('tool executable')
                bad=copy.deepcopy(out['certificate']['witness']);bad['max_tokens']=32
                before=copy.deepcopy(rec)
                status,_=await call('POST',sid,bad,2);assert status==400
                assert (await call('GET',sid))[1]==before
                entry.update(tool_executions=0,continuation_rejected_atomic=True)
            else:
                assert rec['outcome_state']=='recoverable' and out['certificate']['representable']
                next_body=copy.deepcopy(b);next_body['stream']=False;next_body['max_tokens']=64
                next_body['messages'].append(out['response']['choices'][0]['message'])
                if sid=='completed-tool':
                    ledger=LocalToolLedger();effects=[]
                    def effect(call):effects.append(call['id']);return q.tool_stub(call['function']['name'],call['function']['arguments'])
                    results=ledger.execute(sid,1,out,effect)
                    for _ in range(3):assert ledger.execute(sid,1,out,effect)==results
                    assert len(effects)==1
                    next_body['messages']+=results;next_body['tool_choice']='auto'
                    entry.update(tool_executions=len(effects),execution_reobservations=4)
                else:next_body['messages'].append(dict(role='user',content='Say goodbye briefly.'))
                prepared=q.prepare_request('chat_completions',json.dumps(next_body).encode(),tokenizer=app.state.recipe_tokenizer,recipe_path=q.backend.recipe_path)
                assert prepared.token_ids[:len(q.backend.sessions[sid].canonical)]==q.backend.sessions[sid].canonical
                q.backend.delivery_delay_s=0
                status,reply=await call('POST',sid,next_body,2);assert status==200,reply
                after=await settled(sid);assert after['outcome_state']=='recoverable'
                runs=len(q.backend.session_traces);status,_=await call('POST',sid,b,1);assert status==400
                assert len(q.backend.session_traces)==runs and (await call('GET',sid))[1]==after
                entry.update(continuation_exact_prefix=True,after=after,expired_retry_atomic=True)
            result['cases'].append(entry);save();await close(sid)
        # Real active / pre-generation reservation retry race; then complete text length.
        sid='active';await create(sid);b=q.body('Say hello.',8);b['stream']=True
        q.backend.delivery_delay_s=.8
        c,r=await asyncio.to_thread(open_stream,sid,b,1)
        active=(await call('GET',sid))[1];assert active['outcome_state']=='active'
        runs=len(q.backend.session_traces);status,_=await call('POST',sid,b,1);assert status==409
        assert len(q.backend.session_traces)==runs
        drop(c,r);rec,out=await observe(sid,b)
        result['cases'].append(dict(name=sid,active_retry_no_generation=True,recovered=rec));await close(sid)
        assert q.counts['replay']==q.counts['repack']==0
        result.update(status='PASS',sources={p:q.sha(q.ROOT/p) for p in ['ds41f_mlx/serving/internal_mtp.py','ds41f_mlx/serving/server.py','ds41f_mlx/serving/recovery_certificate.py','ds41f_mlx/serving/request_fence.py','tools/run_m36r_http.py']})
    except BaseException as exc:result.update(status='FAIL',error=repr(exc));raise
    finally:
        save();server.should_exit=True;await task
        if not any(s.busy for s in q.backend.sessions.values()):q.backend.close()
if __name__=='__main__':asyncio.run(main())
