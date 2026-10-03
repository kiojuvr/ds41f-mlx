"""Real checkpoint/recipe/HTTP exploratory M36 gate. No release selector.
Uses M35 instrumentation unchanged; preserves any ordinary re-entry blocker.
"""
import asyncio
import copy
import json
from pathlib import Path
import socket
import subprocess
import time
import uvicorn
from tools import run_m35_http_qualification as q

OUT = q.ROOT / 'artifacts/m36/probe.json'
OUT.parent.mkdir(parents=True, exist_ok=True)
result = dict(schema='ds41f.m36.recovery-probe.v1', status='RUNNING',
              base_commit=q.result['base_commit'], cases=[], counts=q.counts)

def save():
    OUT.write_text(json.dumps(result, indent=2)+'\n')

async def main():
    sock=socket.socket()
    sock.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,8192)
    sock.bind(('127.0.0.1',0));q.port=sock.getsockname()[1]
    app=q.create_app(backend=q.backend,recipe_path=q.backend.recipe_path)
    padding=0
    sends=[]
    async def pressure_app(scope,receive,send):
        is_sse=False
        async def measured(message):
            nonlocal is_sse
            if message['type']=='http.response.start':
                is_sse=any(k.lower()==b'content-type' and b'text/event-stream' in v for k,v in message['headers'])
            if is_sse and message['type']=='http.response.body' and message.get('body') and padding:
                message=dict(message,body=b':'+b'x'*padding+b'\n'+message['body'])
                start=time.perf_counter()
                row=dict(bytes=len(message['body']),start_s=start)
                sends.append(row)
                try: await send(message)
                finally: row['wait_s']=time.perf_counter()-start
            else: await send(message)
        await app(scope,receive,measured)
    server=uvicorn.Server(uvicorn.Config(pressure_app,log_level='warning',lifespan='off',http='h11'))
    task=asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started: await asyncio.sleep(.01)
    result['transport']=dict(uvicorn=uvicorn.__version__,http='h11',listener_sndbuf=sock.getsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF))
    async def create(sid):
        status,rec=await q.call('POST','/v1/sessions',dict(id=sid));assert status==200,rec
    async def close(sid):
        status,rec=await q.call('DELETE',f'/v1/sessions/{sid}');assert status==200,rec
    def shutdown(c,r):
        (c.sock or r.fp.raw._sock).shutdown(socket.SHUT_RDWR);r.close();c.close()
    async def settled(sid):
        tick=time.perf_counter()
        for _ in range(1000):
            status,rec=await q.call('GET',f'/v1/sessions/{sid}')
            assert status==200
            if rec['state']!='busy':return rec,time.perf_counter()-tick
            await asyncio.sleep(.01)
        raise AssertionError('settlement timeout')
    async def recover(sid,b,observed):
        rec,wait=await settled(sid)
        before=copy.deepcopy(rec)
        for _ in range(3):
            status,again=await q.call('GET',f'/v1/sessions/{sid}')
            assert status==200 and again==before
        retry_status,retry_reply=await q.call('POST',f'/v1/sessions/{sid}/chat/completions',b)
        _,retry_after=await q.call('GET',f'/v1/sessions/{sid}')
        assert retry_status==400 and retry_after==before
        msg=rec['last_turn']['response']['choices'][0]['message']
        continuation=copy.deepcopy(b)
        continuation['stream']=False;continuation['max_tokens']=64
        continuation['messages'].append(msg)
        for call in msg.get('tool_calls',[]):
            # A non-execution sentinel, not a guessed tool execution/result.
            continuation['messages'].append(dict(role='tool',tool_call_id=call['id'],content='NOT EXECUTED: interrupted call'))
        continuation['messages'].append(dict(role='user',content='Acknowledge in one short sentence. Do not call tools.'))
        continuation['tool_choice']='auto' if 'tools' in b else None
        if continuation['tool_choice'] is None: del continuation['tool_choice']
        # Prove the ordinary recipe boundary independently of poison admission.
        # This is request conversion, never model/token-history replay.
        try:
            prepared=q.prepare_request('chat_completions',json.dumps(continuation).encode(),tokenizer=app.state.recipe_tokenizer,recipe_path=q.backend.recipe_path)
            original=q.prepare_request('chat_completions',json.dumps(b).encode(),tokenizer=app.state.recipe_tokenizer,recipe_path=q.backend.recipe_path)
            canonical=original.token_ids+rec['last_turn']['canonical_generated']
            conversion=dict(succeeded=True,canonical_length=len(canonical),
                exact_prefix=prepared.token_ids[:len(canonical)]==canonical,
                first_mismatch=next((i for i,(a,z) in enumerate(zip(prepared.token_ids,canonical)) if a!=z),None))
        except Exception as exc:
            conversion=dict(succeeded=False,error=str(exc))
        status,reply=await q.call('POST',f'/v1/sessions/{sid}/chat/completions',continuation)
        _,after=await q.call('GET',f'/v1/sessions/{sid}')
        entry=dict(name=sid,request=b,observed=observed,recovered=rec,
                   repeated_get_idempotent=True,recovery_wait_s=wait,
                   duplicate_retry=dict(status=retry_status,response=retry_reply,mutation_atomic=True),
                   ordinary_conversion=conversion,
                   continuation_request=continuation,continuation_status=status,
                   continuation_response=reply,rejection_mutation_atomic=(after==before if status!=200 else None),
                   after=after)
        result['cases'].append(entry);save()
        await close(sid)
        return entry
    try:
        # A real official recipe tool-argument delta, cut before DSML completion.
        await create('partial-tool')
        b=q.initial_body('Berlin');b['stop']=None;b['stream']=True
        q.backend.delivery_delay_s=.025
        def tool_prefix():
            c,r=q.open_sse('partial-tool',b);rows=[]
            while True:
                line=r.readline()
                if not line: raise AssertionError('no incomplete tool arguments')
                if not line.startswith(b'data: '):continue
                raw=line[6:].strip();event=json.loads(raw);rows.append(event)
                calls=event.get('choices',[{}])[0].get('delta',{}).get('tool_calls',[])
                if any(call.get('function',{}).get('arguments') for call in calls):break
            shutdown(c,r)
            return dict(events=rows,tool_executions=0)
        observed=await asyncio.to_thread(tool_prefix)
        await recover('partial-tool',b,observed)
        # Interrupt inside a multibyte Unicode JSON/SSE frame, not a parsed event.
        await create('utf8-byte')
        b=q.body('Output exactly: café 🙂 漢字. Nothing else.',64);b['stream']=True
        def byte_prefix():
            c,r=q.open_sse('utf8-byte',b);raw=bytearray()
            while True:
                byte=r.read(1)
                if not byte:raise AssertionError('no multibyte output')
                raw.extend(byte)
                if byte[0]>=0xc0:break
            shutdown(c,r)
            try:raw.decode('utf-8');invalid=False
            except UnicodeDecodeError:invalid=True
            assert invalid
            return dict(bytes_hex=raw.hex(),incomplete_utf8=True)
        observed=await asyncio.to_thread(byte_prefix)
        await recover('utf8-byte',b,observed)
        # Bounded SSE comment expansion drives actual uvicorn write flow control.
        # Comments carry no model/protocol authority and are never retained.
        await create('socket-pressure')
        b=q.body(q.LONG,768);b['stream']=True
        q.backend.delivery_delay_s=0.;padding=262144
        c,r=await asyncio.to_thread(q.open_sse,'socket-pressure',b)
        samples=[]
        for _ in range(30):
            await asyncio.sleep(.1)
            import mlx.core as mx
            samples.append(dict(t=time.perf_counter(),generated=q.backend.progress['generated'],
                                sends=len(sends),completed=sum('wait_s' in s for s in sends),
                                rss_kib=int(subprocess.check_output(['ps','-o','rss=','-p',str(q.os.getpid())],text=True).strip()),
                                mlx_active_bytes=mx.get_active_memory(),mlx_cache_bytes=mx.get_cache_memory()))
        tick=time.perf_counter();shutdown(c,r)
        rec,wait=await settled('socket-pressure')
        padding=0
        assert max(s['wait_s'] for s in sends)>1, sends
        assert samples[-1]['generated']==samples[-10]['generated']
        result['cases'].append(dict(name='socket-pressure',request=b,samples=samples,sends=sends,
            comment_padding_bytes=262144,pressure_duration_s=3,
            disconnect_to_idle_s=time.perf_counter()-tick,recovered=rec,
            recovery_wait_s=wait,bound='pull-based single-frame send; 768 model responses and 8192 total tokens'))
        await close('socket-pressure')
        result['status']='PROBE_COMPLETE'
        result['identities']=q.ids
        result['sources']={p:q.sha(q.ROOT/p) for p in ['tools/run_m36_recovery_probe.py','ds41f_mlx/serving/internal_mtp.py','ds41f_mlx/serving/server.py']}
    except BaseException as exc:
        result.update(status='FAIL',error=repr(exc));raise
    finally:
        save();server.should_exit=True;await task
        q.backend.close()

if __name__=='__main__':asyncio.run(main())
