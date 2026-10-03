"""Actual InternalLocalClient handle drop under bounded real h11 backpressure.
Synthetic SSE comment expansion, real checkpoint/canonical/cache behavior.
"""
import asyncio
import copy
import gc
import json
import socket
import time
from pathlib import Path
import uvicorn
from tools import run_m35_http_qualification as q
from ds41f_mlx.internal_local_client import InternalLocalClient, OwnedStream
from ds41f_mlx.web_client import RuntimeClient
OUT=q.ROOT/'artifacts/m38/client-pressure.json'

async def main():
    sock=socket.socket();sock.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,8192);sock.bind(('127.0.0.1',0))
    app=q.create_app(backend=q.backend,recipe_path=q.backend.recipe_path);sends=[];padding=262144
    async def wrapped(scope,receive,send):
        sse=False
        async def emit(message):
            nonlocal sse
            if message['type']=='http.response.start':sse=any(k.lower()==b'content-type' and b'text/event-stream' in v for k,v in message['headers'])
            if sse and message['type']=='http.response.body' and message.get('body') and padding:
                message=dict(message,body=b':'+b'x'*padding+b'\n'+message['body']); t=time.perf_counter();row=dict(bytes=len(message['body']));sends.append(row)
                try:await send(message)
                finally:row['wait_s']=time.perf_counter()-t
            else:await send(message)
        await app(scope,receive,emit)
    server=uvicorn.Server(uvicorn.Config(wrapped,log_level='warning',lifespan='off',http='h11'))
    task=asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:await asyncio.sleep(.01)
    c=InternalLocalClient(RuntimeClient(f'http://127.0.0.1:{sock.getsockname()[1]}'))
    result=dict(schema='ds41f.m38.client-pressure.v1',status='RUNNING')
    async def call(fn,*a,**kw):return await asyncio.to_thread(fn,*a,**kw)
    async def settle():
        for _ in range(4000):
            out=await call(c.reconcile)
            if isinstance(out,OwnedStream):await call(lambda:list(out))
            elif out.get('outcome_state')!='active':return out
            await asyncio.sleep(.01)
        raise AssertionError('settlement deadline')
    try:
        await call(c.create,'m38-client-pressure')
        options=dict(model='deepseek-v4.1-flash',temperature=0,reasoning_effort='none',stream=True,max_tokens=768)
        stream=await call(c.submit,[dict(role='user',content=q.LONG)],options=options)
        assert isinstance(stream,OwnedStream)
        for _ in range(12000):
            if sends:break
            await asyncio.sleep(.01)
        else:raise AssertionError('first-send deadline')
        samples=[]
        for _ in range(30):
            await asyncio.sleep(.1)
            samples.append(dict(generated=q.backend.progress['generated'],sends=len(sends),completed=sum('wait_s' in s for s in sends)))
        assert samples[-1]['generated']==samples[-10]['generated']
        assert c.messages==[] and c.next_sequence==1
        identity=c.identity;t=time.perf_counter()
        # Drop the only stream handle; client itself never retains it.
        del stream;gc.collect()
        assert c.state=='ambiguous'
        out=await settle();wait=time.perf_counter()-t;padding=0
        assert out['outcome_state']=='recoverable'
        assert c.messages==json.loads(identity.body)['messages']+[out['response']['choices'][0]['message']]
        first=copy.deepcopy(q.backend.progress);counts=q.counts.copy()
        for _ in range(3):assert await settle()==out
        assert q.counts==counts
        assert max(s.get('wait_s',0) for s in sends)>1
        options.update(stream=False,max_tokens=64)
        await call(c.submit,[dict(role='user',content='Acknowledge briefly without tools.')],options=options)
        await settle();second=copy.deepcopy(q.backend.progress)
        await call(c.retire)
        for trace in [first,second]:
            assert trace['prompt_replay']==trace['full_cache_repack']==0
            assert set(trace['target_offsets']+trace['dspark_offsets'])=={trace['canonical_frontier']}
            assert trace['queue_empty'] and trace['prediction_retired']
        result.update(status='PASS',samples=samples,sends=sends,identity=identity.to_json(),outcome=out,
            traces=[first,second],counts=q.counts,disconnect_to_settlement_s=wait,client_final_state=c.state,
            retained_state_bytes=c.retained_state_bytes(),padding_bytes=262144,listener_sndbuf=8192,
            scope='real checkpoint/HTTP/socket/InternalLocalClient; synthetic bounded SSE comments only',
            sources={p:q.sha(q.ROOT/p) for p in ['tools/run_m38_client_pressure.py','ds41f_mlx/internal_local_client.py','ds41f_mlx/serving/internal_mtp.py']})
    except BaseException as exc:result.update(status='FAIL',error=repr(exc));raise
    finally:
        OUT.write_text(json.dumps(result,indent=2)+'\n');server.should_exit=True;await task
        if not any(s.busy for s in q.backend.sessions.values()):q.backend.close()

if __name__=='__main__':asyncio.run(main())
