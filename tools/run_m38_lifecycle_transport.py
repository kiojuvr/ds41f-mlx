"""Targeted ASGI faults over real loopback sockets; lifecycle executes, body is lost.
No checkpoint generation is needed to exercise create/DELETE ownership.
"""
import asyncio
import json
from pathlib import Path
import socket
import uvicorn
from ds41f_mlx.serving.internal_mtp import InternalMTPQualificationBackend
from ds41f_mlx.serving.server import create_app
from ds41f_mlx.internal_local_client import InternalLocalClient, ClientStateError
from ds41f_mlx.web_client import RuntimeClient

async def main():
    backend=InternalMTPQualificationBackend(omlx_path=Path('/tmp/ds41f-m33-omlx'),recipe_path=Path('/tmp/ds41f-m32-recipe'))
    app=create_app(backend=backend,recipe_path=backend.recipe_path);fault={'method':None}
    async def wrapped(scope,receive,send):
        async def emit(message):
            await send(message)
            if scope['type']=='http' and scope['method']==fault['method'] and message['type']=='http.response.start':
                fault['method']=None
                raise OSError('M38 synthetic fault: headers sent, declared body never delivered')
        await app(scope,receive,emit)
    sock=socket.socket();sock.bind(('127.0.0.1',0))
    server=uvicorn.Server(uvicorn.Config(wrapped,log_level='warning',lifespan='off',http='h11'))
    task=asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:await asyncio.sleep(.01)
    runtime=RuntimeClient(f'http://127.0.0.1:{sock.getsockname()[1]}'); rows=[]
    async def call(fn,*a):return await asyncio.to_thread(fn,*a)
    try:
        for action in ['delete','create']:
            c=InternalLocalClient(runtime);sid=f'm38-body-loss-{action}'
            if action=='delete':await call(c.create,sid)
            fault['method']='DELETE' if action=='delete' else 'POST'
            try:await call(c.retire) if action=='delete' else await call(c.create,sid)
            except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc))
            else:raise AssertionError('body loss not injected')
            assert error['type']=='IncompleteRead',error
            assert c.state=='stopped' and c.lifecycle_uncertain['action']==action
            assert backend.sessions[sid].closed==(action=='delete')
            size=len(backend.sessions)
            for fn in [c.retire,lambda:c.create('forbidden')]:
                try:await call(fn)
                except ClientStateError:pass
                else:raise AssertionError('manual reconciliation bypass')
            assert len(backend.sessions)==size
            rows.append(dict(action=action,error=error,client_state=c.state,uncertain=c.lifecycle_uncertain,
                mutation_reached=True,server_closed=backend.sessions[sid].closed,retry_or_replacement=False))
            if action=='create':await call(runtime.close_session,sid) # external controller cleanup only
        Path('artifacts/m38/lifecycle-transport.json').write_text(json.dumps(dict(schema='ds41f.m38.lifecycle-transport.v1',
            status='PASS',scope='actual HTTP/socket/lifecycle; synthetic post-header ASGI failure; no checkpoint generation',rows=rows),indent=2)+'\n')
    finally:
        server.should_exit=True;await task;backend.close()

if __name__=='__main__':asyncio.run(main())
