"""M38 lifecycle ambiguity gate adapted to server-issued identities; real sockets."""
import asyncio
import json
from pathlib import Path
import socket
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import uvicorn
from ds41f_mlx.serving.internal_mtp import InternalMTPQualificationBackend
from ds41f_mlx.serving.server import create_app
from ds41f_mlx.internal_local_client import InternalLocalClient, ClientStateError
from ds41f_mlx.web_client import RuntimeClient

async def main():
    b=InternalMTPQualificationBackend();app=create_app(backend=b,recipe_path=b.recipe_path)
    fault={'method':None}
    async def wrapped(scope,receive,send):
        async def emit(message):
            await send(message)
            if scope['type']=='http' and scope['method']==fault['method'] and message['type']=='http.response.start':
                fault['method']=None
                raise OSError('M39 synthetic lifecycle body loss after mutation')
        await app(scope,receive,emit)
    sock=socket.socket();sock.bind(('127.0.0.1',0))
    server=uvicorn.Server(uvicorn.Config(wrapped,log_level='warning',lifespan='off',http='h11'))
    task=asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:await asyncio.sleep(.01)
    r=RuntimeClient(f'http://127.0.0.1:{sock.getsockname()[1]}');rows=[]
    async def call(fn,*a):return await asyncio.to_thread(fn,*a)
    try:
        for action in ('delete','create'):
            c=InternalLocalClient(r)
            if action=='delete':await call(c.create)
            issued=b._issued
            sid=c.session_id
            fault['method']='DELETE' if action=='delete' else 'POST'
            try:await call(c.retire) if action=='delete' else await call(c.create)
            except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc))
            else:raise AssertionError('body loss not injected')
            assert error['type']=='IncompleteRead'
            assert c.state=='stopped' and c.lifecycle_uncertain['action']==action
            if action=='delete':assert b.identity_state(sid)=='retired' and not b.sessions
            else:assert b._issued==issued+1 and len(b.sessions)==1
            before=(b._issued,len(b.sessions),len(b.retired_diagnostics))
            for fn in (c.retire,c.create):
                try:await call(fn)
                except ClientStateError:pass
                else:raise AssertionError('client guessed lifecycle')
            assert before==(b._issued,len(b.sessions),len(b.retired_diagnostics))
            rows.append(dict(action=action,error=error,client_state=c.state,uncertain=c.lifecycle_uncertain,
                             mutation_reached=True,no_retry_or_replacement=True))
            if action=='create':await b.close_stateful_session(next(iter(b.sessions))) # external controller cleanup only
        return dict(schema='ds41f.m39.lifecycle-transport.v1',status='PASS',rows=rows,
                    scope='actual loopback lifecycle/socket; controlled ASGI post-header body loss; no model execution')
    finally:server.should_exit=True;await task;b.close()

if __name__=='__main__':
    (ROOT/'artifacts/m39/lifecycle-transport.json').write_text(json.dumps(asyncio.run(main()),indent=2)+'\n')
