"""Actual h11 socket gates: ingress on operator process; synthetic send-pressure fixture.

No model execution in this gate. Timeout values are the real 30-second policy.
Native disconnect/settlement is separately exercised by composed acceptance.
"""
import asyncio
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit

from ds41f_mlx.mtp_profile import PROFILE, LIMITS
from ds41f_mlx.web_client import RuntimeClient

OUT=Path(sys.argv[1]);OUT.parent.mkdir(parents=True,exist_ok=True)
result=dict(schema='ds41f.m41.sockets.v1',status='RUNNING',limits=LIMITS,cases=[])

def raw(port,parts,timeout=40):
    with socket.create_connection(('127.0.0.1',port),timeout=timeout) as sock:
        for part in parts:
            try:sock.sendall(part)
            except (BrokenPipeError,ConnectionResetError):break
        response=sock.recv(65536)
        return int(response.split(b' ',2)[1]),response.decode('utf-8','replace')


async def main():
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    log=OUT.with_suffix('.server.log').open('w')
    process=subprocess.Popen([sys.executable,'-m','ds41f_mlx.ops','start','--profile',PROFILE,'--port',str(port)],stdout=log,stderr=subprocess.STDOUT)
    rt=RuntimeClient(f'http://127.0.0.1:{port}')
    try:
        for _ in range(3000):
            try:await asyncio.to_thread(rt.health);break
            except Exception:
                if process.poll() is not None:raise AssertionError('operator exited before health')
                await asyncio.sleep(.05)
        else:raise AssertionError('operator health')
        headers=f'POST /v1/sessions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\n'.encode()
        fixtures=[('content_length_excess',[headers+b'Content-Length: 1048577\r\n\r\n'],413),
                  ('chunked_cumulative_excess',[headers+b'Transfer-Encoding: chunked\r\n\r\n',
                    b'927c0\r\n'+b'x'*600000+b'\r\n',b'927c0\r\n'+b'x'*600000+b'\r\n0\r\n\r\n'],413),
                  ('compressed_body',[headers+b'Content-Encoding: gzip\r\nContent-Length: 2\r\n\r\n{}'],400),
                  ('overlong_content_length',[headers+b'Content-Length: '+b'9'*4500+b'\r\n\r\n'],400),
                  ('duplicate_sequence',[f'POST /v1/sessions/bad/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nX-DS41F-Request-Sequence: 1\r\nX-DS41F-Request-Sequence: 1\r\nContent-Length: 2\r\n\r\n{{}}'.encode()],400)]
        for name,parts,status in fixtures:
            code,body=await asyncio.to_thread(raw,port,parts);assert code==status,(name,body)
            result['cases'].append(dict(name=name,status=code))
        # No earlier denial may issue an identity or load a model.
        rec=await asyncio.to_thread(rt.create_session);assert rec['id'].endswith(f'{1:032x}')
        await asyncio.to_thread(rt.close_session,rec['id'])
        assert not (await asyncio.to_thread(rt.health))['model_ready']
        # Body slot denial is immediate; incomplete body expires at real policy.
        def incomplete():
            start=time.monotonic()
            code,body=raw(port,[headers+b'Content-Length: 20\r\n\r\n{'])
            return code,time.monotonic()-start
        body_task=asyncio.create_task(asyncio.to_thread(incomplete));await asyncio.sleep(.3)
        code,_=await asyncio.to_thread(raw,port,[headers+b'Content-Length: 2\r\n\r\n{}'])
        assert code==409
        assert (await asyncio.to_thread(rt.health))['status']=='alive'
        code,elapsed=await body_task;assert code==408 and 29 <= elapsed < 35
        result['cases'].append(dict(name='body_timeout_and_single_slot',status=code,seconds=elapsed,overlap_status=409))
        # Accepted empty connections are capped before headers/ASGI creation.
        sockets=[]
        try:
            for _ in range(8):
                sockets.append(socket.create_connection(('127.0.0.1',port),timeout=2))
            await asyncio.sleep(.1)
            excess=socket.create_connection(('127.0.0.1',port),timeout=2)
            excess.settimeout(2)
            assert await asyncio.to_thread(excess.recv,1)==b'';excess.close()
            start=time.monotonic()
            sockets[0].settimeout(35)
            assert await asyncio.to_thread(sockets[0].recv,1)==b''
            elapsed=time.monotonic()-start
            assert 28 <= elapsed < 35
            result['cases'].append(dict(name='idle_connection_cap_and_header_deadline',accepted=8,excess_closed=True,seconds=elapsed))
        finally:
            for s in sockets:s.close()
        await asyncio.sleep(.1)
        rec=await asyncio.to_thread(rt.create_session);assert rec['id'].endswith(f'{2:032x}')
        await asyncio.to_thread(rt.close_session,rec['id'])
        result['ingress_rejections_no_issuance_or_model_work']=True
    finally:
        process.terminate();await asyncio.to_thread(process.wait,120);log.close()
    # Same runtime h11/send/response-finally machinery with bounded synthetic SSE
    # comments. This deliberately forces pressure; not naturally sampled output.
    import uvicorn
    from ds41f_mlx.serving.local_h11 import LocalH11Protocol
    from ds41f_mlx.serving.mtp_public import LocalBoundary
    from ds41f_mlx.serving.server import InferenceStreamingResponse
    closed=asyncio.Event();produced=[];errors=[]
    async def chunks():
        try:
            for i in range(128):
                produced.append(i)
                yield b':'+b'x'*262144+b'\n\n'
        finally:closed.set()
    async def fixture(scope,receive,send):
        await InferenceStreamingResponse(chunks(),media_type='text/event-stream')(scope,receive,send)
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    app=LocalBoundary(fixture,authority=f'127.0.0.1:{port}')
    async def catch(scope,receive,send):
        try:await app(scope,receive,send)
        except BaseException as exc:errors.append(type(exc).__name__)
    server=uvicorn.Server(uvicorn.Config(catch,http=LocalH11Protocol,ws='none',proxy_headers=False,
        log_level='critical',lifespan='off',limit_concurrency=8,h11_max_incomplete_event_size=16384))
    task=asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:await asyncio.sleep(.01)
    peer=socket.socket();peer.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,4096)
    peer.connect(('127.0.0.1',port));peer.sendall(f'GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n'.encode())
    start=time.monotonic();samples=[]
    try:
        await asyncio.sleep(1);first=len(produced)
        await asyncio.sleep(1);second=len(produced)
        assert first==second and first<128
        await asyncio.wait_for(closed.wait(),35)
        elapsed=time.monotonic()-start
        assert 29 <= elapsed < 35 and len(produced)==second and errors
        result['cases'].append(dict(name='saturated_send_stall',fixture='synthetic bounded SSE comments; no model',
            timeout_seconds=elapsed,produced=second,production_plateau=True,iterator_closed=True,exception_types=errors))
    finally:
        peer.close();server.should_exit=True;await task
    result['status']='PASS'

try:asyncio.run(main())
except BaseException as exc:result.update(status='FAIL',error=repr(exc));raise
finally:OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
