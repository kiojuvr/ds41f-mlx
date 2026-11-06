"""Isolated installed Runtime; passive wire/preparation/settlement receipts only."""
import asyncio, gzip, hashlib, json, os, signal, sys, threading, time
from pathlib import Path
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
TEMP = Path('/tmp/ds41f-rc1-e2e')
def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, default=str)+'\n')
def row(name, value):
    with (OUT/name).open('a') as f: f.write(json.dumps(value, default=str)+'\n')

def main():
    import ds41f_mlx
    from ds41f_mlx.execution_identity import inspect_payload
    from ds41f_mlx.mtp_identity import config, inspect
    from dataclasses import replace
    authority, host = inspect_payload()
    cfg = replace(config(port=18097, profile='mtp-serving-v1'), kv_root=TEMP/'kv')
    cfg.apply_environment()
    admitted = inspect(cfg, checkpoint=True)
    save('installed-admission.json', dict(authority=authority, host=host, runtime=admitted, python=sys.version))
    installed = Path(ds41f_mlx.__file__).parent
    source = {str(p.relative_to(ROOT/'ds41f_mlx')):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT/'ds41f_mlx').rglob('*.py')}
    source_match = {p: hashlib.sha256((installed/p).read_bytes()).hexdigest()==h for p,h in source.items()}
    save('source-comparison.json', source_match)
    save('source-match.json', source)
    import mlx.core as mx
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    from ds41f_mlx.serving.server import create_app
    import ds41f_mlx.serving.server as serving
    backend = ProductionMTPBackend(runtime_config=cfg)
    backend.dependency_identity = admitted['identity_sha256']
    backend.glm_profile = 'native'
    original = serving.prepare_request
    def observed(protocol, body, **kwargs):
        p = original(protocol, body, **kwargs)
        roots=[]
        if backend.scheduler is not None:
            for pair in list(backend.scheduler.checkpoints._payloads.values()):
                roots.append(dict(tokens=list(pair.tokens), images=pair.images))
        row('prepared.jsonl', dict(body_sha256=hashlib.sha256(body).hexdigest(), tokens=p.token_ids,
            images=[] if p.multimodal is None else p.multimodal.identities(), available_pairs=roots))
        return p
    serving.prepare_request = observed
    app = create_app(profile='mtp-serving-v1', runtime_config=cfg, backend=backend)
    async def load():
        await backend._call(backend.load)
        save('loaded.json', dict(active=mx.get_active_memory(), cache=mx.get_cache_memory(), peak=mx.get_peak_memory()))
    app.router.add_event_handler('startup', load)
    class Wire:
        async def __call__(self, scope, receive, send):
            if scope['type']!='http': return await app(scope,receive,send)
            rid=str(time.time_ns()); chunks=[]; response=[]; status=None
            async def recv():
                event=await receive()
                if event['type']=='http.request': chunks.append(event.get('body',b''))
                return event
            async def emit(event):
                nonlocal status
                if event['type']=='http.response.start': status=event['status']
                if event['type']=='http.response.body': response.append(event.get('body',b''))
                await send(event)
            try: await app(scope,recv,emit)
            finally:
                if scope['path'] in ('/v1/messages','/v1/files') or scope['path'].startswith('/v1/files/'):
                    raw=b''.join(chunks); received=b''.join(response)
                    row('wire.jsonl', dict(id=rid,path=scope['path'],method=scope['method'],status=status,
                        body_sha256=hashlib.sha256(raw).hexdigest(),
                        body=json.loads(raw) if scope['path']=='/v1/messages' and raw else None,
                        response=received.decode(errors='replace')))
    stop=threading.Event()
    def flush():
        for trace in list(backend.serving_traces):
            p=OUT/('trace-'+trace['response_id']+'.json.gz')
            tmp=p.with_suffix('.tmp'); tmp.write_bytes(gzip.compress(json.dumps(trace,default=str).encode())); tmp.replace(p)
    def watch():
        import psutil
        initial=psutil.swap_memory().used
        while not stop.wait(2):
            flush(); mem=psutil.virtual_memory();swap=psutil.swap_memory()
            row('resources.jsonl',dict(at=time.time(),available=mem.available,swap=swap.used,active=mx.get_active_memory(),cache=mx.get_cache_memory()))
            if mem.available<64*1024**3 or swap.used>initial or time.monotonic()-started>2700:
                row('resource-stop.jsonl',dict(reason='memory/swap/time budget')); server.should_exit=True
    import uvicorn
    from ds41f_mlx.serving.local_h11 import LocalH11Protocol
    server=uvicorn.Server(uvicorn.Config(Wire(),host='127.0.0.1',port=18097,log_level='info',workers=1,
        proxy_headers=False,ws='none',loop='asyncio',http=LocalH11Protocol))
    started=time.monotonic(); thread=threading.Thread(target=watch,daemon=True);thread.start()
    try: server.run()
    finally:
        stop.set();thread.join();flush()
        save('runtime-exit.json',dict(queues_empty=not backend._deliveries and not backend._settled,
            elapsed=time.monotonic()-started,active=mx.get_active_memory(),cache=mx.get_cache_memory()))
if __name__=='__main__': main()
