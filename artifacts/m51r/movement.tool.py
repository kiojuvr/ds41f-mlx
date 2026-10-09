"""Separate M50R copy/sync assay: observable API/CPU bridges + external Metal trace.

Diagnostic wrappers call original functions exactly once, retain no array values,
add no eval/sync, and run on ALL calling threads. Never use this process for rates.
A loaded/warmed process waits for an externally attached Metal System Trace.
"""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter
from contextvars import ContextVar
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import traceback


class Observer:
    def __init__(self, mx, np):
        self.mx,self.np=mx,np
        self.array_type=mx.array
        self.events=[];self.restore=[]
        self.monitor_id=None
        self.native_pending={}
        self.phase=ContextVar('m50r-observation-phase',default='worker/outside response phase')

    def describe(self,x):
        if isinstance(x,self.array_type):
            return dict(kind='mlx-array metadata only',shape=list(x.shape),dtype=str(x.dtype))
        if isinstance(x,self.np.ndarray):
            return dict(kind='cpu-ndarray',shape=list(x.shape),dtype=str(x.dtype),nbytes=x.nbytes,
                        owns_data=bool(x.flags.owndata),base_type=type(x.base).__name__,strides=list(x.strides))
        if isinstance(x,(list,tuple)):
            return [self.describe(v) for v in x]
        if isinstance(x,dict):
            return {str(k):self.describe(v) for k,v in x.items()}
        if isinstance(x,(bytes,bytearray,memoryview)):
            return dict(kind='CPU bytes',nbytes=len(x))
        if isinstance(x,(str,int,float,bool)) or x is None:
            return x
        if type(x).__name__ in ('Stream','ThreadLocalStream'):
            return dict(kind=type(x).__name__,repr=str(x))
        return dict(kind=type(x).__name__)

    def wrap(self,obj,name,label,scope=None,condition=None,detail=None):
        original=getattr(obj,name)
        def observed(*args,**kwargs):
            if condition and not condition(args,kwargs):
                return original(*args,**kwargs)
            token=self.phase.set(scope) if scope else None
            row=dict(label=label,phase=self.phase.get(),thread=threading.current_thread().name,
                     tid=threading.get_native_id(),start_ns=time.perf_counter_ns(),wall_ns=time.time_ns())
            row['args']=self.describe(args if obj is self.mx else args[1:] if isinstance(obj,type) else args)
            row['kwargs']=self.describe(kwargs)
            if detail: row.update(detail(args,kwargs))
            try:
                value=original(*args,**kwargs)
                row['result']=self.describe(value)
                return value
            except BaseException as e:
                row['exception']=repr(e);raise
            finally:
                row['end_ns']=time.perf_counter_ns()
                self.events.append(row)
                if token is not None: self.phase.reset(token)
        setattr(obj,name,observed)
        self.restore.append((obj,name,original))

    def native_event(self,event,code,offset,fn,arg0):
        name=getattr(fn,'__name__','')
        if fn is self.array_type:
            label='native mlx.array constructor'
        elif name in ('item','tolist') and isinstance(arg0,self.array_type):
            label='native array.'+name
        else:
            return
        key=(threading.get_native_id(),id(code),offset,label)
        if event=='call':
            self.native_pending[key]=dict(label=label,phase=self.phase.get(),
                thread=threading.current_thread().name,tid=threading.get_native_id(),
                start_ns=time.perf_counter_ns(),wall_ns=time.time_ns(),
                caller=code.co_filename+':'+code.co_name,args=self.describe(arg0),
                input_payload_nbytes=arg0.nbytes if isinstance(arg0,self.np.ndarray) else None)
        else:
            row=self.native_pending.pop(key,None)
            if row is not None:
                row.update(end_ns=time.perf_counter_ns(),native_return=event)
                self.events.append(row)

    def install(self):
        # PEP 669 observes nanobind CALL/C_RETURN where sys.setprofile does not.
        # It also covers all threads, without proxying/replacing mx.array type.
        assert hasattr(sys,'monitoring'), 'CPython 3.13 monitoring required'
        self.monitor_id=next(i for i in (5,4,3) if sys.monitoring.get_tool(i) is None)
        sys.monitoring.use_tool_id(self.monitor_id,'m50r-native-bridge-observer')
        ev=sys.monitoring.events
        for event,label in ((ev.CALL,'call'),(ev.C_RETURN,'return'),(ev.C_RAISE,'raise')):
            sys.monitoring.register_callback(self.monitor_id,event,
                lambda code,offset,fn,arg0,label=label:self.native_event(label,code,offset,fn,arg0))
        sys.monitoring.set_events(self.monitor_id,ev.CALL|ev.C_RETURN|ev.C_RAISE)
        import omlx.patches.deepseek_v41.storage as storage
        from omlx.patches.deepseek_v41.engram import NgramHash
        from omlx.patches.deepseek_v41.cache import DeepseekV41Cache
        from omlx.patches.deepseek_v41.mtp import DSparkMixin
        from ds41f_mlx.serving.internal_mtp import InternalMTPQualificationBackend
        for name in ('eval','async_eval','synchronize','concatenate','contiguous'):
            self.wrap(self.mx,name,'mlx.'+name)
        self.wrap(self.np,'asarray','mlx-to-numpy export/materialization',
                  condition=lambda a,k: bool(a) and isinstance(a[0],self.array_type))
        self.wrap(storage,'decode_array','CPU-to-MLX decode/import site')
        self.wrap(storage.TensorFile,'read','CPU selected-row copy/read',
                  detail=lambda a,k: dict(file=str(a[0]._file.name),rows=self.describe(a[2] if len(a)>2 else k.get('rows'))))
        self.wrap(storage.DiskEngramEmbedding,'_read_rows','Engram worker rows',scope='Engram CPU read worker')
        for name in ('submit','drain'):
            self.wrap(storage.EngramPrefetch,name,'Engram prefetch '+name)
        self.wrap(NgramHash,'__call__','CPU integer Engram hashing')
        self.wrap(DeepseekV41Cache,'size','native offset scalar materialization')
        self.wrap(DSparkMixin,'dspark_forward','native parallel proposal',scope='proposal enqueue')
        self.wrap(DSparkMixin,'mtp_partial_rollback','native rollback',scope='prefix rollback')
        for name,phase in (('_start','prefill/bootstrap'),('_next','protected next'),('_settle','settlement'),('_retire','retirement')):
            self.wrap(InternalMTPQualificationBackend,name,phase,scope=phase)
        self.wrap(os,'pread','CPU pread',detail=lambda a,k:dict(requested_bytes=a[1],offset=a[2]))

    def close(self):
        if self.monitor_id is not None:
            sys.monitoring.set_events(self.monitor_id,0)
            sys.monitoring.free_tool_id(self.monitor_id)
            self.monitor_id=None
        for obj,name,original in reversed(self.restore): setattr(obj,name,original)
        self.restore.clear()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=Path('artifacts/m50r/movement.json'))
    ap.add_argument('--marker',type=Path,default=Path('artifacts/m50r/movement-ready.json'))
    ap.add_argument('--baseline',type=Path,default=Path('artifacts/m50r/performance.json'))
    args=ap.parse_args()
    from ds41f_mlx.mtp_identity import config,inspect
    cfg=config();identity=inspect(cfg)
    import mlx.core as mx
    import numpy as np
    import httpx
    import uvicorn
    from ds41f_mlx.serving.mtp_public import LocalMTPBackend
    from ds41f_mlx.serving.server import create_app
    from ds41f_mlx.serving.local_h11 import LocalH11Protocol
    baseline=json.loads(args.baseline.read_text())
    assert identity['identity_sha256']==baseline['identity']['identity_sha256']
    row=baseline['rows'][1]
    body=bytes.fromhex(row['request_hex'])
    backend=LocalMTPBackend(runtime_config=cfg)
    backend.dependency_identity=identity['identity_sha256']
    observer=Observer(mx,np)
    out=dict(schema='ds41f.m50r.movement.v1',status='RUNNING',pid=os.getpid(),
             identity_sha256=identity['identity_sha256'],request_sha256=row['request_sha256'],
             tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             instrumentation='original-call-once bridge wrappers + CPython monitoring of native constructors/item/tolist + attached Metal System Trace; NOT rate evidence')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.tool.py').write_bytes(Path(__file__).read_bytes())
    def save(): args.output.write_text(json.dumps(out,indent=2)+'\n')
    save()

    async def run():
        sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen(8)
        actual=replace(cfg,port=sock.getsockname()[1])
        server=uvicorn.Server(uvicorn.Config(create_app(backend=backend,runtime_config=actual,profile='mtp-singleton-v1'),
            lifespan='off',log_level='warning',workers=1,proxy_headers=False,ws='none',loop='asyncio',
            limit_concurrency=8,backlog=8,timeout_keep_alive=5,h11_max_incomplete_event_size=16384,http=LocalH11Protocol))
        task=asyncio.create_task(server.serve(sockets=[sock]))
        try:
            while not server.started: await asyncio.sleep(.01)
            await backend._call(backend.load)
            async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{actual.port}',timeout=600) as client:
                for sample in range(2):
                    if sample==1:
                        args.marker.write_text(json.dumps(dict(pid=os.getpid(),wall_ns=time.time_ns(),ready='loaded and warm; waiting for '+str(args.marker)+'.go'))+'\n')
                        deadline=time.monotonic()+180
                        while not Path(str(args.marker)+'.go').exists():
                            if time.monotonic()>deadline: raise TimeoutError('external trace go marker absent')
                            await asyncio.sleep(.1)
                        observer.install()
                        out['observed_begin_wall_ns']=time.time_ns()
                    response=await client.post('/v1/sessions',content=b'{}',headers={'Content-Type':'application/json'})
                    response.raise_for_status();sid=response.json()['id'];rec=backend.sessions[sid]
                    response=await client.post(f'/v1/sessions/{sid}/chat/completions',content=body,
                        headers={'Content-Type':'application/json','X-DS41F-Request-Sequence':'1'})
                    response.raise_for_status()
                    assert rec.last_turn['canonical_generated']==row['trace']['canonical_generated']
                    assert rec.last_turn['canonical_frontier']==row['trace']['canonical_frontier']
                    if sample==1: out['turn']=dict(rec.last_turn)
                    response=await client.delete(f'/v1/sessions/{sid}',headers={'Content-Type':'application/json'})
                    response.raise_for_status()
                    if sample==1:
                        out['observed_end_wall_ns']=time.time_ns()
                        out['retired']=sid not in backend.sessions
                        observer.close()
            out['status']='CAPTURED'
        finally:
            observer.close();server.should_exit=True;await task;sock.close();backend.close()
    try:
        asyncio.run(run())
    except BaseException:
        out['status']='ERROR';out['error']=traceback.format_exc();raise
    finally:
        out['events']=observer.events
        out['region_counts']=dict(Counter((e['label'] for e in observer.events)))
        out['threads']=dict(Counter((e['thread'] for e in observer.events)))
        save()


if __name__=='__main__': main()
