#!/usr/bin/env python3
"""Small isolated installed-native Processor token-boundary qualification.

No Production changes, external tools, budget overrides or fabricated cache hits.
"""
import argparse
import asyncio
import base64
import ctypes
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import os
import re
import signal
import subprocess
import threading
import time
from unittest.mock import patch

CASES = [('control-1014',(1024,2048),1014,'PNG'),
         ('incident-1016',(1582,1084),1016,'JPEG'),
         ('boundary-1024',(3024,588),1024,'PNG')]
GiB = 1024**3


def fixture_image(size, fmt='PNG'):
    from PIL import Image, ImageDraw
    with Image.new('RGB',size,'white') as image:
        w,h = size
        draw = ImageDraw.Draw(image)
        draw.rectangle((0,0,w//2,h//2),fill=(230,40,20))
        draw.rectangle((w//2,h//2,w-1,h-1),fill=(20,80,220))
        draw.line((0,h-1,w-1,0),fill=(20,180,40),width=7)
        stream = BytesIO()
        image.save(stream,format=fmt)
        return stream.getvalue()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        (a.output/name).write_text(json.dumps(value,indent=2)+'\n')
    def cmd(*args):
        return subprocess.check_output(args,text=True)
    save('host-before.json',dict(vm=cmd('vm_stat'),swap=cmd('sysctl','vm.swapusage'),
        pressure=cmd('memory_pressure','-Q'),hardware=cmd('sysctl','hw.memsize','machdep.cpu.brand_string')))
    from ds41f_mlx.mtp_identity import config, inspect
    from ds41f_mlx.execution_identity import inspect_payload
    cfg = replace(config(profile='mtp-serving-v1'),kv_root=a.output/'unused-kv')
    identity = inspect(cfg)
    authority,_ = inspect_payload()
    import ds41f_mlx
    installed = Path(ds41f_mlx.__file__).parent
    sources = {}
    for src in (a.source/'ds41f_mlx').rglob('*.py'):
        rel = src.relative_to(a.source/'ds41f_mlx')
        data = (installed/rel).read_bytes()
        assert data == src.read_bytes(),str(rel)
        sources[str(rel)] = sha256(data).hexdigest()
    save('identity.json',identity)
    save('sources.json',sources)
    save('provenance.json',authority['provenance'])
    cfg.apply_environment()
    import mlx.core as mx
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    from ds41f_mlx.serving.server import prepare_request,load_v41_tokenizer
    from ds41f_mlx.serving.image_files import ImageFiles
    from ds41f_mlx.serving.paired_checkpoint import pair_bytes
    from ds41f_mlx.model_execution import processing
    from ds41f_mlx.observability import TRACE_FIELDS
    backend = ProductionMTPBackend(runtime_config=cfg)
    backend.dependency_identity,backend.glm_profile = identity['identity_sha256'],identity['glm_profile']
    tokenizer = load_v41_tokenizer(cfg.recipe_path)
    store = ImageFiles()
    images = [fixture_image(size,fmt) for _,size,_,fmt in CASES]
    save('workload.json',[dict(case=label,size=list(size),tokens=tokens,format=fmt,
        encoded_bytes=len(data),sha256=sha256(data).hexdigest())
        for (label,size,tokens,fmt),data in zip(CASES,images)])
    # Existing HTTP multipart implementation, no second upload/parser authority.
    import httpx
    keys = []
    for (label,_,_,fmt),data in zip(CASES,images):
        mime = 'image/jpeg' if fmt == 'JPEG' else 'image/png'
        r = httpx.Request('POST','http://fixture/v1/files',files={'file':(label,data,mime)},
            data={'expires_after[anchor]':'created_at','expires_after[seconds]':'604800'})
        keys.append(store.upload(r.read(),r.headers['content-type'])['id'])
    lib = ctypes.CDLL('/usr/lib/libproc.dylib')
    unsafe,stop,lock = [],threading.Event(),threading.Lock()
    case = 'startup'
    def sample(event,**extra):
        buf = ctypes.create_string_buffer(1024)
        assert lib.proc_pid_rusage(os.getpid(),4,ctypes.byref(buf)) == 0
        row = dict(t=time.monotonic(),case=case,event=event,
            footprint=ctypes.c_uint64.from_buffer(buf,72).value,
            lifetime_max_footprint=ctypes.c_uint64.from_buffer(buf,240).value,
            mlx_active=mx.get_active_memory(),mlx_cache=mx.get_cache_memory(),mlx_peak=mx.get_peak_memory(),**extra)
        if backend.scheduler is not None:
            row.update(roots=len(backend.scheduler.checkpoints._payloads),
                root_logical_bytes=sum(pair_bytes(c) for c in backend.scheduler.checkpoints._payloads.values()))
        with lock,(a.output/'samples.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        if max(row['footprint'],row['mlx_active']+row['mlx_cache']) > 360*GiB:
            unsafe.append('360 GiB process/allocator stop')
        return row
    def check():
        assert not unsafe,unsafe
    def watch():
        try:
            while not stop.wait(2):
                vm,swap,pressure = cmd('vm_stat'),cmd('sysctl','vm.swapusage'),cmd('memory_pressure','-Q')
                reserve = sum(int(re.search(n+r':\s+(\d+)',vm)[1])*16384
                    for n in ('Pages free','Pages inactive','Pages speculative'))
                with (a.output/'host.jsonl').open('a') as f:
                    f.write(json.dumps(dict(t=time.monotonic(),vm=vm,swap=swap,pressure=pressure,reserve_proxy=reserve))+'\n')
                if reserve < 100*GiB or int(re.search(r'free percentage: (\d+)%',pressure)[1]) < 25:
                    unsafe.append('host reserve/pressure stop')
                if re.search(r'used = (?!0\.00M)\S+',swap):
                    unsafe.append('nonzero swap')
                sample('watch')
        except BaseException:
            unsafe.append('observer failed')
    watcher = threading.Thread(target=watch,daemon=True)
    watcher.start()
    history,traces,patches,embeds = [],[],[],[]
    def prepare(messages):
        raw = json.dumps(dict(model=cfg.model_id,max_tokens=16,temperature=0,
            thinking={'type':'disabled'},stream=False,messages=messages)).encode()
        assert len(raw) < 8*1024**2
        return prepare_request('messages',raw,tokenizer=tokenizer,recipe_path=cfg.recipe_path,
            checkpoint=cfg.checkpoint_path,ordinary=True,image_files=store)
    original_pack = processing.image_patch_array
    def packed(image,config):
        result = original_pack(image,config)
        values,h,w,kinds = result
        patches.append(dict(case=case,patch_shape=list(values.shape),grid=[h,w],tokens=len(kinds)))
        return result
    async def run(label,i=None):
        nonlocal case
        case = label
        check()
        content = [dict(type='text',text='Reply only OK.')]
        if i is not None:
            content.insert(0,dict(type='image',source=dict(type='file',file_id=keys[i])))
        history.append(dict(role='user',content=content))
        before_p,before_e = len(patches),len(embeds)
        mx.reset_peak_memory()
        sample('before')
        prepared = prepare(history)
        assert len(prepared.token_ids) < 6000
        assert [s.length for s in prepared.multimodal.spans] == [c[2] for c in CASES[:sum(m['role']=='user' and len(m['content'])>1 for m in history)]]
        await backend._call(backend._enqueue,prepared,tokenizer,label)
        output = None
        for _ in range(64):
            check()
            result = await backend._call(backend.scheduler.step)
            output = next((x for x in result.outputs if x.finished),None)
            if output is not None:
                break
        assert output is not None and not output.error,'native request failed'
        trace = output.serving_trace
        C = trace['canonical_frontier']
        assert set(trace['target_offsets']+trace['dspark_offsets']) == {C}
        assert trace['cache_published'] and trace['prediction_retired'] and trace['queue_empty']
        assert trace['prompt_replay'] == trace['full_cache_repack'] == 0
        assert trace['vision_encoded_count'] == int(i is not None)
        assert len(patches)-before_p == len(embeds)-before_e == int(i is not None)
        if i is not None:
            assert patches[-1]['tokens'] == CASES[i][2] == embeds[-1]['output_shape'][1]
        else:
            assert trace['cached_tokens'] == traces[-1]['canonical_frontier']
        assert not prepared.multimodal._sources
        receipt = {k:v for k,v in trace.items() if k in (*TRACE_FIELDS,'vision_encoded_count','cache_published')
                   and isinstance(v,(int,float,bool))}
        receipt.update(case=label,aligned=True,image_lengths=[s.length for s in prepared.multimodal.spans])
        traces.append(receipt)
        save('requests.json',traces)
        history.append(dict(role='assistant',content=trace['response']['content']))
        sample('settled')
        check()
    negatives = []
    def reject(label,messages,reason):
        before = len(backend.scheduler.checkpoints._payloads)
        try:
            prepare(messages)
        except ValueError as exc:
            assert reason in str(exc)
            negatives.append(label)
        except Exception as exc:
            from ds41f_mlx.serving.server import RequestError
            assert isinstance(exc,RequestError) and reason in str(exc)
            negatives.append(label)
        else:
            raise AssertionError('invalid input accepted')
        assert not backend.scheduler.requests and not backend.fatal_error
        assert len(backend.scheduler.checkpoints._payloads) == before
    async def qualify():
        await backend._call(backend.load)
        sample('loaded')
        check()
        native_type = type(backend._model)
        original_embed = native_type.get_input_embeddings
        def embedded(model,*args,**kwargs):
            value = original_embed(model,*args,**kwargs)
            embeds.append(dict(case=case,output_shape=list(value.inputs_embeds.shape)))
            return value
        with patch.object(processing,'image_patch_array',packed),patch.object(native_type,'get_input_embeddings',embedded):
            for i,(label,_,_,_) in enumerate(CASES):
                await run(label,i)
                await run(label+'-cached')
            bad = dict(type='image',source=dict(type='base64',media_type='image/png',data=base64.b64encode(b'invalid').decode()))
            reject('malformed',[dict(role='user',content=[bad,dict(type='text',text='Reply only OK.')])],'invalid image')
            ref = dict(type='image',source=dict(type='file',file_id=keys[-1]))
            reject('fresh-patch-budget',[dict(role='user',content=[ref]*5+[dict(type='text',text='Reply only OK.')])],'patch budget')
            # Negative integrity check on a real immutable native pair; no arrays
            # or successful prefix are fabricated or published to force a hit.
            saved = max(backend.scheduler.checkpoints._payloads.values(),key=lambda c:len(c.tokens))
            changed = [dict(s) for s in saved.images]
            changed[0]['sha256'] = '0'*64
            try:
                await backend._call(saved.restore,backend.scheduler.checkpoints.identity,
                    list(saved.tokens)+[1,2],mx,changed)
            except ValueError:
                negatives.append('paired-image-identity')
            else:
                raise AssertionError('invalid paired identity accepted')
            await run('boundary-1024-repeat-1')
            await run('boundary-1024-repeat-2')
        save('layout-observations.json',dict(patches=patches,embeddings=embeds,negative_rejections=negatives))
    try:
        asyncio.run(qualify())
        save('result.json',dict(status='COMPLETED',unsafe=unsafe))
    except BaseException as exc:
        save('result.json',dict(status='STOPPED',exception_type=type(exc).__name__,unsafe=unsafe))
        raise
    finally:
        stop.set()
        watcher.join()
        scheduler = backend.scheduler
        asyncio.run(backend.shutdown_serving())
        assert scheduler is None or (not scheduler.requests and not scheduler.checkpoints._payloads)
        store.clear()
        sample('closed',requests=0)
        save('host-after.json',dict(vm=cmd('vm_stat'),swap=cmd('sysctl','vm.swapusage'),pressure=cmd('memory_pressure','-Q')))


if __name__ == '__main__':
    def expired(*_):
        raise TimeoutError('finite 1800 second budget')
    signal.signal(signal.SIGALRM,expired)
    signal.alarm(1800)
    try:
        main()
    finally:
        signal.alarm(0)
