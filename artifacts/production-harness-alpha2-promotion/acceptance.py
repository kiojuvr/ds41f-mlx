"""Bounded public installed Production acceptance; no implementation hooks."""
import json, os, pathlib, subprocess, time, threading, re, ctypes, traceback
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
O=pathlib.Path(__file__).resolve().parent
ENV=pathlib.Path('/Volumes/SDXC-512/ds41f-1017')
BASE='http://127.0.0.1:18087'; CONTROL='http://127.0.0.1:18441'
def save(n,v): (O/n).write_text(json.dumps(v,indent=2)+'\n')
def call(path,body=None,method=None):
    url=path if path.startswith('http') else BASE+path
    try:
        with urlopen(Request(url,data=None if body is None else json.dumps(body).encode(),method=method,headers={'Content-Type':'application/json'}),timeout=180) as r:
            return r.status,json.load(r)
    except HTTPError as e:return e.code,json.load(e)
def idle():
    deadline=time.monotonic()+30
    while True:
        s=call(CONTROL+'/ds41f/status')[1]
        if s['state']=='READY':return s
        assert time.monotonic()<deadline,s
        time.sleep(.2)
def settled(label):
    s=idle(); last=s['last_request']; m=last['metrics']
    assert last['native_started'] and last['canonical_settled'] and last['cache_published'],last
    assert m['aligned'] and m['prompt_replay']==m['full_cache_repack']==0,m
    result['messages'].append(dict(case=label,**last));save('acceptance.json',result)
    return m
result=dict(messages=[],forced_stop=False); stop=threading.Event(); unsafe=[]
for url in (BASE+'/health',CONTROL+'/ds41f/status'):
    try: call(url)
    except URLError:pass
    else:raise RuntimeError('existing listener; refusing startup')
env=dict(os.environ,DS41F_CHECKPOINT='/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017')
log=(O/'runtime.log').open('wb')
p=subprocess.Popen([str(ENV/'bin/python'),'-I','-m','ds41f_mlx.ops','start','--profile','mtp-serving-v1','--host','127.0.0.1','--port','18087'],env=env,cwd='/tmp',stdout=log,stderr=subprocess.STDOUT)
result['pid']=p.pid
lib=ctypes.CDLL('/usr/lib/libproc.dylib')
def observe():
    try:
        while not stop.wait(3):
            vm=subprocess.check_output(['vm_stat'],text=True);swap=subprocess.check_output(['sysctl','vm.swapusage'],text=True);pressure=subprocess.check_output(['memory_pressure','-Q'],text=True)
            reserve=sum(int(re.search(n+r':\s+(\d+)',vm)[1])*16384 for n in ('Pages free','Pages inactive','Pages speculative'))
            buf=ctypes.create_string_buffer(1024)
            if lib.proc_pid_rusage(p.pid,4,ctypes.byref(buf))!=0:
                if p.poll() is not None:return
                raise RuntimeError('resource observation failed')
            footprint=ctypes.c_uint64.from_buffer(buf,72).value
            row=dict(time=time.time(),footprint=footprint,reserve_proxy=reserve,swap=swap,pressure=pressure)
            with (O/'resources.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if footprint>360*1024**3 or reserve<100*1024**3 or re.search(r'used = (?!0\.00M)\S+',swap) or int(re.search(r'free percentage: (\d+)%',pressure)[1])<25:unsafe.append(row)
    except BaseException:unsafe.append(traceback.format_exc())
t=threading.Thread(target=observe,daemon=True);t.start()
try:
    deadline=time.monotonic()+420
    while True:
        assert p.poll() is None and not unsafe
        try:
            code,h=call('/health')
            if code==200 and h['model_ready']:break
        except URLError:pass
        assert time.monotonic()<deadline,'READY deadline'
        time.sleep(1)
    result['initial_health']=h;save('acceptance.json',result)
    ordinary=subprocess.run([str(ENV/'bin/python'),'-I','-m','ds41f_mlx.ops','accept','--profile','mtp-serving-v1','--runtime-url',BASE,'--output',str(O/'ordinary-acceptance.json')],env=env,cwd='/tmp',stdout=(O/'ordinary.log').open('wb'),stderr=subprocess.STDOUT,timeout=360)
    assert ordinary.returncode==0,ordinary.returncode
    assert not unsafe
    # A harmless synthetic read_image tool: no arbitrary file or external effects.
    tools=[dict(name='read_image',description='Read the synthetic red circle test image.',input_schema={'type':'object','properties':{},'additionalProperties':False},defer_loading=True)]
    history=[dict(role='user',content='Prepare the synthetic fixture.'),dict(role='system',content=[dict(type='tool_addition',tool=dict(type='tool_reference',name='read_image'))]),dict(role='user',content='Call read_image exactly once with no arguments.')]
    body=dict(model='deepseek-flash',max_tokens=96,temperature=0,thinking={'type':'disabled'},tools=tools,messages=history,tool_choice={'type':'tool','name':'read_image'})
    code,response=call('/v1/messages',body);assert code==200,response
    save('all-deferred-tool.json',response)
    calls=[b for b in response['content'] if b['type']=='tool_use'];assert len(calls)==1 and calls[0]['name']=='read_image' and calls[0]['input']=={},response
    m0=settled('all-deferred-json-tool')
    # Generate/upload a small known PNG through actual Files multipart transport.
    from PIL import Image,ImageDraw
    import io,httpx,hashlib
    image=Image.new('RGB',(384,384),'white');ImageDraw.Draw(image).ellipse((64,64,320,320),fill='red');buf=io.BytesIO();image.save(buf,format='PNG');raw=buf.getvalue()
    result['image_sha256']=hashlib.sha256(raw).hexdigest()
    req=httpx.Request('POST',BASE+'/v1/files',files={'file':('synthetic-circle.png',raw,'image/png')})
    with urlopen(Request(BASE+'/v1/files',data=req.read(),headers={'Content-Type':req.headers['content-type']}),timeout=30) as r:
        assert r.status==200;file=json.load(r)
    key=file['id'];result['uploaded_file']=file
    code,listing=call('/v1/files');assert code==200 and any(f['id']==key for f in listing['data']),listing
    result['files_listing']=listing
    history.extend([dict(role='assistant',content=response['content']),dict(role='user',content=[dict(type='tool_result',tool_use_id=calls[0]['id'],content=[dict(type='image',source=dict(type='file',file_id=key)),dict(type='text',text='Describe the color and shape briefly. Do not call tools.')])])])
    body.update(messages=history,tool_choice={'type':'none'},stream=True)
    with urlopen(Request(BASE+'/v1/messages',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=180) as r:
        assert r.status==200;wire=r.read().decode()
    (O/'messages.sse').write_text(wire)
    events=[json.loads(line[6:]) for line in wire.splitlines() if line.startswith('data: ')]
    assert events[0]['type']=='message_start' and events[-1]['type']=='message_stop'
    text=''.join(e['delta']['text'] for e in events if e['type']=='content_block_delta' and e['delta']['type']=='text_delta')
    assert 'red' in text.lower() and 'circle' in text.lower(),text
    m1=settled('all-deferred-exact-id-files-sse');assert m1['cached_tokens']>0
    result['exact_tool_id']=calls[0]['id'];result['image_description']=text
    history.extend([dict(role='assistant',content=[dict(type='text',text=text)]),dict(role='user',content='Reply only OK.')])
    body.update(messages=history,stream=False,max_tokens=16)
    code,response=call('/v1/messages',body);assert code==200,response
    save('cached-continuation.json',response)
    m2=settled('all-deferred-cached-image-json');assert m2['cached_tokens']==m1['canonical_frontier'],(m1,m2)
    code,error=call('/v1/messages',dict(body,temperature=-1));assert code in (400,422),error
    s=idle();failure=s['failures'][-1];assert not failure['native_started'] and not failure['cache_published'] and not failure['runtime_fatal'],failure
    result['refusal']=failure
    assert call('/v1/files/'+key,method='DELETE')[0]==200
    assert not any(f['id']==key for f in call('/v1/files')[1]['data'])
    result['files_deleted']=True
    h=call('/health')[1];assert h['status']=='ready' and h['fatal_error'] is None and h['active_requests']==h['queued_requests']==0,h
    assert not unsafe
    result.update(final_health=h,status='PASS')
except BaseException:
    result.update(status='FAIL',failure=traceback.format_exc());raise
finally:
    try:
        if p.poll() is None:
            code,receipt=call(CONTROL+'/ds41f/control/shutdown',method='POST');assert code==202,(code,receipt)
            result['shutdown_response']=receipt
        result['exit_code']=p.wait(timeout=240)
    except BaseException:
        result['shutdown_error']=traceback.format_exc() # Never force-kill native work.
    stop.set();t.join(timeout=10);log.close();result['unsafe']=unsafe;save('acceptance.json',result)
assert result['exit_code']==0,result
