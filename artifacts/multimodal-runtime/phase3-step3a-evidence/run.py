"""One supervised load and one official headless session. No inference POSTs here."""
import ctypes,datetime,hashlib,json,os,re,shutil,signal,socket,subprocess,time,urllib.request
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[2]
TEMP=Path(json.loads((OUT/'location.json').read_text())['temp']); G=1024**3
lib=ctypes.CDLL('/usr/lib/libproc.dylib');lib.proc_pid_rusage.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
runtime=harness=None; start=time.monotonic();receipt={'model_load_budget':1,'session_budget':1,'samples':[],'stop':None}
def save(n,x): (OUT/n).write_text(json.dumps(x,indent=2,default=str)+'\n')
def sample():
 b=(ctypes.c_uint64*64)();rc=-1 if runtime is None else lib.proc_pid_rusage(runtime.pid,4,ctypes.byref(b))
 vm=subprocess.check_output(['/usr/bin/vm_stat'],text=True)
 competitors=[]
 for p in psutil.process_iter(['pid','name','cmdline']):
  try:
   if p.pid in [os.getpid(),None if runtime is None else runtime.pid]:continue
   name=(p.info['name'] or '').lower()
   if 'python' in name or name in ('omlx','ds41f'):competitors.append(dict(pid=p.pid,cmd=p.info['cmdline']))
  except psutil.Error:pass
 return dict(elapsed=time.monotonic()-start,available=psutil.virtual_memory().available,pressure=int(subprocess.check_output(['/usr/sbin/sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True)),swap_used=psutil.swap_memory().used,swapouts=int(re.search(r'Swapouts:\s+(\d+)',vm)[1]),footprint=0 if rc else b[9],lifetime_footprint=0 if rc else b[30],competitors=competitors)
def retire(p):
 if p is None or p.poll() is not None:return
 os.killpg(p.pid,signal.SIGTERM)
 try:p.wait(timeout=90)
 except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def health():
 with urllib.request.urlopen('http://127.0.0.1:18083/health',timeout=2) as r:return json.load(r)
def main():
 global runtime,harness
 base=sample();receipt['initial']=base
 try:
  assert not base['competitors'] and base['pressure']==1 and base['available']>=455.226*G,'pre-launch resource refusal'
  with socket.socket() as s:s.bind(('127.0.0.1',18083))
  for n in ['ready','go']:(TEMP/n).unlink(missing_ok=True)
  env=dict(os.environ,DSH_HOME=str(TEMP/'home'),DSH_AGENTS_HOME=str(TEMP/'agents'),PHASE3_WORKSPACE=str(TEMP/'workspace'),PHASE3_EVIDENCE=str(OUT),DS41F_CHECKPOINT='/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017',PYTHONDONTWRITEBYTECODE='1')
  with (OUT/'runtime.log').open('w') as log:
   runtime=subprocess.Popen(['/tmp/ds41f-step3a/bin/python','-I','-B',str(OUT/'runtime.py')],env=env,cwd=TEMP/'runtime',stdout=log,stderr=log,start_new_session=True)
  receipt['runtime_pid']=runtime.pid;receipt['supervisor_pid']=os.getpid()
  loaded=False;session_start=None;last_progress=time.monotonic();last_size=0
  while runtime.poll() is None:
   s=sample();receipt['samples'].append(s)
   reason=None
   if s['competitors']:reason='competing worker'
   elif s['pressure']!=1:reason='memory pressure'
   elif s['swap_used']>base['swap_used'] or s['swapouts']>base['swapouts']:reason='swap increase'
   elif s['available']<=64*G:reason='host reserve'
   elif max(s['footprint'],s['lifetime_footprint'])>=425*G:reason='physical footprint stop'
   size=sum((OUT/n).stat().st_size if (OUT/n).exists() else 0 for n in ['runtime.log','wire.jsonl','lifecycle.jsonl'])
   if size!=last_size:last_size=size;last_progress=time.monotonic()
   if time.monotonic()-last_progress>=600:reason='10 minutes without progress'
   if s['elapsed']>=2100:reason='finite total wall budget'
   if not loaded and (TEMP/'ready').exists():
    receipt['preload']=s
    if s['available']<455.226*G:reason='pre-load reserve'
    if reason is None:(TEMP/'go').write_text('supervisor authorized one load\n');loaded=True
   if reason:raise RuntimeError(reason)
   if harness is None:
    if s['elapsed']>=1200:raise TimeoutError('startup budget')
    try:ready=health()
    except Exception:ready={}
    if ready.get('model_ready') and ready.get('status')=='ready':
     receipt['health_before']=ready
     henv=dict(env,DS41F_LOCAL_API_KEY='local-only-placeholder',NODE_OPTIONS='--import='+str(OUT/'observer.mjs'))
     with (OUT/'harness-events.jsonl').open('w') as stdout,(OUT/'harness-stderr.log').open('w') as stderr,(TEMP/'prompt.txt').open() as prompt:
      harness=subprocess.Popen(['node',str(Path.home()/'.local/share/deepseek-harness/node_modules/@deepseek-ai/dsh/lib/bin.js'),'--profile','headless','--json'],env=henv,cwd=TEMP/'workspace',stdin=prompt,stdout=stdout,stderr=stderr,start_new_session=True)
     receipt['harness_pid']=harness.pid;session_start=time.monotonic()
   else:
    if harness.poll() is not None:receipt['harness_exit']=harness.returncode;receipt['health_after']=health();break
    if time.monotonic()-session_start>900:raise TimeoutError('session wall budget')
    rows=[]
    if (OUT/'wire.jsonl').exists():
     for line in (OUT/'wire.jsonl').read_text().splitlines():
      try:rows.append(json.loads(line))
      except json.JSONDecodeError:pass
    if any(r.get('kind')=='budget_stop' for r in rows):raise RuntimeError('request budget fence')
    requests=[r for r in rows if r.get('kind')=='request' and r.get('path')=='/v1/messages']
    if requests:
     last=requests[-1];done=any(r.get('kind')=='native_message_stop' and r['request']==last['request'] for r in rows)
     elapsed=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(last['at'])).total_seconds()
     if not done and elapsed>180:raise TimeoutError('generation wall budget')
   save('resources.json',receipt);time.sleep(2)
  if runtime.poll() is not None:raise RuntimeError('Runtime exited before conversation end')
 except BaseException as e:receipt['stop']=repr(e)
 finally:
  retire(harness);retire(runtime)
  receipt['runtime_exit']=None if runtime is None else runtime.returncode
  receipt['harness_exit']=None if harness is None else harness.returncode
  receipt['elapsed_s']=time.monotonic()-start;receipt['postflight']=sample()
  save('resources.json',receipt)
  save('source-final.json',[dict(label=r['label'],unchanged=hashlib.sha256(Path(r['copy']).read_bytes()).hexdigest()==r['source_sha256']) for r in json.loads((OUT/'images.json').read_text())])
  with socket.socket() as s:free=s.connect_ex(('127.0.0.1',18083))!=0
  save('process-cleanup.json',dict(port_free=free,runtime_exited=runtime is None or runtime.poll() is not None,harness_exited=harness is None or harness.poll() is not None))
 print(json.dumps({k:v for k,v in receipt.items() if k!='samples'},indent=2))
if __name__=='__main__':main()
