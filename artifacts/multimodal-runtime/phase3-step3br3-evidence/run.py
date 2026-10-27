"""One supervised load; sequential official headless invocations, no inference POST here.
Reuses Step 3A resource sampling/retirement verbatim through its retained module.
"""
import datetime,hashlib,importlib.util,json,os,signal,socket,subprocess,time,urllib.request
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
spec=importlib.util.spec_from_file_location('step3a_supervisor',ROOT/'artifacts/multimodal-runtime/phase3-step3a-evidence/run.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
TEMP=Path(json.loads((OUT/'location.json').read_text())['temp']);G=1024**3
base.OUT=OUT;base.TEMP=TEMP;base.start=time.monotonic()
runtime=harness=None
receipt=dict(starting_head='d7b1d37ff0e00c0011b324eef309534d01947e44',samples=[],invocations=[],stop=None,model_load_budget=1,summary_attempt_budget=1)
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,default=str)+'\n')
def rows(name):
 if not (OUT/name).exists():return []
 result=[]
 for line in (OUT/name).read_text().splitlines():
  try:result.append(json.loads(line))
  except json.JSONDecodeError:pass
 return result
def resource_check():
 s=base.sample();receipt['samples'].append(s)
 initial=receipt['initial'];reason=None
 if s['competitors']:reason='competing worker'
 elif s['pressure']!=1:reason='memory pressure'
 elif s['swap_used']!=0 or s['swapouts']>initial['swapouts']:reason='swap increase'
 elif s['available']<=64*G:reason='host reserve'
 elif max(s['footprint'],s['lifetime_footprint'])>=425*G:reason='physical footprint stop'
 elif s['elapsed']>=2100:reason='overall wall budget'
 if reason:raise RuntimeError(reason)
 save('resources.json',receipt);return s
def main():
 global runtime,harness
 initial=base.sample();receipt['initial']=initial
 try:
  assert not initial['competitors'] and initial['pressure']==1 and initial['swap_used']==0 and initial['available']>=455.226*G,'pre-launch resource refusal'
  assert (OUT/'preload.json').exists() and (OUT/'effective-configuration.json').exists()
  frozen=json.loads((OUT/'frozen-preflight.json').read_text())
  assert frozen['essential_prerequisites']
  assert all(hashlib.sha256((OUT/n).read_bytes()).hexdigest()==h for n,h in frozen['input_sha256'].items()),'frozen input drift'
  with socket.socket() as s:s.bind(('127.0.0.1',18083))
  for n in ['ready','go']:(TEMP/n).unlink(missing_ok=True)
  env=dict(os.environ,DSH_HOME=str(TEMP/'home'),DSH_AGENTS_HOME=str(TEMP/'agents'),PHASE3_WORKSPACE=str(TEMP/'workspace'),PHASE3_EVIDENCE=str(OUT),DS41F_CHECKPOINT='/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017',PYTHONDONTWRITEBYTECODE='1')
  with (OUT/'runtime.log').open('w') as log:
   runtime=subprocess.Popen(['/tmp/ds41f-step3br3/bin/python','-I','-B',str(OUT/'runtime.py')],env=env,cwd=TEMP/'runtime',stdout=log,stderr=log,start_new_session=True)
  base.runtime=runtime;receipt['runtime_pid']=runtime.pid;receipt['supervisor_pid']=os.getpid()
  loaded=False;last_progress=time.monotonic();last_size=0
  while True:
   assert runtime.poll() is None,'worker exited before readiness'
   s=resource_check()
   if not loaded and (TEMP/'ready').exists():
    assert s['available']>=455.226*G,'pre-load reserve';receipt['preload']=s
    (TEMP/'go').write_text('authorized exactly one R3 load\n');loaded=True
   if s['elapsed']>1200:raise TimeoutError('startup budget')
   try:health=base.health()
   except Exception:health={}
   if health.get('model_ready') and health.get('status')=='ready':receipt['health_before']=health;break
   time.sleep(2)
  tasks=json.loads((OUT/'tasks.json').read_text());session_id=None;session_start=time.monotonic();next_task=tasks['initial'];kind='initial';pressure_index=0
  cli=['node',str(Path.home()/'.local/share/deepseek-harness/node_modules/@deepseek-ai/dsh/lib/bin.js'),'--profile','headless','--json']
  while next_task is not None:
   index=len(receipt['invocations']);name=f'{index:02}';prompt=TEMP/('prompt-'+name+'.txt');prompt.write_text(next_task+'\n')
   henv=dict(env,DS41F_LOCAL_API_KEY='local-only-placeholder',PHASE3_INVOCATION=name,NODE_OPTIONS='--import='+str(OUT/'observer.mjs'))
   args=cli+([] if session_id is None else ['--session-id',session_id])
   with (OUT/('harness-'+name+'.jsonl')).open('w') as stdout,(OUT/('harness-'+name+'.stderr')).open('w') as stderr,prompt.open() as stdin:
    harness=subprocess.Popen(args,env=henv,cwd=TEMP/'workspace',stdin=stdin,stdout=stdout,stderr=stderr,start_new_session=True)
   entry=dict(index=index,kind=kind,pid=harness.pid,session_id=session_id,prompt_sha256=hashlib.sha256(next_task.encode()).hexdigest(),args=args);receipt['invocations'].append(entry)
   while harness.poll() is None:
    s=resource_check();assert runtime.poll() is None,'Runtime exited during session'
    if time.monotonic()-session_start>900:raise TimeoutError('session wall budget')
    wire=rows('wire.jsonl');lifecycle=rows('lifecycle.jsonl')
    size=sum((OUT/n).stat().st_size if (OUT/n).exists() else 0 for n in ['runtime.log','wire.jsonl','lifecycle.jsonl'])
    if size!=last_size:last_size=size;last_progress=time.monotonic()
    if time.monotonic()-last_progress>600:raise TimeoutError('no progress')
    if any(r.get('kind')=='budget_stop' for r in wire):raise RuntimeError('request fence')
    requests=[r for r in wire if r.get('kind')=='request' and r.get('path')=='/v1/messages']
    if requests:
     last=requests[-1];done=any(r.get('kind')=='native_message_stop' and r['request']==last['request'] for r in wire)
     age=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(last['at'])).total_seconds()
     if not done and age>180:raise TimeoutError('per-generation budget')
    time.sleep(1)
   entry['exit']=harness.returncode
   lifecycle=rows('lifecycle.jsonl');surfaces=[r for r in lifecycle if r.get('kind')=='surface' and r.get('invocation')==name]
   assert surfaces,'missing completed surface'
   session_id=surfaces[-1]['session_id'];entry['session_id']=session_id
   entry['pressure']=surfaces[-1]['measurement']['totalTokens']
   p=subprocess.run(['node',str(OUT/'persisted.mjs'),session_id,name],capture_output=True,text=True,timeout=60)
   (OUT/('persistence-'+name+'.log')).write_text((p.stdout+p.stderr).rstrip()+'\n');assert p.returncode==0,'official persistence verification failed'
   persisted=json.loads((OUT/('persisted-'+name+'.json')).read_text());chain=persisted['compaction']
   starts=[e for e in chain if e['type']=='compaction/start'];summaries=[e for e in chain if e['type']=='compaction/summary'];ends=[e for e in chain if e['type']=='compaction/end']
   save('resources.json',receipt)
   if harness.returncode!=0 or any('error' in e['data'] for e in ends):
    receipt['stop']='official run/compaction failed; no second attempt';break
   assert len(starts)<=1,'second compaction marker'
   if kind=='initial':
    assert len([r for r in lifecycle if r.get('kind')=='tool_guard' and r['decision']=='allow'])==2,'did not complete two acquisitions'
   if summaries:
    assert len(summaries)==len(starts)==len(ends)==1,'incomplete durable transaction'
    if kind=='compatible':receipt['completed']=True;break
    if entry['pressure']>=4096:receipt['stop']='post-compaction pressure unsafe for another hook';break
    next_task=tasks['compatible'];kind='compatible'
   elif entry['pressure']>=4096:next_task=tasks['trigger'];kind='trigger'
   elif pressure_index<3:next_task=tasks['pressure_followups'][pressure_index];pressure_index+=1;kind='pressure-'+str(pressure_index)
   else:receipt['stop']='threshold not reached within three frozen followups';break
   ordinary=[r for r in rows('wire.jsonl') if r.get('kind')=='request' and r.get('path')=='/v1/messages' and r.get('purpose')=='ordinary']
   if len(ordinary)>=9:receipt['stop']='ordinary generation budget exhausted';break
  receipt['health_after']=base.health()
 except BaseException as error:receipt['stop']=repr(error)
 finally:
  base.retire(harness);base.retire(runtime)
  receipt['runtime_exit']=None if runtime is None else runtime.returncode
  receipt['elapsed_s']=time.monotonic()-base.start;receipt['postflight']=base.sample()
  save('resources.json',receipt)
  save('source-final.json',[dict(label=r['label'],unchanged=hashlib.sha256(Path(r['copy']).read_bytes()).hexdigest()==r['source_sha256']) for r in json.loads((OUT/'images.json').read_text())])
  with socket.socket() as s:free=s.connect_ex(('127.0.0.1',18083))!=0
  save('process-cleanup.json',dict(port_free=free,runtime_exited=runtime is None or runtime.poll() is not None,harness_exited=harness is None or harness.poll() is not None))
 print(json.dumps({k:v for k,v in receipt.items() if k!='samples'},indent=2))
if __name__=='__main__':main()
