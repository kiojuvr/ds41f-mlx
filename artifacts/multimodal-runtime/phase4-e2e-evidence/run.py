"""Single supervised native attempt, retained Step3A sampler/retirement and R3 inspector.
No inference POST, tool dispatch, Session mutation, cache control or retry here.
"""
import datetime,gzip,hashlib,importlib.util,json,os,shutil,socket,subprocess,time
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];R1=OUT.parent/'phase4-step4ar1-evidence';R3=OUT.parent/'phase3-step3br3-evidence';A=OUT.parent/'phase4-step4a-evidence'
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
base=module('provenance_monitor',OUT/'monitor.py');safe=module('qualified_safety',R1/'safety.py')
TEMP=Path(json.loads((OUT/'location.json').read_text())['temp']);base.OUT=OUT;base.TEMP=TEMP;base.start=time.monotonic()
clock=safe.ProgressClock(base.start);budget=safe.LaunchBudget();runtime=harness=None;loaded=False;session_start=None;persistence_ok=False
receipt=dict(starting_head='16a3bff68de47424bb84a9f0ee592a81586d20eb',samples=[],invocations=[],stop=None,loads=0,session_id=None)
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,default=str)+'\n')
def rows(n):
 result=[]
 if (OUT/n).exists():
  for line in (OUT/n).read_text().splitlines():
   try:result.append(json.loads(line))
   except json.JSONDecodeError:pass
 return result
def storage():
 return sum(p.stat().st_size for p in (TEMP/'home').rglob('*') if p.is_file()),sum(p.stat().st_size for p in (TEMP/'home').rglob('*') if p.is_file() and 'attachments' in p.parts)
def sample():
 s=base.sample();receipt['samples'].append(s);wire=rows('wire.jsonl');requests=[r for r in wire if r.get('kind')=='request' and r.get('path')=='/v1/messages'];age=None
 if requests:
  last=requests[-1]
  if not any(r.get('kind')=='native_message_stop' and r['request']==last['request'] for r in wire):age=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(last['at'])).total_seconds()
 semantic=[r for r in wire if r.get('kind') in ('sse_event','native_message_stop')]
 if semantic:clock.note('canonical_output',len(semantic),time.monotonic())
 sb,ab=storage();reason=safe.resource_stop(s,receipt['initial'],loaded=loaded,overall_s=s['elapsed'],session_s=None if session_start is None else time.monotonic()-session_start,last_progress_s=clock.age(time.monotonic()),generation_age_s=age,session_bytes=sb,attachment_bytes=ab)
 safe.publish_receipt(OUT/'supervision.json',stop=reason,session_bytes=sb,attachment_bytes=ab,persistence_ok=persistence_ok)
 save('resources.json',receipt)
 if reason:raise RuntimeError(reason)
 return s

def inspect_native():
 preparations=rows('prepared.jsonl');results=[]
 for i,p in enumerate(preparations):
  path=OUT/f'trace-{i:02}.json.gz'
  if not path.exists():raise RuntimeError('missing native trace at safe boundary')
  t=json.loads(gzip.decompress(path.read_bytes()));reason=safe.settled_trace_stop(p,t)
  if reason:raise RuntimeError(reason)
  C=t['cached_tokens'];results.append(dict(index=i,N=len(p['token_ids']),C=C,frontier=t['canonical_frontier'],fresh=t.get('vision_encoded_count',0),images=p['image_identities'],published=t['cache_published']))
 return results

def inspect_persistence(sid,name,env):
 """Only read-only inspection can be repeated; never resubmit Session work.
 On diagnostic failure pause with resources still sampled, awaiting an explicit
 repair receipt. Genuine protocol/effect/resource uncertainty remains fatal.
 """
 attempt=0
 while True:
  attempt+=1;log_path=OUT/f'persistence-{name}-{attempt}.log'
  with log_path.open('w') as log:
   diagnostic=subprocess.Popen(['node',str(OUT/'persisted.mjs'),sid,name],env=env,stdout=log,stderr=log,start_new_session=True)
   began=time.monotonic()
   try:
    while diagnostic.poll() is None:
     sample()
     if time.monotonic()-began>=60:base.retire(diagnostic);break
     time.sleep(1)
   finally:
    if diagnostic.poll() is None:base.retire(diagnostic)
  if diagnostic.returncode==0:return json.loads((OUT/f'persisted-{name}.json').read_text())
  save('diagnostic-pause.json',dict(session_id=sid,invocation=name,attempt=attempt,log=log_path.name,health=base.health(),action='Session paused; repair read-only inspection then create resume-inspection; no inference resubmission'))
  while not (OUT/'resume-inspection').exists():
   sample();assert runtime.poll() is None,'Runtime exited during diagnostic pause';time.sleep(1)
  (OUT/'resume-inspection').unlink()
  state=rows('gate.jsonl')[-1]['state']
  assert not state['stopped'] and not state['pending'],'uncertain protocol/effect outcome'
  assert base.health().get('model_ready'),'Runtime unhealthy during diagnostic pause'
  assert len(inspect_native())==len([r for r in rows('wire.jsonl') if r.get('kind')=='request' and r.get('path')=='/v1/messages']),'unaccounted native request'

def continuation(count,target):
 labels=list('ABCDEFGHI'+'A')[count:target]
 paths=', then '.join('image-'+label+'.png' for label in labels)
 return ('Continue the unfinished work only: read '+paths+
         ' in that order, one read_image call per response. Describe each image, '
         'then proceed to the next until all listed reads are complete. '
         'Do not repeat any completed read. Preserve the original comparison task.')

def main():
 global runtime,harness,loaded,session_start,persistence_ok
 assert not (TEMP/'go').exists(),'load already consumed; never relaunch this attempt'
 receipt['initial']=base.sample()
 try:
  for path,digest in [(OUT.parent/'phase4-step4br1-evidence/observer.mjs','41a9b1b35a4c30d65034ac33fb01f44567d3b48d6070b551731073507da6f766'),(OUT.parent/'phase4-step4br1-evidence/terminal.mjs','94fd324362bada98aa9df7b4f666bba26b0b57d98b2bed385cd51850b07c4f03'),(Path.home()/'.local/share/deepseek-harness/node_modules/@deepseek-ai/dsh-llm-deepseek/lib/index.js','eba35025ed1eedda89c51f2da46a3a968f7be1d59719f2ceeadfc4518dff37d6')]:
   assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,'qualified instrumentation/adapter identity'
  assert json.loads((OUT/'effective-configuration.json').read_text())['guard_registered']
  diagnostic_proof=json.loads((OUT/'diagnostic-preflight.json').read_text())
  assert diagnostic_proof['pass_'] and diagnostic_proof['inspector_sha256']==hashlib.sha256((OUT/'persisted.mjs').read_bytes()).hexdigest(),'actual corrected inspector identity'
  for n,h in json.loads((OUT/'frozen-inputs.json').read_text()).items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
  sample()
  with socket.socket() as s:s.bind(('127.0.0.1',18083))
  env=dict(os.environ,DSH_HOME=str(TEMP/'home'),DSH_AGENTS_HOME=str(TEMP/'agents'),PHASE4_WORKSPACE=str(TEMP/'workspace'),PHASE4_EVIDENCE=str(OUT),DS41F_CHECKPOINT='/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017',PYTHONDONTWRITEBYTECODE='1')
  with (OUT/'runtime.log').open('w') as log:runtime=subprocess.Popen(['/tmp/ds41f-phase4-e2e-runtime/bin/python','-I','-B',str(OUT/'runtime.py')],cwd=TEMP/'runtime',env=env,stdout=log,stderr=log,start_new_session=True)
  base.runtime=runtime;receipt['runtime_pid']=runtime.pid;receipt['supervisor_pid']=os.getpid()
  while True:
   assert runtime.poll() is None,'worker exited before readiness'
   s=sample()
   if not loaded and (TEMP/'ready').exists():
    # All worker-installed identity/source/function gates have completed before load.
    assert json.loads((OUT/'preload.json').read_text())['authenticated_graph_preserved']
    assert json.loads((OUT/'admission.json').read_text())['result_sha256']=='185d7692ae717f721a3e33ed4661b857c3582ca9ad4eb01691c8415c05bb7d99'
    assert json.loads((OUT/'source-match.json').read_text())['source_files_matched']==114
    assert len(json.loads((OUT/'source-admission.json').read_text())['rows'])==12
    clock.note('installed_proof','worker',time.monotonic());budget.authorize_load();receipt['loads']=budget.loads;receipt['preload']=s
    (TEMP/'go').write_text('single supervised Phase4B load authorized\n');loaded=True
   try:health=base.health()
   except Exception:health={}
   if health.get('model_ready') and health.get('status')=='ready':receipt['health_before']=health;clock.note('model_loaded',runtime.pid,time.monotonic());break
   if s['elapsed']>=1200:raise TimeoutError('startup duration')
   time.sleep(2)
  tasks=json.loads((A/'tasks.json').read_text());phase='round1';prompt=tasks[phase];review=0;sid=None;session_start=time.monotonic()
  cli=['node',str(Path.home()/'.local/share/deepseek-harness/node_modules/@deepseek-ai/dsh/lib/bin.js'),'--profile','headless','--json']
  while prompt is not None:
   sample();budget.begin_invocation(sid);index=len(receipt['invocations']);name=f'{index:02}'
   henv=dict(env,DS41F_LOCAL_API_KEY='local-only-placeholder',PHASE4_ROUND=phase,PHASE4_INVOCATION=name,NODE_OPTIONS='--import='+str(OUT.parent/'phase4-step4br1-evidence/observer.mjs'))
   args=cli+([] if sid is None else ['--session-id',sid]);prompt_path=TEMP/('prompt-'+name+'.txt');prompt_path.write_text(prompt+'\n')
   entry=dict(index=index,phase=phase,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),args=args,session_id=sid);receipt['invocations'].append(entry)
   with (OUT/f'harness-{name}.jsonl').open('w') as stdout,(OUT/f'harness-{name}.stderr').open('w') as stderr,prompt_path.open() as stdin:harness=subprocess.Popen(args,cwd=TEMP/'workspace',env=henv,stdin=stdin,stdout=stdout,stderr=stderr,start_new_session=True)
   while harness.poll() is None:
    sample();assert runtime.poll() is None,'Runtime exited during Session'
    states=rows('gate.jsonl')
    if states and states[-1]['state']['stopped']:raise RuntimeError(states[-1]['state']['stopped'])
    time.sleep(1)
   entry['exit']=harness.returncode
   surfaces=[r for r in rows('lifecycle.jsonl') if r.get('kind')=='surface' and r.get('invocation')==name]
   assert surfaces,'no official completed surface'
   sid=surfaces[-1]['session_id'];entry['session_id']=sid;receipt['session_id']=sid;pressure=surfaces[-1]['measurement']['totalTokens'];entry['pressure']=pressure
   assert harness.returncode==0,'official CLI failure'
   budget.end_invocation(sid,success=True)
   persisted=inspect_persistence(sid,name,henv);chain=persisted['compaction'];starts=[e for e in chain if e['type']=='compaction/start'];summaries=[e for e in chain if e['type']=='compaction/summary'];ends=[e for e in chain if e['type']=='compaction/end']
   assert len(starts)<=1 and all('error' not in e['data'] for e in ends),'failed/second compaction'
   time.sleep(.6);native=inspect_native();save('native-checks.json',native);clock.note('persisted',name,time.monotonic())
   state=rows('gate.jsonl')[-1]['state'];assert not state['stopped'] and not state['pending'],'uncertain tool or request outcome'
   # Ordinary end_turn completes a turn, not necessarily the requested workload.
   # Completed effects are authoritative: ask only for the unfinished suffix.
   count=len(state['effects'])
   target={'round1':3,'round2':6,'round3':10}.get(phase)
   if target is not None and count<target:
    prompt=continuation(count,target)
    entry['unfinished_effects']=list('ABCDEFGHI'+'A')[count:target]
   elif phase=='round1':
    phase='round2';prompt=tasks[phase]
   elif phase=='round2' or phase=='review':
    if starts:
     persistence_ok=True;phase='round3';prompt=tasks[phase]
    elif pressure>=8192:
     phase='trigger';prompt=tasks['trigger']
    elif review<4:
     phase='review';prompt=tasks['pressure_followups'][review];review+=1
    else:
     # No additional filler or changed policy to force a checkpoint.
     phase='trigger';prompt=tasks['trigger']
   elif phase=='trigger':
    persistence_ok=bool(summaries)
    phase='round3';prompt=tasks[phase]
   elif phase=='round3':
    phase='compatible';prompt=tasks[phase]
   else:receipt['completed']=True;prompt=None
   if phase=='round3' and not state['round3Entered'] and pressure>4608:
    raise RuntimeError('frozen Round3 headroom budget exhausted')
   if prompt is not None:
    # Reserve the final comparison and the minimum genuine generations needed
    # for remaining reads. Never spend the frozen final slots on review filler.
    remaining=10-count
    needed=remaining+1+(phase!='compatible')
    if state['http']+needed>20 or budget.invocations>=9:
     raise RuntimeError('finite execution budget exhausted before remaining work')
   save('resources.json',receipt)
  receipt['health_after']=base.health()
 except BaseException as error:receipt['stop']=repr(error)
 finally:
  sb,ab=storage();safe.publish_receipt(OUT/'supervision.json',stop=receipt['stop'] or 'normal shutdown',session_bytes=sb,attachment_bytes=ab,persistence_ok=persistence_ok)
  base.retire(harness);base.retire(runtime)
  receipt['runtime_exit']=None if runtime is None else runtime.returncode;receipt['elapsed_s']=time.monotonic()-base.start;receipt['postflight']=base.sample();save('resources.json',receipt)
  save('source-final.json',[dict(label=r['label'],unchanged=hashlib.sha256(Path(r['copy']).read_bytes()).hexdigest()==r['sha256']) for r in json.loads((OUT/'sources.json').read_text())])
  with socket.socket() as s:free=s.connect_ex(('127.0.0.1',18083))!=0
  save('process-cleanup.json',dict(port_free=free,runtime_exited=runtime is None or runtime.poll() is not None,harness_exited=harness is None or harness.poll() is not None))
 print(json.dumps({k:v for k,v in receipt.items() if k!='samples'},indent=2))
if __name__=='__main__':main()
