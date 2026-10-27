"""Actual normal headless profile, model-free responses/no-effect execution only.
Never starts Runtime or uses model weights. No summary/replace is fabricated.
"""
import hashlib,json,os,re,subprocess,tempfile,time,shutil
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];OLD=OUT.parent/'phase4-step4a-evidence'
CLI=Path.home()/'.local/share/deepseek-harness/node_modules/@deepseek-ai/dsh/lib/bin.js'
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x] if p.exists() else []
def make_profile(t,controlled=True):
 w=t/'workspace';w.mkdir();p=t/'home/profiles/headless';p.mkdir(parents=True)
 for r in json.loads((OLD/'images.json').read_text()):shutil.copyfile(ROOT/r['source'],w/('image-'+r['label']+'.png'))
 (p/'package.json').write_text(json.dumps(dict(name='phase4-model-free-profile',private=True,dependencies={},dsh=dict(profile=dict(bundles=['@deepseek-ai/dsh-base','@deepseek-ai/dsh-headless'])))))
 (p/'cordis.yml').write_text('[]\n')
 base=(OUT.parent/'phase3-step3a-evidence/profile.patch.yml').read_text().split('- insert:')[0]
 base=re.sub(r'workspaceRoot: .*','workspaceRoot: '+str(w),base)
 base=base.replace('    streamIdleTimeoutMs:', '    retryPolicy: {mode: normal, maxRetries: 0}\n    streamIdleTimeoutMs:')
 base+=(OLD/'policy.patch.yml').read_text()
 base+='- insert:\n    - id: phase4-budget\n      name: '+str(OUT/'guard.mjs')+'\n'
 if controlled:base+='    - id: phase4-control\n      name: '+str(OUT/'control.mjs')+'\n'
 (p/'cordis.patch.yml').write_text(base)
 return w

def main():
 results=[];tasks=json.loads((OLD/'tasks.json').read_text())
 for scenario in ['concise','representative','large','usage-low','usage-high']:
  with tempfile.TemporaryDirectory(prefix='ds41f-4ar1-'+scenario+'-') as td:
   t=Path(td);w=make_profile(t);ev=t/'evidence';ev.mkdir()
   env=dict(os.environ,DSH_HOME=str(t/'home'),DSH_AGENTS_HOME=str(t/'agents'),PHASE4_WORKSPACE=str(w.resolve()),PHASE4_EVIDENCE=str(ev),PHASE4_SCENARIO=scenario,DS41F_LOCAL_API_KEY='model-free-only',NODE_OPTIONS='--import='+str(OUT/'control.mjs'))
   c=subprocess.run(['node',str(CLI),'--profile','headless','--dump-config'],env=env,cwd=w,capture_output=True,text=True,timeout=60)
   assert c.returncode==0,c.stderr
   # Config output has !!js expressions; actual CLI, not a hand-built substitute.
   if scenario=='concise':
    (OUT/'composed-config.yml').write_text(c.stdout.rstrip()+'\n')
    (OUT/'composition.json').write_text(json.dumps(dict(exit=c.returncode,composed_sha256=hashlib.sha256(c.stdout.encode()).hexdigest(),weights_loaded=0,kind='NORMAL_OFFICIAL_CLI_COMPOSITION'),indent=2)+'\n')
   sid=None;inv=[]
   prompts=[('round1',tasks['round1']),('round2',tasks['round2'])]+[('review',x) for x in tasks['pressure_followups']]+[('trigger',tasks['trigger'])]
   for index in range(7):
    if not prompts:break
    phase,prompt=prompts.pop(0)
    env['PHASE4_ROUND']=phase
    args=['node',str(CLI),'--profile','headless','--json']+([] if sid is None else ['--session-id',sid])
    with (ev/'stdout').open('w') as stdout,(ev/'stderr').open('w') as stderr:
     proc=subprocess.Popen(args,env=env,cwd=w,stdin=subprocess.PIPE,stdout=stdout,stderr=stderr,text=True)
     proc.stdin.write(prompt);proc.stdin.close()
     start=time.monotonic()
     while proc.poll() is None:
      (ev/'supervision.tmp').write_text(json.dumps(dict(at_ms=int(time.time()*1000),stop=None,session_bytes=sum(p.stat().st_size for p in (t/'home').rglob('*') if p.is_file()),attachment_bytes=31800,kind='MODEL_FREE_EXTERNAL_SUPERVISOR_INPUT')))
      (ev/'supervision.tmp').replace(ev/'supervision.json')
      if time.monotonic()-start>60:proc.terminate();raise TimeoutError('model-free CLI bound')
      time.sleep(.1)
    cr=rows(ev/'control.jsonl');snaps=[x for x in cr if x['kind']=='durable_snapshot']
    assert snaps,(scenario,proc.returncode,(ev/'stderr').read_text(),(ev/'stdout').read_text())
    sid=snaps[-1]['session_id'];pressure=snaps[-1]['pressure']['totalTokens']
    inv.append(dict(index=index,phase=phase,exit=proc.returncode,pressure=pressure,session_id=sid,adopted=index>0))
    if any(x['kind']=='summary_boundary' for x in cr):break
    if proc.returncode!=0:break
    # Same supported scheduling as retained R3: once completed pressure is high,
    # the next frozen followup is the SHORT trigger, not another review. No
    # event injection, threshold change or extra generation is involved.
    if pressure>=8192 and phase not in ('round1','trigger'):prompts=[('trigger',tasks['trigger'])]
   cr=rows(ev/'control.jsonl')
   negative=None
   if any(x['kind']=='summary_boundary' for x in cr):
    before_responses=sum(x['kind']=='controlled_response' for x in cr);before_gate=rows(ev/'gate.jsonl')[-1]['state']
    env['PHASE4_ROUND']='compatible'
    reject=subprocess.run(['node',str(CLI),'--profile','headless','--json','--session-id',sid],env=env,cwd=w,input='MODEL-FREE negative boundary test: this must be rejected before provider submission.',capture_output=True,text=True,timeout=60)
    cr=rows(ev/'control.jsonl');after_gate=rows(ev/'gate.jsonl')[-1]['state']
    assert sum(x['kind']=='controlled_response' for x in cr)==before_responses
    assert after_gate['attempts']==before_gate['attempts'] and after_gate['summary']==1
    negative=dict(kind='MODEL_FREE_NEGATIVE_ADOPTION_MUST_NOT_SUBMIT',exit=reject.returncode,provider_submissions=0,second_summary_attempts=0,same_session_id=sid)
   last=[x for x in cr if x['kind']=='durable_snapshot'][-1]
   events=last['snapshot'];ranges=[x for x in cr if x['kind']=='selected_range']
   result=dict(scenario=scenario,negative_adoption=negative,kind='CONTROLLED_FULL_OFFICIAL_PROFILE_NOT_NATIVE_EVIDENCE',invocations=inv,pre_steps=[dict(reads=x['reads'],pressure=x['measurement']['totalTokens'],baseline=x['measurement']['baseline'],header_config=x.get('header',{}).get('config'),surface=x['surface']) for x in cr if x['kind']=='pre_step'],read_count=len([x for x in cr if x['kind']=='controlled_effect']),offloads=[x for x in cr if x['kind']=='official_event' and x['event']['type']=='image/offload'],selected_range=ranges[0] if ranges else None,tool_definition=next(x for x in cr if x['kind']=='official_tool_definition'),actual_image_prices=next(x for x in cr if x['kind']=='actual_image_prices'),system_messages=[e for e in events if e['type']=='system/message'],runtime_context_messages=[e for e in events if e['type']=='user/message' and e['data'].get('source',{}).get('kind')=='runtime-context'],sample_tool_result=next((e['data']['message'] for e in events if e['type']=='tool/result'),None),latest_header=last['requestHeader'],gate_state=rows(ev/'gate.jsonl')[-1]['state'],attempt_failures=[e for e in events if e['type']=='assistant/attempt'],last_stderr=(ev/'stderr').read_text()[-4000:],compaction_events=[e for e in events if e['type'].startswith('compaction/')],provider_requests_after_failed_summary=len([x for x in cr[ next((i for i,x in enumerate(cr) if x['kind']=='summary_boundary'),len(cr))+1:] if x['kind']=='controlled_response']),summary_succeeded=False,replacement=False,max_local_storage_bytes=sum(p.stat().st_size for p in t.rglob('*') if p.is_file()))
   assert not any(e['type']=='compaction/summary' or e.get('surfaceOp',{}).get('op')=='replace' for e in events if isinstance(e.get('surfaceOp',{}),dict))
   assert result['provider_requests_after_failed_summary']==0
   results.append(result)
   print(scenario,'reads',result['read_count'],'offloads',len(result['offloads']),'pressure',[x['pressure'] for x in inv], 'range',ranges[0]['selected'] if ranges else None,flush=True)
 (OUT/'gate-b-profile.json').write_text(json.dumps(dict(weights_loaded=0,results=results),indent=2)+'\n')
if __name__=='__main__':main()
