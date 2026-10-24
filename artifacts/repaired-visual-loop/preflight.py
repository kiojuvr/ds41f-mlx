"""Model-free, fail-closed preflight; never imports numerical backends."""
from pathlib import Path
import hashlib,json,os,re,subprocess,socket
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
TEMP=Path(json.loads((OUT/'location.json').read_text())['temp'])
def save(n,x): (OUT/n).write_text(json.dumps(x,indent=2)+'\n')
def cmd(*a): return subprocess.check_output(a,text=True)
host={k:cmd(*v) for k,v in dict(processes=['ps','-axo','pid,ppid,rss,command'],vm=['vm_stat'],swap=['sysctl','vm.swapusage'],pressure=['memory_pressure','-Q'],hardware=['sysctl','hw.memsize'],os=['sw_vers']).items()}
old=json.loads((ROOT/'artifacts/extended-visual-loop/host-before.json').read_text())
host['deployed_manifests']={p:dict(sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in old['deployed_manifests']}
save('host-before.json',host)
page=int(re.search(r'page size of (\d+)',host['vm'])[1]); pages=lambda n:int(re.search(n+r':\s+(\d+)',host['vm'])[1])*page
reserve=sum(pages(n) for n in ['Pages free','Pages inactive','Pages speculative'])
assert reserve> (359.226+100)*1024**3, 'insufficient headroom for measured prior peak plus reserve'
assert 'used = 0.00M' in host['swap'] and pages('Swapins')==pages('Swapouts')==0
assert int(re.search(r'free percentage: (\d+)%',host['pressure'])[1])>=25
# Only inventory; do not terminate unrelated services.
assert not any('runtime.py' in l or 'run_ds41f_recipe_server.py' in l for l in host['processes'].splitlines() if 'python' in l.lower()), 'another Runtime visible'
with socket.socket() as s:s.bind(('127.0.0.1',18083))
env=dict(os.environ,DSH_HOME=str(TEMP/'home'),DSH_AGENTS_HOME=str(TEMP/'agents'),PHASE3_WORKSPACE=str((TEMP/'workspace').resolve()),PHASE3_EVIDENCE=str(OUT),DS41F_LOCAL_API_KEY='local-only-placeholder')
cli=Path.home()/'.local/share/deepseek-harness/node_modules/@deepseek-ai/dsh/lib/bin.js'
c=subprocess.run(['node',str(cli),'--profile','headless','--dump-config'],cwd=TEMP/'workspace',env=env,capture_output=True,text=True,timeout=60)
(OUT/'composed-config.yml').write_text(c.stdout);(OUT/'config-stderr.log').write_text(c.stderr)
assert c.returncode==0,c.stderr
assert c.stdout.count('maxTokens: 2600')==2 and 'maxTokens: 1000' not in c.stdout
for text in ['mode: native','enableRunInBackground: false','promoteOnTimeout: false','maxParallelToolCalls: 1','http://127.0.0.1:18083'] : assert text in c.stdout,text
b=json.loads((OUT/'budgets.json').read_text());assert b['output_reservation']==2600 and b['aggregate_output']==52000
observer=(OUT/'observer.mjs').read_text();controller=(OUT/'run.py').read_text()
assert 'r.max_tokens>2600' in observer and 'outputTokens + r.max_tokens > 52000' in observer
assert '>52000' in controller and '>18000' not in controller
policy=json.loads((TEMP/'policy.json').read_text()); policy['audit']=str(OUT/'preflight-policy.jsonl')
(TEMP/'preflight-policy.json').write_text(json.dumps(policy));env['PHASE3_POLICY']=str(TEMP/'preflight-policy.json')
subprocess.run(['node',str(OUT/'policy-check.mjs')],env=env,check=True,timeout=60)
# Use unchanged renderer, outside conversation; not agent screenshot occurrences.
images=[]
for label in ['original','formatted']:
 p=OUT/f'preflight-{label}.png'
 r=subprocess.run(['/tmp/ds41f-phase2-renderer/bin/python',str(ROOT/'tools/visual_feedback_renderer.py'),str(OUT/f'starter-{label}.html'),str(p)],capture_output=True,text=True,check=True,timeout=25)
 save(f'preflight-render-{label}.json',json.loads(r.stdout));images.append(hashlib.sha256(p.read_bytes()).hexdigest())
assert images[0]==images[1], 'presentation changed pixels'
assert images[0]==hashlib.sha256((ROOT/'artifacts/extended-visual-loop/renders/render-00.png').read_bytes()).hexdigest()
# Native source equality and exact installed identities were checked by prepare.
assert json.loads((OUT/'pre-execution.json').read_text())['status']=='PASS'
save('model-free-preflight.json',dict(status='PASS',model_loaded=False,provider_and_model_max_tokens=2600,aggregate_output=52000,observer_controller_consistent=True,source_inspection=True,policy_and_official_sandbox_checked=True,original_and_formatted_png_sha256=images[0],previous_challenge_pixel_identical=True,host_reserve_GiB=reserve/1024**3,prior_peak_GiB=359.226,process_ceiling_GiB=360,reserve_floor_GiB=100,source_and_harness_identity_match=True))
print((OUT/'model-free-preflight.json').read_text())
