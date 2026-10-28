"""No weights: run the exact corrected inspector on real retained histories.
Temporary copies only; no new Agent, synthetic event or native request.
"""
import hashlib,json,os,shutil,subprocess,tarfile,tempfile
from pathlib import Path
OUT=Path(__file__).resolve().parent
PARENT=OUT.parent
results=[]
with tempfile.TemporaryDirectory(prefix='ds41f-phase4-inspector-test-') as td:
 root=Path(td)
 for case in ['offload','compaction']:
  temp=root/case;temp.mkdir();output=temp/'output';output.mkdir()
  if case=='offload':
   with tarfile.open(PARENT/'phase4-integrated-evidence/official-session.tar.gz') as tar:tar.extractall(temp,filter='data')
   sid='session-91663bd1-4ee1-4e07-8568-757a381cd94c'
  else:
   shutil.copytree(PARENT/'phase3-step3br3-evidence/session-storage',temp/'home/sessions')
   sid='session-6c57807d-625b-43c4-8790-d8b11afc3a79'
  def hashes():return {str(p.relative_to(temp)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (temp/'home/sessions').rglob('*.zstd')}
  before=hashes()
  env=dict(os.environ,PHASE4_INSPECT_ROOT=str(temp),PHASE4_INSPECT_OUTPUT=str(output))
  result=subprocess.run(['node',str(OUT/'persisted.mjs'),sid,case],env=env,capture_output=True,text=True,timeout=30)
  assert result.returncode==0,result.stderr
  data=json.loads((output/f'persisted-{case}.json').read_text())
  assert data['officialReadOnlyHandle'] and before==hashes()
  offloads=sum(e['type']=='image/offload' for e in data['events']);summaries=sum(e['type']=='compaction/summary' for e in data['events'])
  assert (offloads==3 and len(data['events'])==65) if case=='offload' else summaries==1
  results.append(dict(case=case,session_id=sid,events=len(data['events']),offloads=offloads,summaries=summaries,durable_unchanged=True,inspector_exit=result.returncode))
location=json.loads((OUT/'location.json').read_text());temp=Path(location['temp'])
env=dict(os.environ,PHASE4_EVIDENCE=str(OUT),PHASE4_WORKSPACE=str(temp/'workspace'))
config=subprocess.run(['node',str(OUT/'config-preflight.mjs')],env=env,capture_output=True,text=True,timeout=60)
(OUT/'config-preflight.log').write_text((config.stdout+config.stderr).rstrip()+'\n');assert config.returncode==0,config.stderr
report=dict(pass_=True,weights_loaded=0,native_requests=0,new_sessions=0,inspector_sha256=hashlib.sha256((OUT/'persisted.mjs').read_bytes()).hexdigest(),real_history_tests=results,config_registration=True)
(OUT/'diagnostic-preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
