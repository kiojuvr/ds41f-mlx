"""Independent read-only evidence audit of the stopped pre-load attempt."""
import hashlib,json,subprocess
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
def obj(n):return json.loads((OUT/n).read_text())
r=obj('resources.json');proof=obj('admission.json');c=obj('cleanup.json');audit=obj('audit.json')
assert r['starting_head']=='147367541f750a611a242f727032e5593138f7d6'
assert r['loads']==0 and not r['invocations'] and r['session_id'] is None
assert r['stop']=="RuntimeError('competing model worker')"
assert not r['initial']['competitors']
stop=r['samples'][-1];assert len(stop['competitors'])==1
candidate=stop['competitors'][0];assert candidate['pid']==204
assert candidate['cmd'][-1]==str(OUT/'runtime.py')
assert candidate['pid']!=r['runtime_pid'] and r['runtime_exit']==-15
assert c['all_absent'] and c['runtime_exited'] and c['port_free']
assert not any(Path(p).exists() for p in c['deleted_roots'])
for name in ['preload.json','source-admission.json','loaded-memory.json','prepared.jsonl','wire.jsonl','gate.jsonl','native-checks.json']:
 assert not (OUT/name).exists(),name
assert not list(OUT.glob('trace-*.json.gz')) and not list(OUT.glob('harness-*.jsonl'))
assert audit['native_requests']==audit['effects']==audit['offloads']==audit['compaction_starts']==0
assert audit['request_table']==[]
assert obj('preflight.json')['status']=='PASS' and obj('preflight.json')['source_files_matched']==114
assert proof['result_sha256']=='185d7692ae717f721a3e33ed4661b857c3582ca9ad4eb01691c8415c05bb7d99'
assert proof['runtime']['build']['source_commit']==r['starting_head']
assert len(proof['runtime']['checkpoint']['shards'])==48
assert {s['source_revision'] for s in proof['runtime']['checkpoint']['shards']}=={'1bae0e1088993b9b82de08f59de5be5a34e42170'}
for phase in ['before','after']:assert obj('integrity-'+phase+'.json')['original_runtime_matches']
assert all(x['unchanged'] for x in obj('source-final.json'))
for x in c['retained_inputs']:
 p=OUT/x['path'];assert p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']
for path,digest in obj('frozen-inputs.json').items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
historical=[]
for stem in ['phase4-step4a','phase4-step4ar1','phase4-step4b','phase4-step4br1']:
 historical+=list((OUT.parent).glob(stem+'-evidence'))
 historical+=list((OUT.parent).glob(stem+'-*.md'))
subprocess.run(['git','diff','--exit-code',r['starting_head'],'--',*[str(p.relative_to(ROOT)) for p in historical],'ds41f_mlx','tools'],cwd=ROOT,check=True,capture_output=True)
result=dict(pass_=True,classification='PRELOAD_SUPERVISOR_COMPETITION_REFUSAL',native_request_table=[],loads=0,sessions=0,images_acquired=0,effects=0,generations=0,offloads=0,compactions=0,candidate=candidate,candidate_ancestry='NOT_CAPTURED; argv matches task worker, not proof of external model ownership',worker_preload_completed=False,separate_isolated_full_preflight_pass=True,cleanup_pass=True,historical_and_production_unchanged=True,operational_records=817,authorization_exhausted=True)
(OUT/'preload-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS: independent empty native chronology, exact pre-load stop, admission/source identities, bounded cleanup and immutable history')
