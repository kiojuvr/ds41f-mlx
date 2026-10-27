"""Final model-free gate and immutable-input hashes before any load authorization."""
import hashlib,json,subprocess
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
def read(n):return json.loads((OUT/n).read_text())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()=='d7b1d37ff0e00c0011b324eef309534d01947e44'
admission=read('admission.json');preload=read('preload.json');envelope=read('image-envelope.json');effective=read('effective-configuration.json')
assert admission['actual_pre_woa_materialized'] and admission['identity_object_preserved']
assert admission['profile']=='flash-1017-pre-woa-fp8-v1'
assert admission['result_sha256']=='185d7692ae717f721a3e33ed4661b857c3582ca9ad4eb01691c8415c05bb7d99'
assert preload['zero_weight_gate'] and preload['authenticated_graph_preserved']
assert len(envelope['images'])==2 and envelope['patches']==3042 and envelope['pixels']==294912 and envelope['canonical_tokens']<=16384
assert [effective['spec'][k] for k in ['thresholdTokens','retainTokens','maxTokens','compactionRetries']]==[4096,1024,2048,0]
assert len(effective['guardChecks'])==7
assert read('reference-regression.json')['passed']
assert '7 passed' in (OUT/'lifecycle-tests.log').read_text()
assert read('summary-budget-preflight.json')['provider_accepts_2048']
assert read('auxiliary-admission.json')['output_reservation']==2048
assert read('auxiliary-admission.json')['prepared_plus_output']<=16384
assert read('integrity-before.json')['original_runtime_paths']==817
assert read('source-match.json')['starting_head']=='d7b1d37ff0e00c0011b324eef309534d01947e44'
files={}
for n in ['runtime.py','run.py','references.py','reference-regression.py','reference-regression.json','framework-identities.json','lifecycle-tests.log','summary-budget-preflight.json','auxiliary-admission.json','guard.mjs','observer.mjs','persisted.mjs','tasks.json','budgets.json','profile.patch.yml','composed-config.yml','effective-configuration.json','images.json','source-match.json','admission.json','image-envelope.json']:
 files[n]=hashlib.sha256((OUT/n).read_bytes()).hexdigest()
for rel in ['phase3-step3b-evidence/lifecycle.py','phase3-step3b-evidence/lifecycle-tests.log','phase3-step3br1-evidence/result.json','phase3-step3a-evidence/run.py','phase3-step3a-evidence/audit-results.json']:
 files['../'+rel]=hashlib.sha256((OUT.parent/rel).read_bytes()).hexdigest()
receipt=dict(essential_prerequisites=True,weights_loaded=0,head='d7b1d37ff0e00c0011b324eef309534d01947e44',input_sha256=files,notes=['R1 actual event-registration tests reused unchanged','official headless help exposes --session-id; same cwd/profile/storage frozen','prior native read_image/Files transport reused plus current two-image native Recipe admission','seven supported waterfall stop checks; incomplete/failed transaction rejected at post-listener pre-step boundary','Runtime admission precedes serving imports; actual numerical pre_woa materialized','resource supervisor sampled gates before load; trace retention includes incomplete requests','nonintrusive intermediate N-2/N-1 snapshots remain unavailable'])
(OUT/'frozen-preflight.json').write_text(json.dumps(receipt,indent=2)+'\n')
# Keep the zero-weight receipts distinct from worker's repeated normal admission.
for name in ['admission','preload','image-envelope']:(OUT/(name+'-preflight.json')).write_bytes((OUT/(name+'.json')).read_bytes())
print('PASS: frozen essential model-free gates; zero model loads')
