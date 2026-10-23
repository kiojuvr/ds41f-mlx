"""One isolated protocol gate setup; no generation or image transformation."""
import hashlib, json, os, shutil, subprocess, tempfile
from pathlib import Path
ROOT=Path('/Volumes/SDXC-512/ds41f-mlx')
OUT=ROOT/'artifacts/visual-feedback-loop/phase-3-evidence/protocol-gate'
TEMP=Path(tempfile.mkdtemp(prefix='ds41f-gate-p-'))
for d in ['workspace','agents','runtime','home/profiles/headless']:
    (TEMP/d).mkdir(parents=True,exist_ok=True)
def save(n,x): (OUT/n).write_text(json.dumps(x,indent=2)+'\n')
save('location.json',{'temp':str(TEMP)})
images=['phase-0-evidence/probe.png','phase-2-evidence/continuation-1/model-run/render-00.png','phase-2-evidence/continuation-1/model-run/render-01.png']
rows=[]
for label,src in zip('ABC',images):
    p=ROOT/'artifacts/visual-feedback-loop'/src; dest=TEMP/'workspace'/f'image-{label}.png'
    shutil.copyfile(p,dest); dest.chmod(0o444)
    rows.append(dict(label=label,source=str(p.relative_to(ROOT)),copy=str(dest),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),copy_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),bytes=p.stat().st_size))
save('images.json',rows)
profile=TEMP/'home/profiles/headless'
(profile/'package.json').write_text(json.dumps(dict(name='ds41f-gate-p',private=True,dependencies={},dsh=dict(profile=dict(bundles=['@deepseek-ai/dsh-base','@deepseek-ai/dsh-headless'])))))
(profile/'cordis.yml').write_text('[]\n')
patch=(ROOT/'artifacts/visual-feedback-loop/phase-0-evidence/profile.patch.yml').read_text().replace(':8000',':18083').replace('maxTokens: 512','maxTokens: 1200').replace('streamIdleTimeoutMs: 300000','streamIdleTimeoutMs: 180000').replace('/tmp/ds41f-phase0-harness.oLw1RR/workspace',str(TEMP/'workspace'))
patch+='''- id: plan-mode
  disabled: true
- id: approval
  config:
    policy: never
- id: agent-loop
  config:
    maxParallelToolCalls: 1
- insert:
    - id: protocol-budget
      name: '''+str(OUT/'budget-guard.mjs')+'\n'
(profile/'cordis.patch.yml').write_text(patch); (OUT/'profile.patch.yml').write_text(patch)
prompt='''Perform one sequential image-observation task using only read_image. First call read_image for image-A.png. After receiving its actual image result, describe distinctive visible details, then call read_image for image-B.png. Only after receiving B's actual result, describe B and distinguish it visually from A, then call read_image for image-C.png. After receiving C's actual result, describe visible differences between B and C and finish with a short comparison of all three observed images. Use exactly three read_image calls, one per model response, in A, B, C order. Do not batch or parallelize calls. Do not read any other files or use other tools. Base descriptions on the observed images; do not infer contents from filenames. No rendering or editing is requested.'''
(TEMP/'prompt.txt').write_text(prompt+'\n'); save('task.json',{'prompt':prompt})
save('budgets.json',dict(conversations=1,requests=4,image_effects=3,max_tokens_per_request=1200,total_s=900,generation_wall_s=180,startup_s=1200,body_bytes=16777216,thinking='disabled',retry=False,note='Minimal monotonic guard only enforces three image effects and allowed paths; no scheduling, arguments/results changes, or model requests.'))
old=json.loads((ROOT/'artifacts/visual-feedback-loop/phase-3-evidence/identities.json').read_text())
install=Path.home()/'.local/share/deepseek-harness/node_modules/@deepseek-ai'
actual={}
for n,expected in old['packages'].items():
    p=install/n; actual[n]=dict(version=json.loads((p/'package.json').read_text())['version'],implementation_sha256=hashlib.sha256((p/('lib/bin.js' if n=='dsh' else 'lib/index.js')).read_bytes()).hexdigest())
    assert actual[n]==expected,n
save('identities.json',dict(node=subprocess.check_output(['node','--version'],text=True).strip(),packages=actual,all_match_phase3=True))
# Reuse admitted bootstrap byte-for-byte. Retained observer adds only a request fence.
shutil.copyfile(OUT.parent/'runtime.py',OUT/'runtime.py')
assert (OUT/'observer.mjs').exists()
print(TEMP)
