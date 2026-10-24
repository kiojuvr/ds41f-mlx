"""Gate F0 evaluation-only setup; no model requests or production modifications."""
import hashlib, json, subprocess
from pathlib import Path
ROOT=Path('/Volumes/SDXC-512/ds41f-mlx')
OUT=Path(__file__).resolve().parent
P=ROOT/'artifacts/visual-feedback-loop/phase-3-evidence/protocol-gate'
V=P.parent/'visual-loop'
def adapt(name, source, changes):
    text=(source/name).read_text()
    for a,b in changes:
        assert a in text,(name,a)
        text=text.replace(a,b)
    (OUT/name).write_text(text)
adapt('budget-guard.mjs',P,[("['A','B','C']","['A','B']"),('admitted>=3','admitted>=2'),('three read_image','two read_image')])
adapt('setup.py',P,[('artifacts/visual-feedback-loop/phase-3-evidence/protocol-gate','artifacts/harness-files/gate-f0-evidence'),("'phase-2-evidence/continuation-1/model-run/render-00.png','phase-2-evidence/continuation-1/model-run/render-01.png'","'phase-2-evidence/continuation-1/model-run/render-00.png'"),("zip('ABC',images)","zip('AB',images)"),('ds41f-gate-p','ds41f-gate-f0'),('requests=4,image_effects=3','requests=5,image_effects=2'),('three image effects','two image effects'),("shutil.copyfile(OUT.parent/'runtime.py',OUT/'runtime.py')","# Runtime is prepared separately with final synchronous trace flush."),("Only after receiving B's actual result, describe B and distinguish it visually from A, then call read_image for image-C.png. After receiving C's actual result, describe visible differences between B and C and finish with a short comparison of all three observed images. Use exactly three read_image calls, one per model response, in A, B, C order.","After receiving B's actual result, describe its visible elements, compare A and B, and finish. Use exactly two read_image calls, one per model response, in A, B order.")])
adapt('runtime.py',V,[("'2fdea4fe7367e540300d7c81d4fed402d716a274'","'6766b652fd3293f7523533decc102a9697168af1'"),("sources = ['serving/server.py', 'serving/production_mtp.py', 'serving/deepseek_recipe_backend.py']","sources = [str(p.relative_to(ROOT/'ds41f_mlx')) for p in (ROOT/'ds41f_mlx').rglob('*.py')]" )])
adapt('run.py',P,[('/tmp/ds41f-phase1-1017-tests/bin/python','/tmp/ds41f-f0-runtime/bin/python')])
adapt('observer.mjs',P,[("source_type:b.source?.type,media_type:b.source?.media_type","source_type:b.source?.type,file_id:b.source?.file_id,media_type:b.source?.media_type"),("r.multipart=true;","r.multipart=true; r.fields=[...body.entries()].map(([name,value])=>({name,...(value instanceof Blob?{mime:value.type,filename:value.name}:{value})}));"),('++generations>4','++generations>5'),("else if(!response.ok)","else if(response.ok && url.pathname==='/v1/files') response.clone().json().then(j=>record({kind:'file_receipt',request,metadata:j})).catch(e=>record({kind:'file_receipt_error',request,error_class:e.name}));\n  else if(!response.ok)")])
subprocess.run(['python3',str(OUT/'setup.py')],check=True)
rows=json.loads((OUT/'images.json').read_text())
expected=['7f594ec2e888cf9777a8a9bddc1baacc03df39459e30a43f37eaa9d7dd41f08a','fc6dd2f551cb196b84eb64d1bd0c91b01633ce881403d4bc4f0a1d602c1176d9']
assert [r['source_sha256'] for r in rows]==expected
assert all(r['source_sha256']==r['copy_sha256'] for r in rows)
subprocess.run(['uv','pip','freeze','--python','/tmp/ds41f-f0-runtime/bin/python'],stdout=(OUT/'dependencies-after.txt').open('w'),check=True)
a=(OUT/'dependencies-before.txt').read_text().splitlines(); b=(OUT/'dependencies-after.txt').read_text().splitlines()
assert [s for s in a if not s.startswith('ds41f-mlx')]==[s for s in b if not s.startswith('ds41f-mlx')]
(OUT/'dependency-check.json').write_text(json.dumps(dict(unchanged=True,installation='offline --no-deps',wheel_sha256=hashlib.sha256(next(Path('/tmp/ds41f-f0-wheels').glob('*.whl')).read_bytes()).hexdigest()),indent=2)+'\n')
