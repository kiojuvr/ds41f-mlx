"""Reuse retained official Harness/external-renderer setup, not agent orchestration."""
from pathlib import Path
import json, subprocess, hashlib
ROOT=Path('/Volumes/SDXC-512/ds41f-mlx')
OUT=Path(__file__).resolve().parent
OLD=ROOT/'artifacts/visual-feedback-loop/phase-4-q2-evidence'
def save(n,x): (OUT/n).write_text(json.dumps(x,indent=2,default=str)+'\n')
# Existing setup creates only a disposable Harness home/workspace.
s=(OLD/'setup.py').read_text().replace('tools=45, max_renders=7','tools=70, max_renders=12')
(OUT/'setup.py').write_text(s)
subprocess.run(['python3',str(OUT/'setup.py')],check=True)
TEMP=Path(json.loads((OUT/'location.json').read_text())['temp'])
task=json.loads((OLD/'task.json').read_text())
# Presentation only: whitespace between tags and CSS rules; preflight demands
# pixel-identical rendering against the untouched original before model loading.
import re
starter = task['starter']
formatted = starter.replace('><', '>\n<')
formatted = re.sub(r'(<style>)(.*?)(</style>)', lambda m: m[1]+'\n'+m[2].replace('}', '}\n')+m[3], formatted, flags=re.S)
(TEMP/'workspace/index.html').write_text(formatted)
(OUT/'starter-original.html').write_text(starter)
(OUT/'starter-formatted.html').write_text(formatted)
command='/tmp/ds41f-phase2-renderer/bin/python /Volumes/SDXC-512/ds41f-mlx/tools/phase3_render_handoff.py request'
goal='''Repair the existing static conference operations dashboard index.html for 1280×720. Preserve ALL visible text and figures (eight full session titles, names, rooms, times, tracks), readable without scrolling, clipping or overlap. Text >=12px (table headers >=11px). Static HTML/CSS only, no scripts/network/assets. Shared header, four aligned statistic cards, readable dense schedule, coherent registration summary, visible status footer and consistent spacing are required.
Use actual screenshots, not just source inspection. First render and inspect the initial page. Work incrementally: diagnose concrete visible discrepancies, state briefly what you see and its location before each corrective edit, fix one coherent visual issue/section at a time, then render and inspect the result for regressions before moving on. Inspect header/cards, table, registration and footer; continue refining only while genuine defects remain. Do not blindly rewrite everything from source. Finish naturally when all requirements are met or explain a concrete blocker. No minimum image/edit count; never repeat an unchanged image just to use budget. Up to 12 screenshot observations are available.
Be concise: short image-specific observations and focused edit replacements, no lengthy reasoning or full-file rewrites. One tool per response. Use official read/edit only for index.html (write allowed if necessary). The source is presentation-formatted without changing its rendering. Bounded read-only bash inspection is also allowed: pwd, ls index.html, wc -lc index.html, or sed -n 'N,Mp' index.html (positive line numbers, at most 40 lines). Render with bash using EXACTLY: '''+command+'''. Use session workspace workdir, no background or escalation. Then official read_image on returned PNG path; bash rendering does not show the image. Never edit and render in the same response. Do not retry uncertain tool effects or failures. Do not read a screenshot twice.'''
(TEMP/'prompt.txt').write_text(goal)
save('task.json',dict(goal=goal,starter=task['starter'],rubric=task['rubric'],frozen_before_execution=True))
p=TEMP/'home/profiles/headless/cordis.patch.yml'
s=p.read_text(); assert s.count('maxTokens: 2600') == 2
p.write_text(s);(OUT/'profile.patch.yml').write_text(s)
s=(OLD/'observer.mjs').read_text().replace('++generations>32','++generations>50').replace('12000','52000').replace('r.images.length>7','r.images.length>12')
# Files metadata observations only: preserve fetch and provider behavior.
s=s.replace("else if(!response.ok)","else if(response.ok && url.pathname==='/v1/files') response.clone().json().then(j=>record({kind:'file_receipt',request,file:j})).catch(()=>{});\n  else if(!response.ok)")
s=s.replace("media_type:b.source?.media_type","media_type:b.source?.media_type,file_id:b.source?.file_id")
(OUT/'observer.mjs').write_text(s)
policy=json.loads((TEMP/'policy.json').read_text());policy.update(tools=70,max_renders=12);(TEMP/'policy.json').write_text(json.dumps(policy))
save('budgets.json',dict(requests=50,tools=70,renders=12,output_reservation=2600,aggregate_output=52000,body_bytes=8388608,prompt_positions=32768,total_s=5400,process_or_allocator_GiB=360,reserve_proxy_GiB=100,swap_allowed=0,pressure_free_min_percent=25,pressure_sustained_samples=2))
# Start from established single-conversation supervisor, never synthesize requests.
s=(OLD/'run.py').read_text().replace("OUT.parent/'phase-3-evidence/visual-loop/identities.json'","ROOT/'artifacts/visual-feedback-loop/phase-3-evidence/visual-loop/identities.json'")
s=s.replace("PHASE3_MAX_RENDERS='7'","PHASE3_MAX_RENDERS='12'").replace('/tmp/ds41f-phase1-1017-tests/bin/python','/private/tmp/ds41f-visual-capacity-test/bin/python').replace('>12000','>52000')
s=s.replace("assert json.loads((OUT/'preflight/result.json').read_text())['status']=='PASS'","assert json.loads((OUT/'pre-execution.json').read_text())['status']=='PASS'")
s=s.replace("            rows=[]","            if (OUT/'SAFETY_STOP.json').exists(): raise RuntimeError('resource safety stop; see receipt')\n            if runtime.poll() is not None: raise RuntimeError('Runtime exited')\n            rows=[]")
s=s.replace("        receipt.update(status='RUNNING'","        assert not (OUT/'SAFETY_STOP.json').exists(), 'startup safety stop'\n        receipt.update(status='RUNNING'")
s=s.replace("            assert runtime.poll() is None,'Runtime exited before ready'","            assert runtime.poll() is None,'Runtime exited before ready'\n            assert not (OUT/'SAFETY_STOP.json').exists(), 'startup safety stop'")
s=s.replace("        assert json.loads((OUT/'pre-execution.json').read_text())['status']=='PASS'", "        assert json.loads((OUT/'pre-execution.json').read_text())['status']=='PASS'\n        assert json.loads((OUT/'model-free-preflight.json').read_text())['status']=='PASS'")
s=s.replace("        assert c.returncode==0, c.stderr", "        assert c.returncode==0, c.stderr\n        assert c.stdout.count('maxTokens: 2600') == 2, 'effective generation budget'")
(OUT/'run.py').write_text(s)
# Full source equality; package/checkpoint admission will also run in Runtime process.
installed=Path('/private/tmp/ds41f-visual-capacity-test/lib/python3.13/site-packages/ds41f_mlx')
rows={}
for p in (ROOT/'ds41f_mlx').rglob('*.py'):
 rel=p.relative_to(ROOT/'ds41f_mlx');assert p.read_bytes()==(installed/rel).read_bytes(),str(rel)
 rows[str(rel)]=hashlib.sha256(p.read_bytes()).hexdigest()
save('source-package-match.json',rows)
assert json.loads((OUT/'identities.json').read_text())==json.loads((ROOT/'artifacts/visual-feedback-loop/phase-3-evidence/visual-loop/identities.json').read_text())
save('pre-execution.json',dict(status='PASS',sources=len(rows),official_harness_identity_match=True,head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),model_loaded=False))
s=(ROOT/'artifacts/harness-files/gate-f0-evidence/runtime.py').read_text().replace('6766b652fd3293f7523533decc102a9697168af1','dbbaa7289e4b16c89b2a9d437b029bd57ee549e7')
s=s.replace('        row = dict(body_sha256=',"        assert len(prepared.token_ids)<=32768, 'evaluation prompt cap'\n        row = dict(body_sha256=")
s=s.replace("    app = create_app(","    import mlx.core as mx\n    import telemetry\n    telemetry_stop, telemetry_thread, sample = telemetry.start(backend, mx)\n    app = create_app(")
s=s.replace("    server = uvicorn.Server", "    if (OUT/'SAFETY_STOP.json').exists(): raise RuntimeError('startup safety stop')\n    server = uvicorn.Server")
s=s.replace("            backend.close()\n            save('runtime-exit.json'", "            backend.close()\n            sample('after-shutdown')\n            telemetry_stop.set(); telemetry_thread.join(timeout=10)\n            save('runtime-exit.json'")
(OUT/'runtime.py').write_text(s)
print(TEMP)
