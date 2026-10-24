"""Evaluation-only Q2 setup; reuses Phase 3 official profile and observers."""
from pathlib import Path
import json, hashlib, subprocess
ROOT=Path('/Volumes/SDXC-512/ds41f-mlx')
OUT=Path(__file__).resolve().parent
OLD=OUT.parent/'phase-3-evidence/visual-loop'
def save(n,x): (OUT/n).write_text(json.dumps(x,indent=2)+'\n')
def sha(b): return hashlib.sha256(b).hexdigest()
# Verify before starting any backend. No numerical imports.
installed=Path('/tmp/ds41f-phase1-1017-tests/lib/python3.13/site-packages/ds41f_mlx')
rows=json.loads((OLD/'source-package-match.json').read_text())['sources']
check=[dict(path=r['path'],expected=r['sha256'],actual=sha((installed/r['path']).read_bytes())) for r in rows]
save('source-package-match.json',check)
assert all(r['expected']==r['actual'] for r in check)
# Pin checkout used for bootstrap source comparisons to the qualified source.
qualified=OUT/'qualified-source'
qualified.mkdir(exist_ok=True)
for name in ['serving/server.py','serving/production_mtp.py','serving/deepseek_recipe_backend.py']:
 p=qualified/'ds41f_mlx'/name;p.parent.mkdir(parents=True,exist_ok=True)
 p.write_bytes(subprocess.check_output(['git','show','2fdea4fe7367e540300d7c81d4fed402d716a274:ds41f_mlx/'+name],cwd=ROOT))
 assert p.read_bytes()==(installed/name).read_bytes()
# Reuse existing setup, changing only Q2 budgets/task below.
source=(OLD/'setup.py').read_text().replace('policy = dict(command=command, tools=30,','policy = dict(command=command, tools=45, max_renders=7,').replace('requests=24','requests=32').replace('tools=30','tools=45').replace('max_renders=6','max_renders=7').replace('max_correction_iterations=5','max_correction_iterations=6').replace('body_limit_bytes=16777216','body_limit_bytes=8388608')
(OUT/'setup.py').write_text(source)
subprocess.run(['python3',str(OUT/'setup.py')],check=True)
TEMP=Path(json.loads((OUT/'location.json').read_text())['temp'])
# Fixed long content; intentionally interacting CSS defects, not disclosed to model.
sessions=[('08:30–09:00','Main Hall','Opening keynote: Building Resilient Conference Communities Across Continents','Dr. Alexandra Montgomery-Wells','All delegates'),('09:00–09:45','Main Hall','Responsible AI in Public Services: From Research Pilots to Everyday Operations','Prof. Christopher Fernández','Technology'),('09:45–10:30','Harbor Room','Accessible Event Design for Multilingual and Neurodiverse Audiences','Priya Ramanathan-Sørensen','Operations'),('10:45–11:30','Main Hall','Climate-Aware Infrastructure and the Future of Large-Scale Scientific Meetings','Dr. Benjamin Okonkwo','Strategy'),('11:30–12:15','Harbor Room','Incident Response Tabletop: Coordinating Volunteers, Vendors and Venue Teams','María-José Castellanos','Workshop'),('13:15–14:00','Main Hall','Open Data Partnerships: Practical Lessons from International Research Networks','Dr. Evelyn Chen-Laurent','Research'),('14:00–14:45','Harbor Room','Making Hybrid Participation Feel Equitable for Remote and On-Site Delegates','Mohammed Al-Hassan','Experience'),('15:00–15:45','Main Hall','Closing Panel: Sustainable Growth Without Losing Community Trust','Dr. Sophie van der Meer','All delegates')]
css='''*{box-sizing:border-box}body{margin:0;background:#eef2f7;color:#1c2b42;font:14px Arial}#page{width:1216px;margin:18px auto}#header{background:#102c49;color:white;padding:12px 24px;height:80px}h1{font-size:26px;margin:0 0 6px}.subtitle{color:#b8c9dc;margin:0}#cards{display:flex;gap:13px;margin:25px 0 12px}.card{background:white;padding:12px;width:25%;height:86px;border-left:5px solid #278d90}.card:nth-child(2){height:110px;padding-top:26px}.card:nth-child(3){margin-top:15px}.card strong{display:block;font-size:29px}.label{color:#61718a}#content{display:grid;grid-template-columns:875px 350px;gap:20px}section{background:white;padding:16px}h2{margin:0 0 14px;font-size:19px}table{width:100%;table-layout:fixed;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:12px 9px;border-bottom:1px solid #dbe3ed;white-space:nowrap;overflow:hidden}th{color:#61718a;font-size:11px;text-transform:uppercase}th:nth-child(1){width:92px}th:nth-child(2){width:95px}th:nth-child(3){width:285px}th:nth-child(4){width:155px}.summary-row{display:flex;justify-content:space-between;margin-bottom:21px}.summary-row b{font-size:20px}.note{line-height:1.6;color:#61718a;margin-top:35px}.pill{color:#187465;background:#e2f4ef;padding:5px 9px}#footer{margin-top:52px;padding:20px;background:#102c49;color:white;display:flex;justify-content:space-between}'''
body=''.join('<tr>'+''.join('<td>'+v+'</td>' for v in row)+'</tr>' for row in sessions)
starter='<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Northstar Conference Operations</title><style>'+css+'</style></head><body><main id="page"><header id="header"><h1>Northstar Conference 2026 — Operations</h1><p class="subtitle">Wednesday, 21 October · Harbor Convention Center · All times local</p></header><div id="cards"><div class="card" id="card-1"><span class="label">Registered delegates</span><strong>1,248</strong><span>96 organizations represented</span></div><div class="card" id="card-2"><span class="label">Checked in today</span><strong>986</strong><span>79% of registered delegates</span></div><div class="card" id="card-3"><span class="label">Sessions on schedule</span><strong>8 / 8</strong><span>Two rooms · Four tracks</span></div><div class="card" id="card-4"><span class="label">Volunteer coverage</span><strong>42 / 48</strong><span>Six afternoon shifts open</span></div></div><div id="content"><section id="schedule"><h2>Today’s conference schedule</h2><table><thead><tr><th>Time</th><th>Room</th><th>Session</th><th>Speaker</th><th>Track</th></tr></thead><tbody>'+body+'</tbody></table></section><section id="registration"><h2>Registration summary</h2><div class="summary-row"><span>Standard passes</span><b>864</b></div><div class="summary-row"><span>Student passes</span><b>276</b></div><div class="summary-row"><span>Speaker / partner</span><b>108</b></div><div class="summary-row"><span>Awaiting payment</span><b>32</b></div><p class="pill">Registration desk: OPEN</p><p class="note">Desk hours: 07:30–17:30<br>Help desk: Harbor Foyer, Level 1<br>Accessibility support: ask any desk volunteer.</p></section></div><footer id="footer"><span>Operations status: All systems operational</span><span>Last updated 08:15 · Next briefing 12:30</span></footer></main></body></html>'
command='/tmp/ds41f-phase2-renderer/bin/python /Volumes/SDXC-512/ds41f-mlx/tools/phase3_render_handoff.py request'
goal='''Repair this disposable static conference operations dashboard for a fixed 1280×720 viewport. Preserve all supplied visible text and figures, including all eight sessions, full session titles, speaker names, times, rooms and tracks. The finished page should have a shared header/title, aligned overview statistic cards, a readable dense schedule, a coherent registration summary and a visible status footer. All content must be legible without horizontal or vertical scrolling, clipping or overlap, with consistent spacing and a clear visual hierarchy. Keep text at least 12px (table headers may be 11px). Static HTML/CSS only, no scripts, network or external assets. Use screenshots to decide what needs correction and verify the whole page when useful. You may finish when satisfied; there is no required number of edits or images.
Start by saving the supplied starter verbatim to index.html with official write, then render and inspect it before making layout corrections. Use only official write/edit/read for index.html. Use one tool per response; never edit and render in the same response. For rendering call bash with exactly: '''+command+'''. Use the session workspace as workdir, no background execution or escalation. Then call official read_image on the returned screenshot pathname; rendering alone does not show you the image. At most seven renders AND seven image observations are available; do not request more. Do not retry uncertain tool effects or failed renders. Work only in index.html.
Starter HTML:
'''
rubric=['All fixed content retained, eight complete schedule rows with full titles/names','Header title/subtitle readable and coherent','Four cards aligned with equal dimensions and readable labels/values/support text','Schedule columns aligned and all cells readable without truncation','Registration counts, desk status/hours/location/support fully readable','Status footer and both messages entirely visible within viewport','No clipping/overlap or viewport scrolling in any section','Consistent spacing and visual hierarchy across sections','Body/table text >=12px, table headers >=11px with adequate contrast']
save('task.json',dict(goal=goal,starter=starter,rubric=rubric,task_sha256=sha((goal+starter).encode()),frozen_before_execution=True,acceptance='All nine checks required; independently assess screenshots, not renderer legacy quality scores.'))
(TEMP/'prompt.txt').write_text(goal+starter)
b=json.loads((OUT/'budgets.json').read_text());b.update(requests=32,tools=45,max_renders=7,aggregate_output_tokens=12000,expanded_prompt_positions=65536,body_limit_bytes=8388608,screenshot_bytes=524288);save('budgets.json',b)
# Same observer, only budget fence extended; native response/content are untouched.
s=(OLD/'observer.mjs').read_text().replace('let sequence=0, generations=0;','let sequence=0, generations=0, outputTokens=0;').replace("if(e.type==='message_delta') stop=e.delta?.stop_reason??stop;","if(e.type==='message_delta') { stop=e.delta?.stop_reason??stop; if(e.usage?.output_tokens) outputTokens+=e.usage.output_tokens; }") .replace('++generations>24 || r.body_bytes>16777216','++generations>32 || outputTokens>=12000 || r.body_bytes>8388608 || r.images.length>7')
(OUT/'observer.mjs').write_text(s)
s=(OLD/'runtime.py').read_text().replace("(ROOT/'ds41f_mlx'/source)","(OUT/'qualified-source/ds41f_mlx'/source)").replace('        row = dict(body_sha256=',"        assert len(prepared.token_ids) <= 65536, 'expanded prompt budget exhausted'\n        row = dict(body_sha256=")
(OUT/'runtime.py').write_text(s)
for n in ['preflight.mjs','preflight.py']:
 s=(OLD/n).read_text().replace('phase-3-evidence/visual-loop/denial-probe.txt','phase-4-q2-evidence/denial-probe.txt')
 if n=='preflight.py': s=s.replace("OUT.parent/'task.json'","OUT/'task.json'")
 (OUT/n).write_text(s)
# Run script uses the same single official CLI invocation and final flush ordering.
s=(OLD/'run.py').read_text().replace("OUT.parent/'identities.json'","OUT.parent/'phase-3-evidence/visual-loop/identities.json'")
s=s.replace("PHASE3_WORKSPACE=str", "PHASE3_MAX_RENDERS='7',PHASE3_WORKSPACE=str")
s=s.replace("    try:\n        assert json.loads", "    total_start=time.monotonic()\n    try:\n        assert json.loads")
s=s.replace("if time.monotonic()-start>5400", "if time.monotonic()-total_start>5400")
s=s.replace("            time.sleep(.2)",'''            traces=sorted(OUT.glob('trace-*.raw.json.gz'))
            import gzip
            parsed=[json.loads(gzip.decompress(p.read_bytes())) for p in traces]
            if sum(t['response']['usage']['output_tokens'] for t in parsed)>12000: raise RuntimeError('aggregate output cap')
            for t in parsed:
                assert t['prompt_replay']==t['full_cache_repack']==0, 'forbidden replay/repack'
                assert t['cache_published'] and t['authority_connection']['disposition']=='settled', 'missing settlement'
            memory=subprocess.run(['ps','-o','pid=,rss=','-p',str(runtime.pid)+','+str(harness.pid)],capture_output=True,text=True)
            with (OUT/'memory.jsonl').open('a') as f: f.write(json.dumps(dict(at=time.time(),ps_rss_kib=memory.stdout))+'\\n')
            time.sleep(.2)''')
(OUT/'run.py').write_text(s)
# Installed identities are independently compared with the retained qualified baseline.
assert json.loads((OUT/'identities.json').read_text())==json.loads((OLD/'identities.json').read_text())
save('pre-execution.json',dict(status='PASS',runtime_sources_matching=len(check),harness_identity_match=True,model_loaded=False,conversations=0,head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
print('Q2 setup frozen',TEMP)
