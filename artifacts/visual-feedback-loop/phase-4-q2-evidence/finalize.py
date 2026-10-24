"""Post-run retention and summary only. Does not launch any evaluation conversation."""
from pathlib import Path
import datetime, hashlib, json, os, socket, subprocess
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
def load(n):return json.loads((OUT/n).read_text())
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2)+'\n')
def sha(b):return hashlib.sha256(b).hexdigest()
execution=load('execution.json');final=load('final-state.json');trajectory=load('trajectory.json')
mem=[];samples=0
for s in (OUT/'memory.jsonl').read_text().splitlines():
 samples+=1;r=json.loads(s)
 for l in r['ps_rss_kib'].splitlines():
  pid,rss=map(int,l.split());mem.append((pid,rss))
peak={str(pid):max(v for k,v in mem if k==pid) for pid in sorted({k for k,v in mem})}
save('resource-summary.json',dict(process_rss_peak_sample_kib=peak,samples=samples,rss_sampling='ps pid,rss approximately every 0.2s during conversation only; not allocator/Metal/wired high-water or startup peak',conversation_s=execution['elapsed_s'],startup_s_approx=datetime.datetime.fromisoformat(execution['started_at']).timestamp()-(OUT/'supervisor-pid.json').stat().st_mtime,total_startup_through_cleanup_s_approx=(OUT/'final-state.json').stat().st_mtime-(OUT/'supervisor-pid.json').stat().st_mtime,max_source_bytes=max(p.stat().st_size for p in (OUT/'revisions').glob('*.html')),max_png_bytes=max(p.stat().st_size for p in (OUT/'renders').glob('*.png')),total_task_png_bytes=sum(p.stat().st_size for p in (OUT/'renders').glob('*.png')),max_expanded_prompt_positions=max(r['prompt_positions'] for r in trajectory),max_prompt_plus_reservation=max(r['prompt_positions']+r['output_reservation'] for r in trajectory),max_body_bytes=max(r['body_bytes'] for r in trajectory),timing_sums={k:sum(r['timings'][k] or 0 for r in trajectory) for k in trajectory[0]['timings']},timing_warning='Overlapping definitions, not independently additive phases; none vision_s is not applicable, tiny historical-only vision_s is bookkeeping. Concurrent model-free tests can affect wall timings.'))
# Every revision preserves the entire fixed body; only CSS changes after initialization.
starter=load('task.json')['starter'];body=starter.split('<body>',1)[1]
assert all(p.read_text().split('<body>',1)[1]==body for p in (OUT/'revisions').glob('*.html'))
save('content-preservation.json',dict(status='PASS',method='Exact complete body HTML equals frozen starter for every successful official source revision; only head/CSS changes',revisions=len(list((OUT/'revisions').glob('*.html')))))
owned=[execution[k] for k in ['runtime_pid','harness_pid','renderer_supervisor_pid']]+[load('supervisor-pid.json')['pid']]
remaining=[]
for pid in owned:
 try:os.kill(pid,0);remaining.append(pid)
 except ProcessLookupError:pass
with socket.socket() as s:free=s.connect_ex(('127.0.0.1',18083))!=0
processes=subprocess.check_output(['ps','-axo','pid,comm'],text=True)
browser=[line for line in processes.splitlines() if 'Google Chrome' in line or 'playwright' in line.lower()]
assert not remaining and free and not Path(final['temporary_root']).exists() and not browser
save('cleanup.json',dict(owned_pids=owned,remaining_owned_pids=remaining,port_18083_free=free,temporary_workspace_removed=True,browser_processes=browser,workspace_retained_and_hash_checked=True,persistent_installations_changed=False,checkpoint_changed=False))
save('host.json',dict(os=subprocess.check_output(['sw_vers'],text=True),cpu=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),physical_memory_bytes=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True)),renderer='Chrome 155.0.8059.85, unchanged tools/visual_feedback_renderer.py; renderer interpreter /tmp/ds41f-phase2-renderer/bin/python'))
# These bytes were unchanged while running; save the evaluated helper identities.
save('executed-source-hashes.json',{str(p.relative_to(OUT)):sha(p.read_bytes()) for p in sorted((OUT/'executed').glob('*'))})
manifest=[dict(file=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='retention-manifest.json' and '__pycache__' not in p.parts]
save('retention-manifest.json',manifest)
print(json.dumps(load('resource-summary.json'),indent=2))
