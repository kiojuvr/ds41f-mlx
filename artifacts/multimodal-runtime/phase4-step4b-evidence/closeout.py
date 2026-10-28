"""Preserve focused evidence, authenticate and remove only this task's disposable roots."""
import gzip,hashlib,json,os,shutil,socket,subprocess,time
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
def save(n,v):(OUT/n).write_text(json.dumps(v,indent=2,default=str)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
location=json.loads((OUT/'location.json').read_text());temp=Path(location['temp'])
assert temp.name.startswith('ds41f-step4b-session-') and not temp.is_symlink()
cleanup=json.loads((OUT/'process-cleanup.json').read_text());assert all(cleanup.values())
resources=json.loads((OUT/'resources.json').read_text())
for pid in (resources['runtime_pid'],resources['supervisor_pid']):assert not psutil.pid_exists(pid),pid
with socket.socket() as s:assert s.connect_ex(('127.0.0.1',18083))!=0
# Keep original durable official storage, not another ledger.
store=OUT/'session-storage';assert not store.exists()
shutil.copytree(temp/'home/sessions',store)
retained=[]
for p in store.rglob('*'):
 if p.is_file():retained.append(dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)))
refs=[ROOT/'artifacts/multimodal-runtime'/n for n in ['phase3-step3a-evidence/run.py','phase3-step3br3-evidence/runtime.py','phase3-step3br3-evidence/references.py','phase3-step3br3-evidence/persisted.mjs','phase3-step3b-evidence/lifecycle.py','phase4-step4a-acceptance-contract.md','phase4-step4ar1-readiness.md']]
plugin=Path.home()/'.local/share/deepseek-harness/node_modules/@deepseek-ai/dsh-llm-deepseek/lib/index.js'
refs.append(plugin)
save('source-references.json',{str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha(p) for p in refs})
lines=plugin.read_text().splitlines()
save('observer-finding.json',dict(classification='TASK_INSTRUMENTATION_BLOCKER_NOT_DEMONSTRATED_RUNTIME_PROTOCOL_FAILURE',wire_request='00:1',message_stop='2026-10-27T23:57:54.755Z',captured_stop='2026-10-27T23:57:54.757Z',observer_error='2026-10-27T23:57:54.761Z',error_class='AbortError',message_stop_already_captured=True,source=str(plugin),source_sha256=sha(plugin),source_excerpts=[dict(start=1985,lines=lines[1984:2020]),dict(start=2140,lines=lines[2139:2162])],frozen_observer='phase4-step4ar1-evidence/observer.mjs',mechanism='Official translator returns normally at message_stop; adapter generate finally aborts its consumer. Frozen cloned-body observer marks even post-terminal read errors uncertain.',next_change='Separately authorize a task-observer terminal/cancellation classification fix and offline post-terminal-abort regression; preserve pre-terminal uncertainty fence. No change made here.',native_retry_performed=False))
# Authenticate the disposable installation against the admitted root and build provenance.
proof=json.loads((OUT/'admission.json').read_text());venv=Path(location['installation']);build=Path(location['build'])
assert venv.resolve()==Path('/private/tmp/ds41f-step4b-runtime') and not venv.is_symlink()
assert build.resolve()==Path('/private/tmp/ds41f-step4b-build') and not build.is_symlink()
assert Path(proof['origins']['ds41f_mlx']['path']).is_relative_to(venv.resolve())
assert proof['runtime']['build']['source_commit']=='993fd338107430ecde5e80c06c8159ac08a06c85'
save('starting-authority.json',dict(branch='Multimodal-Runtime',expected_head='993fd338107430ecde5e80c06c8159ac08a06c85',actual_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),initial_worktree_clean=True,initial_observation='git status --short empty before provisioning',checkpoint_revision='1bae0e1088993b9b82de08f59de5be5a34e42170',installed_identity=proof['runtime']['identity_sha256'],source_files_matched=114,production_changes=False,native_loads=1,native_sessions=1))
# No source facts, fixture bytes, native traces or official saved events are rewritten.
(temp/'workspace').chmod(0o755)
for root in [temp,venv,build]:shutil.rmtree(root)
save('cleanup.json',dict(retained_official_storage=retained,deleted_roots=[str(p) for p in [temp,venv,build]],all_absent=all(not p.exists() for p in [temp,venv,build]),runtime_exited=True,port_free=True,post_exit_host=dict(available_bytes=psutil.virtual_memory().available,swap_used=psutil.swap_memory().used,pressure=int(subprocess.check_output(['/usr/sbin/sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True)),swapouts=[x for x in subprocess.check_output(['/usr/bin/vm_stat'],text=True).splitlines() if x.startswith('Swapouts:')]),C7='PARTIAL: Backend/Model weakrefs live in worker; process exit is distinct'))
print('PASS: authenticated task roots retired after official storage/native evidence preservation')
