"""Authenticate pre-load stop evidence, then delete only this attempt's roots.
No admission replay, model load, Session creation or Runtime ownership changes.
"""
import hashlib,json,os,shutil,socket,subprocess
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent
location=json.loads((OUT/'location.json').read_text());resources=json.loads((OUT/'resources.json').read_text())
temp=Path(location['temp']);venv=Path(location['installation']);build=Path(location['build'])
assert temp.name.startswith('ds41f-step4br2-session-') and not temp.is_symlink()
assert venv.resolve()==Path('/private/tmp/ds41f-step4br2-runtime') and not venv.is_symlink()
assert build.resolve()==Path('/private/tmp/ds41f-step4br2-build') and not build.is_symlink()
assert resources['loads']==0 and resources['invocations']==[] and resources['session_id'] is None
assert all(json.loads((OUT/'process-cleanup.json').read_text()).values())
for pid in [resources['runtime_pid'],resources['supervisor_pid'],204]:assert not psutil.pid_exists(pid),pid
with socket.socket() as s:assert s.connect_ex(('127.0.0.1',18083))!=0
assert not (temp/'ready').exists() and not (temp/'go').exists()
assert not (temp/'home/sessions').exists()
assert not list((temp/'runtime').rglob('*'))
proof=json.loads((OUT/'admission.json').read_text())
assert proof['runtime']['build']['source_commit']=='147367541f750a611a242f727032e5593138f7d6'
assert Path(proof['origins']['ds41f_mlx']['path']).is_relative_to(venv.resolve())
# Preserve normal full-checkpoint cache and manifest without reading/loading weights.
for name in ['checkpoint-bytes.json','manifest.json']:
 source=venv/'share/ds41f-mtp'/name
 if source.exists():shutil.copyfile(source,OUT/('installed-'+name))
installed=Path(proof['origins']['ds41f_mlx']['path']).parent
payload=installed/'execution_manifest.json'
if payload.exists():shutil.copyfile(payload,OUT/'installed-execution-manifest.json')
retained=[]
for source in temp.rglob('*'):
 if source.is_file():
  relative=source.relative_to(temp)
  # Profile/configuration and authenticated sources, no fabricated Session storage.
  target=OUT/'disposable-inputs'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
  retained.append(dict(path=str(target.relative_to(OUT)),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
(temp/'workspace').chmod(0o755)
for root in [temp,venv,build]:shutil.rmtree(root)
receipt=dict(retained_inputs=retained,deleted_roots=[str(x) for x in [temp,venv,build]],all_absent=all(not x.exists() for x in [temp,venv,build]),runtime_exited=True,runtime_exit=resources['runtime_exit'],port_free=True,model_loads=0,sessions=0,quiescence='No go/ready, Backend, Model, Scheduler, executor, request/native/source/capture/delivery owners or application lifespan were reached. Bounded original process-group SIGTERM retirement only; not native retirement/reclamation qualification.',post_exit_host=dict(available_bytes=psutil.virtual_memory().available,swap_used=psutil.swap_memory().used,pressure=int(subprocess.check_output(['/usr/sbin/sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True)),swapouts=[s for s in subprocess.check_output(['/usr/bin/vm_stat'],text=True).splitlines() if s.startswith('Swapouts:')]))
(OUT/'cleanup.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('PASS: pre-load stop evidence preserved; only authenticated task-owned roots deleted')
