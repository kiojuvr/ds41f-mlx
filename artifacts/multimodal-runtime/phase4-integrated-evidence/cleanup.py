"""Post-exit preservation and deletion of this task's disposable roots only."""
import hashlib,json,shutil,socket,tarfile
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent
location=json.loads((OUT/'location.json').read_text());temp=Path(location['temp'])
assert temp.name.startswith('ds41f-phase4-integrated-session-')
r=json.loads((OUT/'resources.json').read_text())
assert r['runtime_exit']==0 and not psutil.pid_exists(r['runtime_pid'])
assert json.loads((OUT/'process-cleanup.json').read_text())['port_free']
archive=OUT/'official-session.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for name in ['home/sessions','home/attachments','home/llm-deepseek','home/profiles','go','ready']:
  tar.add(temp/name,arcname=name)
installed=Path(location['installation']);build=Path(location['build'])
assert installed==Path('/tmp/ds41f-phase4-integrated-runtime') and build==Path('/tmp/ds41f-phase4-integrated-build')
for source,dest in [(installed/'share/ds41f-mtp/checkpoint-bytes.json','installed-checkpoint-bytes.json'),(installed/'lib/python3.13/site-packages/ds41f_mlx/execution_manifest.json','installed-execution-manifest.json')]:shutil.copyfile(source,OUT/dest)
(temp/'workspace').chmod(0o755)
removed=[]
for root in [temp,installed,build]:shutil.rmtree(root);removed.append(dict(path=str(root),absent=not root.exists()))
with socket.socket() as s:free=s.connect_ex(('127.0.0.1',18083))!=0
report=dict(archive=archive.name,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),archive_bytes=archive.stat().st_size,removed=removed,port_free=free,available=psutil.virtual_memory().available,swap_used=psutil.swap_memory().used,scope='POST_EXIT_OS_RECLAMATION_NOT_SAME_PROCESS_MODEL_DESTRUCTION')
(OUT/'cleanup.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
