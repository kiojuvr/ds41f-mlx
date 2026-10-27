"""Retain task-owned durable storage, authenticate attachments, remove only owned roots."""
import hashlib,json,shutil,socket
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent
resources=json.loads((OUT/'resources.json').read_text());temp=Path(json.loads((OUT/'location.json').read_text())['temp'])
pids=[resources[k] for k in ['runtime_pid','supervisor_pid']]+[r['pid'] for r in resources['invocations']]
assert not any(psutil.pid_exists(p) for p in pids)
with socket.socket() as s:assert s.connect_ex(('127.0.0.1',18083))!=0
assert json.loads((OUT/'integrity-after.json').read_text())['original_runtime_matches']
# Preserve the actual official physical log as well as official read-handle receipts.
shutil.copytree(temp/'home/sessions',OUT/'session-storage')
storage=[]
for p in sorted((OUT/'session-storage').rglob('*')):
 if p.is_file():storage.append(dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
attachments=[]
for p in sorted((temp/'home/attachments').rglob('*')):
 if p.is_file():
  digest=hashlib.sha256(p.read_bytes()).hexdigest()
  attachments.append(dict(path=str(p.relative_to(temp)),bytes=p.stat().st_size,sha256=digest))
manifest=json.loads((OUT/'images.json').read_text())
assert all(any(a['sha256']==r['source_sha256'] for a in attachments) for r in manifest)
owned=[temp,Path('/tmp/ds41f-step3br3'),Path('/tmp/ds41f-step3br3-build')]
assert temp.name.startswith('ds41f-step3br3-session-') and temp.parent.is_dir()
for p in owned:shutil.rmtree(p)
receipt=dict(pids=pids,all_exited=True,port_free=True,removed=[str(p) for p in owned],all_removed=all(not p.exists() for p in owned),session_storage=storage,attachments_authenticated=attachments,source_fixtures_retained='phase3-step3a-evidence/image-{A,B}.png',original_inventory_verified=817,host_available_after_bytes=psutil.virtual_memory().available,swap_used=psutil.swap_memory().used,cleanup_qualification='PARTIAL; native final weakrefs alive, not requalified by deleting temp files')
(OUT/'cleanup.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
