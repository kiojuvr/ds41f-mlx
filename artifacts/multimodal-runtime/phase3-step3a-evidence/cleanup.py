"""Postflight identity check and deletion of exclusively task-owned disposable roots."""
import hashlib,json,shutil,socket,subprocess,time
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent
identities=json.loads((OUT/'identities.json').read_text());install=Path.home()/'.local/share/deepseek-harness/node_modules/@deepseek-ai'
for n,expected in identities['packages'].items():
 p=install/n;actual=dict(version=json.loads((p/'package.json').read_text())['version'],implementation_sha256=hashlib.sha256((p/('lib/bin.js' if n=='dsh' else 'lib/index.js')).read_bytes()).hexdigest())
 assert actual==expected,n
owned=[Path(json.loads((OUT/'location.json').read_text())['temp']),Path('/tmp/ds41f-step3a'),Path('/tmp/ds41f-step3a-build')]
assert all('step3a' in str(p) for p in owned)
resources=json.loads((OUT/'resources.json').read_text());pids=[resources[k] for k in ['runtime_pid','harness_pid','supervisor_pid']]
assert not any(psutil.pid_exists(p) for p in pids)
with socket.socket() as s:assert s.connect_ex(('127.0.0.1',18083))!=0
for p in owned:shutil.rmtree(p)
receipt=dict(official_packages_unchanged=True,inventory_integrity='integrity-after.json',pids=pids,all_absent=True,port_18083_free=True,removed=[str(p) for p in owned],all_removed=all(not p.exists() for p in owned),host_available_after_bytes=psutil.virtual_memory().available,swap_used=psutil.swap_memory().used,model_weakref_death='not observed',executor_shutdown='not independently observed',native_owner_cleanup='per-request trace verified; final destruction not independently observed')
(OUT/'cleanup.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
