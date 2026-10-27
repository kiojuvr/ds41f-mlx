"""External, model-free exit observations; no Runtime import or lifecycle control."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import psutil

OUT = Path(__file__).resolve().parent
INSTALL = Path.home() / '.local/share/deepseek-harness/node_modules/@deepseek-ai'
identity = json.loads((OUT / 'installed-identities.json').read_text())
for name, record in identity['packages'].items():
    for relative, expected in record['files'].items():
        assert hashlib.sha256((INSTALL / name / relative).read_bytes()).hexdigest() == expected
with socket.socket() as sock:
    port_free = sock.connect_ex(('127.0.0.1', 18083)) != 0
assert port_free
result = dict(official_packages_unchanged=True, port_18083_free=port_free,
              task_owned_runtime_processes_started=0, task_owned_harness_sessions_started=0,
              task_owned_installations_created=0, weights_loaded=0,
              temporary_config_root_removed=json.loads((OUT / 'preflight-cleanup.json').read_text())['removed'],
              host_available_bytes=psutil.virtual_memory().available,
              pressure_level=int(subprocess.check_output(
                  ['/usr/sbin/sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], text=True)),
              swap_used_bytes=psutil.swap_memory().used,
              mlx_cleanup='not applicable: no Runtime or model was started')
(OUT / 'postflight.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
