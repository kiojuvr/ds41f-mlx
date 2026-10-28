"""Read-only preflight: diagnostic wheel identity, provenance and rollback."""
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import sys
import zipfile
root = Path(sys.prefix)
dist = md.distribution('ds41f-mlx')
rollback = root.parent/'ds41f-1017-diagnostics-rollback-1af7108'
wheel = rollback/'promoted-wheel/ds41f_mlx-0.23.0-py3-none-any.whl'
assert hashlib.sha256(wheel.read_bytes()).hexdigest() == '2b5061ee5d4acadf4a2491412e8da663de53f441740940c9c659ab4c92179c98'
with zipfile.ZipFile(wheel) as z:
    names = [n for n in z.namelist() if n.startswith('ds41f_mlx/') and not n.endswith('/')]
    for name in names:
        assert z.read(name) == Path(dist.locate_file(name)).read_bytes(), name
manifest = json.loads(Path(dist.locate_file('ds41f_mlx/execution_manifest.json')).read_text())
print(json.dumps(dict(verified_wheel_members=len(names), wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
    runtime_root=str(root), direct_url=json.loads(dist.read_text('direct_url.json')),
    provenance=manifest.get('provenance')), indent=2))
