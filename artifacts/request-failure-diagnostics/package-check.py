"""Installed-only inventory/rollback verification; never changes dependencies."""
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import sys
import tarfile
import zipfile

root = Path(sys.prefix)
out = Path(sys.argv[2])
mode = sys.argv[1]
dist = md.distribution('ds41f-mlx')
def sha(data):
    return hashlib.sha256(data).hexdigest()
def inventory():
    files = {}
    for member in dist.files:
        path = Path(dist.locate_file(member)).resolve()
        assert path.is_relative_to(root)
        if path.is_file():
            files[str(path.relative_to(root))] = sha(path.read_bytes())
    return dict(files=files, dependencies={d.metadata['Name']: dict(version=d.version,
        direct_url=d.read_text('direct_url.json')) for d in md.distributions()
        if d.metadata['Name'].lower().replace('_', '-') != 'ds41f-mlx'})
if mode == 'snapshot':
    out.mkdir(exist_ok=False)
    inv = inventory()
    # Confirm the previous delivered wheel's package members, not just a version.
    wheel = Path(sys.argv[3])
    with zipfile.ZipFile(wheel) as z:
        checked = 0
        for name in z.namelist():
            if name.startswith('ds41f_mlx/') and not name.endswith('/'):
                assert sha(z.read(name)) == sha((Path(dist.locate_file(''))/name).read_bytes())
                checked += 1
    assert checked > 100
    (out/'previous-package.json').write_text(json.dumps(inv, indent=2))
    archive = out/'previous-package.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name in inv['files']:
            tar.add(root/name, arcname=name, recursive=False)
    with tarfile.open(archive) as tar:
        for name, digest in inv['files'].items():
            assert sha(tar.extractfile(name).read()) == digest
    receipt = dict(verified_files=len(inv['files']), verified_previous_wheel_members=checked,
                   archive_sha256=sha(archive.read_bytes()), root=str(root))
    (out/'rollback-verification.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))
elif mode == 'verify':
    before = json.loads((out/'previous-package.json').read_text())
    after = inventory()
    assert before['dependencies'] == after['dependencies']
    wheel = Path(sys.argv[3])
    with zipfile.ZipFile(wheel) as z:
        count = 0
        for name in z.namelist():
            if name.startswith('ds41f_mlx/') and not name.endswith('/'):
                assert sha(z.read(name)) == sha((Path(dist.locate_file(''))/name).read_bytes())
                count += 1
    (out/'promoted-package.json').write_text(json.dumps(after, indent=2))
    print(json.dumps(dict(unchanged_dependencies=True, wheel_sha256=sha(wheel.read_bytes()),
                         verified_wheel_members=count)))
