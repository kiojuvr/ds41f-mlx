"""Promotion receipts: installed RECORD integrity, rollback and wheel equivalence.

Runs with the selected installed interpreter in isolated mode. No admission or
payload rewriting. Runtime-only archive includes actual metadata and entrypoints.
"""
import base64
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import sys
import tarfile
import zipfile

O = Path(__file__).resolve().parent
ROOT = Path(sys.prefix).resolve()
SITE = ROOT/'lib/python3.13/site-packages'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def snapshot():
    packages = {}
    for name in ('ds41f-mlx', 'deepseek-recipe'):
        dist = md.distribution(name)
        files = []
        for f in dist.files:
            p = Path(dist.locate_file(f)).resolve()
            assert p.is_relative_to(ROOT) and p.is_file() and not p.is_symlink(), p
            digest = sha(p)
            if f.hash:
                assert f.hash.mode == 'sha256'
                assert base64.urlsafe_b64encode(bytes.fromhex(digest)).rstrip(b'=').decode() == f.hash.value, p
            files.append(dict(path=str(p.relative_to(ROOT)), sha256=digest, bytes=p.stat().st_size,
                              mode=p.stat().st_mode & 0o777))
        packages[name] = dict(version=dist.version, direct_url=dist.read_text('direct_url.json'), files=files)
    return dict(prefix=str(ROOT), packages=packages,
        dependencies=sorted((d.metadata['Name'], d.version, str(d.locate_file('')), d.read_text('direct_url.json'))
                            for d in md.distributions()),
        manifest=json.loads((SITE/'ds41f_mlx/execution_manifest.json').read_text()),
        recipe_native_sha256=sha(SITE/'deepseek_recipe/_native.abi3.so'))


def verify_archive(archive, inventory):
    files = inventory['packages']['ds41f-mlx']['files']
    with tarfile.open(archive) as t:
        assert len(t.getnames()) == len(files)
        assert set(t.getnames()) == {f['path'] for f in files}
        for f in files:
            m = t.getmember(f['path'])
            assert m.isfile() and m.size == f['bytes'] and m.mode == f['mode'], f
            assert hashlib.sha256(t.extractfile(m).read()).hexdigest() == f['sha256'], f
    return dict(archive=str(archive), sha256=sha(Path(archive)), members=len(files), verified=True)


def verify_wheel(wheel):
    with zipfile.ZipFile(wheel) as z:
        members = [n for n in z.namelist() if not n.endswith('/') and not n.endswith('/RECORD')]
        assert not any('.data/' in n for n in members), 'unexpected wheel relocation'
        for name in members:
            p = SITE/name
            assert p.is_file() and sha(p) == hashlib.sha256(z.read(name)).hexdigest(), name
    return dict(wheel=str(wheel), sha256=sha(Path(wheel)), members=len(members), verified=True)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'snapshot':
        save(sys.argv[2], snapshot())
    elif mode == 'backup':
        state = snapshot()
        archive = Path(sys.argv[3])
        assert not archive.exists()
        save(sys.argv[2], state)
        with tarfile.open(archive, 'w:gz') as t:
            for f in state['packages']['ds41f-mlx']['files']:
                t.add(ROOT/f['path'], arcname=f['path'], recursive=False)
        receipt = verify_archive(archive, state)
        save(O/'rollback-verification.json', receipt)
        print(json.dumps(receipt))
    elif mode == 'archive':
        receipt = verify_archive(sys.argv[2], json.loads(Path(sys.argv[3]).read_text()))
        save(sys.argv[4], receipt)
        print(json.dumps(receipt))
    elif mode == 'installed':
        state = snapshot()
        state['wheel'] = verify_wheel(sys.argv[3])
        save(sys.argv[2], state)
        print(json.dumps(state['wheel']))
    else:
        raise ValueError(mode)
