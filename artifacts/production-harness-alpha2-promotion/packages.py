"""RECORD-owned dual-distribution snapshot and exact wheel verification."""
import base64, hashlib, importlib.metadata as md, json, pathlib, sys, tarfile, zipfile
root = pathlib.Path(sys.prefix).resolve()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def inventory():
    packages = {}
    for name in ('ds41f-mlx', 'deepseek-recipe'):
        dist = md.distribution(name); files = []
        for f in dist.files:
            p = pathlib.Path(dist.locate_file(f)).resolve()
            assert p.is_relative_to(root), p
            assert p.is_file(), p
            digest = sha(p)
            if f.hash:
                assert f.hash.mode == 'sha256'
                assert base64.urlsafe_b64encode(bytes.fromhex(digest)).rstrip(b'=').decode() == f.hash.value, p
            files.append(dict(path=str(p.relative_to(root)), sha256=digest, bytes=p.stat().st_size))
        packages[name] = dict(version=dist.version, direct_url=dist.read_text('direct_url.json'), files=files)
    deps = sorted((d.metadata['Name'], d.version, str(d.locate_file('')), d.read_text('direct_url.json')) for d in md.distributions())
    manifest = json.loads(pathlib.Path(md.distribution('ds41f-mlx').locate_file('ds41f_mlx/execution_manifest.json')).read_text())
    return dict(prefix=str(root), executable=sys.executable, packages=packages, dependencies=deps, manifest=manifest)
mode, output = sys.argv[1:3]; output = pathlib.Path(output)
if mode == 'snapshot':
    result = inventory(); output.write_text(json.dumps(result, indent=2)+'\n')
    if len(sys.argv)>3:
        archive = pathlib.Path(sys.argv[3]); assert not archive.exists()
        files = [f for p in result['packages'].values() for f in p['files']]
        assert len({f['path'] for f in files}) == len(files)
        with tarfile.open(archive, 'w:gz') as t:
            for f in files: t.add(root/f['path'], arcname=f['path'], recursive=False)
        with tarfile.open(archive) as t:
            assert set(t.getnames()) == {f['path'] for f in files}
            for f in files:
                member=t.getmember(f['path']); assert member.isfile() and member.size==f['bytes']
                assert hashlib.sha256(t.extractfile(member).read()).hexdigest()==f['sha256']
        receipt=dict(archive=str(archive),sha256=sha(archive),members=len(files),verified=True)
        output.with_name('rollback-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt))
elif mode == 'verify':
    result=inventory(); wheels={}
    for arg in sys.argv[3:]:
        wheel=pathlib.Path(arg)
        with zipfile.ZipFile(wheel) as z:
            members=[n for n in z.namelist() if not n.endswith('/') and '.data/' not in n and not n.endswith('/RECORD')]
            for name in members:
                p=root/'lib/python3.13/site-packages'/name
                assert p.is_file() and sha(p)==hashlib.sha256(z.read(name)).hexdigest(), name
            wheels[str(wheel)]=dict(sha256=sha(wheel),members_verified=len(members))
    result['wheels']=wheels; output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(wheels))
else: raise ValueError(mode)
