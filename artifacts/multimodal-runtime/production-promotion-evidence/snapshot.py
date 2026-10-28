"""Read-only installed identities and exact RECORD-owned rollback snapshot."""
import base64, hashlib, importlib.metadata as md, json, pathlib, sys, tarfile
root = pathlib.Path(sys.prefix)
out = pathlib.Path(sys.argv[1])
dist = md.distribution('ds41f-mlx')
files = []
for f in dist.files:
    p = pathlib.Path(dist.locate_file(f)).resolve()
    assert p.is_relative_to(root), p
    if p.is_file():
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if f.hash:
            assert f.hash.mode == 'sha256'
            assert base64.urlsafe_b64encode(bytes.fromhex(digest)).rstrip(b'=').decode() == f.hash.value, p
        files.append(dict(path=str(p.relative_to(root)), sha256=digest, bytes=p.stat().st_size))
manifest = json.loads(pathlib.Path(dist.locate_file('ds41f_mlx/execution_manifest.json')).read_text())
result = dict(executable=sys.executable, prefix=sys.prefix, version=dist.version,
    direct_url=json.loads(dist.read_text('direct_url.json')), manifest=manifest,
    distributions=sorted((d.metadata['Name'], d.version, str(d.locate_file(''))) for d in md.distributions()), files=files)
out.write_text(json.dumps(result, indent=2)+'\n')
if len(sys.argv) > 2:
    with tarfile.open(sys.argv[2], 'w:gz') as archive:
        for f in files:
            archive.add(root/f['path'], arcname=f['path'], recursive=False)
    print('rollback sha256', hashlib.sha256(pathlib.Path(sys.argv[2]).read_bytes()).hexdigest())
print('package files', len(files), 'execution source', manifest['provenance']['source_commit'])
