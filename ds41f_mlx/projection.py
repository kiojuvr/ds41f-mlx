"""Inspect one-way source origin; runtime semantic patches are not a release process."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def verify(root=ROOT):
    root = Path(root)
    m = json.loads((root/'release/promotion.json').read_bytes())
    if m['schema'] != 'ds41f.promotion.v1':
        raise ValueError('unsupported promotion schema')
    if hashlib.sha256(canonical(m['files'])).hexdigest() != m['source_tree_sha256']:
        raise ValueError('source manifest identity drift')
    for name, entry in m['files'].items():
        p = root/name
        if p.is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'unsupported runtime-only source drift: {name}; fix in ds41f-mlx and promote')
        if bool(p.stat().st_mode & 0o111) != (entry['mode'] == '100755'):
            raise ValueError(f'source mode drift: {name}')
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if str(rel) in m['files'] or str(rel) == 'release/promotion.json':
            continue
        if any(part == '__pycache__' for part in rel.parts) or rel.parts[0] in m['nondeterministic_artifacts']:
            continue
        raise ValueError(f'unexpected projected material: {rel}')
    return m


def main():
    m = verify()
    print(json.dumps(dict(status='PASS', source_commit=m['source_commit'], reference=m['reference'],
                         source_tree_sha256=m['source_tree_sha256'], qualification=m['qualification']), indent=2))


if __name__ == '__main__':
    main()
