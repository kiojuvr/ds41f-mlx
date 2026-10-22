#!/usr/bin/env python3
"""Verify recorded official-source identities, including explicit historical methods.

This checks reference evidence, not installed Runtime admission. Missing resources,
unknown methods and mismatches fail closed; no acceptance receipt is rewritten.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.source_identity import (
    DEFAULT_CHECKPOINT, SOURCE_HASH_METHOD, file_sha256, official_source_path, source_sha256,
)
HISTORICAL = ROOT/'tools/source_identity_historical_methods.json'


def walk_sources(obj: Any, context_file=None, context_hash=None):
    if isinstance(obj, dict):
        cf = obj.get('file', context_file)
        digest = obj.get('file_sha256', context_hash)
        if 'source_lines' in obj and 'source_sha256' in obj and cf:
            yield obj, str(cf), digest
        for value in obj.values():
            yield from walk_sources(value, cf, digest)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_sources(value, context_file, context_hash)


def verify_artifact(path, checkpoint, historical=None):
    data = json.loads(path.read_text())
    spec = (historical or {}).get(path.as_posix())
    if spec is not None:
        original = subprocess.check_output(['git', 'show', f"{spec['revision']}:{path.as_posix()}"], cwd=ROOT)
        if original != path.read_bytes():
            raise ValueError(f'{path}: historical artifact differs from original revision')
    count = 0
    for entry, file_name, digest in walk_sources(data):
        lines = entry['source_lines']
        if not (isinstance(lines, list) and len(lines) == 2 and all(type(n) is int for n in lines)):
            raise ValueError(f'{path}: invalid source_lines: {lines!r}')
        source = official_source_path(file_name, checkpoint)
        if digest is None and spec is not None:
            digest = spec.get('source_files', {}).get(file_name)
        if not digest or file_sha256(source) != digest:
            raise ValueError(f'{path}: whole-source identity mismatch: {file_name}')
        method = entry.get('source_hash_method')
        key = f'{file_name}:{lines[0]}:{lines[1]}'
        if method is None and spec is not None:
            method = spec['methods'].get(key)
        if method is None:
            raise ValueError(f'{path}: missing source_hash_method: {key}')
        if source_sha256(source, *lines, method=method) != entry['source_sha256']:
            raise ValueError(f'{path}: source_sha256 mismatch: {key} ({method})')
        count += 1
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('paths', nargs='*', default=['artifacts/*.json'])
    args = parser.parse_args(argv)
    historical = json.loads(HISTORICAL.read_text())['artifacts']
    files = set()
    for pattern in args.paths:
        matches = sorted(Path().glob(pattern)) if any(c in pattern for c in '*?[]') else [Path(pattern)]
        if not matches:
            parser.error(f'no artifacts matched: {pattern}')
        files.update(matches)
    failures, count = [], 0
    for path in sorted(files):
        try:
            count += verify_artifact(path, args.checkpoint, historical)
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
            failures.append(str(exc))
    if failures or not count:
        print('FAIL source identity: ' + '\n'.join(failures or ['no source assertions executed']))
        return 1
    print(f'PASS source identity: {count} source spans verified; whole-file identities and declared extraction methods checked')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
