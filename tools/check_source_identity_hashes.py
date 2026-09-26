#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.source_identity import DEFAULT_CHECKPOINT, SOURCE_HASH_METHOD, official_source_path, source_sha256


def walk_sources(obj: Any, context_file: str | None = None):
    if isinstance(obj, dict):
        cf = obj.get('file', context_file)
        if 'source_lines' in obj and 'source_sha256' in obj and cf:
            yield obj, str(cf)
        for v in obj.values():
            yield from walk_sources(v, cf)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_sources(v, context_file)


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=str(DEFAULT_CHECKPOINT)); ap.add_argument('paths',nargs='*',default=['artifacts/*.json']); args=ap.parse_args(); ck=Path(args.checkpoint)
    failures=[]
    files=[]
    for pat in args.paths:
        matches=sorted(Path().glob(pat)) if any(ch in pat for ch in '*?[]') else [Path(pat)]
        files.extend(matches)
    for p in files:
        if not p.exists() or p.suffix!='.json': continue
        try: data=json.loads(p.read_text())
        except Exception: continue
        for entry,file_name in walk_sources(data):
            lines=entry.get('source_lines')
            if not (isinstance(lines,list) and len(lines)==2): continue
            src=official_source_path(file_name,ck)
            if not src.exists(): continue
            exp=source_sha256(src,int(lines[0]),int(lines[1]))
            if entry.get('source_hash_method')!=SOURCE_HASH_METHOD:
                failures.append(f"{p}: {file_name}:{lines}: source_hash_method={entry.get('source_hash_method')!r}, expected {SOURCE_HASH_METHOD}")
            if entry.get('source_sha256')!=exp:
                failures.append(f"{p}: {file_name}:{lines}: source_sha256={entry.get('source_sha256')}, expected {exp}")
    if failures:
        print('source-identity hash check failed:')
        for f in failures[:200]: print(f)
        if len(failures)>200: print(f'... {len(failures)-200} more')
        return 1
    print('source-identity hash check passed')
    return 0
if __name__=='__main__': raise SystemExit(main())
