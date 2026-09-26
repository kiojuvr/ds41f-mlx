#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, hashlib, json, sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.source_identity import DEFAULT_CHECKPOINT, SOURCE_HASH_METHOD, file_sha256, official_source_path, source_sha256

ARTIFACTS = Path('artifacts')
TARGETS = [
    ('inference/model.py', (613, 789), 'Attention'),
    ('inference/model.py', (429, 485), 'Compressor'),
    ('inference/model.py', (488, 580), 'Indexer'),
    ('inference/kernel.py', (392, 403), 'sparse_attn'),
    ('inference/kernel.py', (184, 204), 'fp4_act_quant'),
]


def ast_digest(path: Path, start: int, end: int) -> str | None:
    text = path.read_text()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and getattr(node, 'lineno', None) == start and getattr(node, 'end_lineno', None) == end:
            seg = ast.get_source_segment(text, node)
            if seg is not None:
                return hashlib.sha256(seg.encode('utf-8')).hexdigest()
    return None


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


def patch_obj(obj: Any, checkpoint: Path) -> tuple[int, list[dict[str, Any]]]:
    changed = 0; records = []
    for entry, file_name in walk_sources(obj):
        lines = entry.get('source_lines')
        if not (isinstance(lines, list) and len(lines) == 2):
            continue
        path = official_source_path(file_name, checkpoint)
        if not path.exists():
            continue
        expected = source_sha256(path, int(lines[0]), int(lines[1]))
        old = entry.get('source_sha256')
        method = entry.get('source_hash_method')
        ast_sha = ast_digest(path, int(lines[0]), int(lines[1]))
        classification = 'canonical_match' if old == expected else ('ast_get_source_segment_without_final_newline' if old == ast_sha else 'other_or_unknown')
        records.append({'file': file_name, 'source_lines': lines, 'old_source_sha256': old, 'canonical_source_sha256': expected, 'old_source_hash_method': method, 'classification': classification})
        if old != expected:
            entry['source_sha256'] = expected; changed += 1
        if entry.get('source_hash_method') != SOURCE_HASH_METHOD:
            entry['source_hash_method'] = SOURCE_HASH_METHOD; changed += 1
    return changed, records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT))
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--out', default='artifacts/source-identity-hash-audit.json')
    args = ap.parse_args(); checkpoint = Path(args.checkpoint)

    audit: dict[str, Any] = {
        'schema': 'ds41f.source-identity-hash-audit.v1',
        'classification': 'source_identity_metadata_audit_no_model_math',
        'checkpoint': str(checkpoint),
        'source_hash_method': SOURCE_HASH_METHOD,
        'targets': [],
        'artifact_updates': [],
        'unresolved_mismatch_count': 0,
        'numerical_expected_values_changed': False,
        'ok': True,
    }
    for file_name, span, name in TARGETS:
        path = official_source_path(file_name, checkpoint)
        raw = source_sha256(path, *span); ast_sha = ast_digest(path, *span)
        audit['targets'].append({'name': name, 'file': file_name, 'file_sha256': file_sha256(path), 'source_lines': list(span), 'canonical_raw_inclusive_sha256': raw, 'ast_get_source_segment_sha256': ast_sha, 'cause': 'hash method difference: raw inclusive line span includes final line ending; ast.get_source_segment omits it' if ast_sha and ast_sha != raw else 'single method observed'})

    groups = defaultdict(set)
    for p in sorted(ARTIFACTS.glob('*.json')):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        before_expected = json.dumps(data.get('expected', None), sort_keys=True, separators=(',', ':'))
        before_digests = json.dumps(data.get('digests', None), sort_keys=True, separators=(',', ':'))
        changed, records = patch_obj(data, checkpoint)
        for r in records:
            groups[(r['file'], tuple(r['source_lines']), SOURCE_HASH_METHOD)].add(r['canonical_source_sha256'])
        if changed:
            after_expected = json.dumps(data.get('expected', None), sort_keys=True, separators=(',', ':'))
            after_digests = json.dumps(data.get('digests', None), sort_keys=True, separators=(',', ':'))
            audit['artifact_updates'].append({'path': str(p), 'metadata_fields_changed': changed, 'expected_unchanged': before_expected == after_expected, 'digests_unchanged': before_digests == after_digests, 'records': records})
            if before_expected != after_expected or before_digests != after_digests:
                audit['numerical_expected_values_changed'] = True
            if args.write:
                p.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')

    collisions = [{'file': k[0], 'source_lines': list(k[1]), 'source_hash_method': k[2], 'digests': sorted(v)} for k, v in groups.items() if len(v) > 1]
    audit['unresolved_collisions'] = collisions
    audit['unresolved_mismatch_count'] = len(collisions)
    audit['ok'] = not collisions and not audit['numerical_expected_values_changed']
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(audit, indent=2, sort_keys=True) + '\n')
    print(out)
    return 0 if audit['ok'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
