#!/usr/bin/env python3
"""Model-free live documentation links, current owners and retained provenance."""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CURRENT = {'README.md', 'docs/README.md', 'docs/doc-classification.json'} | {
    f'docs/{name}.md' for name in (
        'architecture', 'capabilities', 'correctness', 'documentation-policy',
        'mtp-production-serving', 'operations', 'operator-chat', 'production-sessions',
        'provenance', 'qualification', 'runtime-admission', 'session-state',
    )
}
OLD_REPO_LIVE_TOKENS = (
    '/Volumes/SDXC-512/deepseek-v41-flash-mlx', '../deepseek-v41-flash-mlx',
    'inspect the old repo', 'look at the old repo', 'live qualification oracle',
)
LINK_RE = re.compile(r'(?<!!)\[[^\]]*\]\(([^)\n]+)\)')


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def contained(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    require(resolved.is_relative_to(root.resolve()), f'path escapes repository: {path}')
    return resolved


def anchors(text: str) -> set[str]:
    result = set(re.findall(r'''<a\s+(?:id|name)=["']([^"']+)''', text))
    counts: dict[str, int] = {}
    fence = None
    for line in text.splitlines():
        marker = re.match(r'^\s*(`{3,}|~{3,})', line)
        if marker:
            if fence is None:
                fence = marker[1][0]
            elif marker[1][0] == fence:
                fence = None
            continue
        if fence is not None:
            continue
        heading = re.match(r'^#{1,6}\s+(.+?)\s*#*$', line)
        if not heading:
            continue
        value = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', heading[1]).lower()
        slug = ''.join(c for c in value if c in '-_ ' or unicodedata.category(c)[0] in 'LN').replace(' ', '-')
        n = counts.get(slug, 0)
        counts[slug] = n + 1
        result.add(slug if not n else f'{slug}-{n}')
    return result


def check_documentation(root: Path) -> dict[str, int]:
    root = root.resolve()
    cmap = json.loads((root / 'docs/doc-classification.json').read_text())
    require(cmap.get('schema') == 'ds41f.docs_classification.v3', 'unsupported classification schema')
    require(cmap.get('default') == 'EVIDENCE', 'unlisted documents must be source-scoped evidence')
    current = set(cmap['current'])
    require(len(current) == len(cmap['current']), 'duplicate current owner')
    require(REQUIRED_CURRENT <= current, f'required CURRENT owner missing: {sorted(REQUIRED_CURRENT - current)}')
    for path in current:
        require(not Path(path).is_absolute() and '..' not in Path(path).parts and Path(path).as_posix() == path,
                f'noncanonical owner path: {path}')
        target = contained(root, root / path)
        require(not (root / path).is_symlink(), f'symlink owner alias unsupported: {path}')
        require(target.is_file(), f'stale current owner: {path}')
        require(not path.startswith('docs/archive/'), f'archive cannot be CURRENT: {path}')
        require(Path(path).suffix in {'.md', '.json'}, f'inert source cannot be CURRENT: {path}')
    sources = {root / 'README.md', root / 'tools/README.md'} | set((root / 'docs').rglob('*.md'))
    count = 0
    anchor_cache = {}
    for source in sorted(sources):
        contained(root, source)
        name = source.relative_to(root).as_posix()
        text = source.read_text()
        if name in current:
            require(not re.search(r'^\s*\[[^\]]+\]:|<[^>]+\bhref\s*=|\]\s*\[', text, re.M | re.I),
                    f'unsupported reference-style/HTML link in CURRENT; use inline links: {name}')
            for token in OLD_REPO_LIVE_TOKENS:
                require(token not in text, f'live old-repo dependency wording in {name}: {token}')
        for match in LINK_RE.finditer(text):
            raw = match[1].strip()
            target = raw[1:raw.index('>')] if raw.startswith('<') else raw.split()[0]
            url = urlsplit(target)
            if url.scheme or url.netloc:
                require(url.scheme != 'file', f'file URI bypasses repository containment: {name} -> {target}')
                continue
            resolved = contained(root, source.parent / unquote(url.path)) if url.path else source
            require(resolved.exists(), f'broken markdown link: {name} -> {target}')
            dest = resolved.relative_to(root).as_posix()
            # Historical internal anchors are reproduction data; current boundaries are live.
            if url.fragment and resolved.suffix == '.md' and (name in current or dest in current):
                if dest not in anchor_cache:
                    anchor_cache[dest] = anchors(resolved.read_text())
                require(unquote(url.fragment) in anchor_cache[dest], f'broken current-boundary anchor: {name} -> {target}')
            count += 1
    return {'current_owners': len(current), 'markdown_files': len(sources), 'live_links': count}


def check_retained_provenance(root: Path) -> None:
    provenance = (root / 'docs/provenance.md').read_text()
    require('deepseek-v41-flash-mlx@1b7d0a2c7d33602437dffd44e26a33f39f189661' in provenance,
            'provenance must record historical native source import identity')
    for phrase in ('runtime source dependency: none', 'build dependency: none', 'test dependency: none'):
        require(phrase in provenance, f'provenance missing scoped dependency statement: {phrase}')
    cmake = (root / 'native/CMakeLists.txt').read_text()
    for forbidden in ('context-ladder', 'wide-attention', 'packed-moe-probe', 'benchmark'):
        require(forbidden not in cmake, f'benchmark/diagnostic target leaked into native CMake: {forbidden}')
    manifest = json.loads((root / 'artifacts/provenance/legacy-import.json').read_text())
    for imp in manifest.get('imports', []):
        dest = imp.get('destination_path')
        if dest and imp.get('import_mode') != 'not_imported':
            require(contained(root, root / dest).exists(), f'imported provenance destination missing: {dest}')
    recon = root / 'artifacts/legacy-evidence-boundary-reconciliation.json'
    if recon.exists():
        for entry in json.loads(recon.read_text()).get('local_legacy_import_resolution', {}).get('entries', []):
            local = entry.get('local_path')
            if local:
                require(contained(root, root / local).exists(), f'legacy proof graph local path missing: {local}')


def main() -> int:
    try:
        counts = check_documentation(ROOT)
        check_retained_provenance(ROOT)
    except (ValueError, KeyError, OSError) as exc:
        print(f'FAIL: {exc}')
        return 1
    print(f'PASS repository self-containment: {counts}; live links/owners and retained provenance verified (not qualification)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
