#!/usr/bin/env python3
"""Model-free documentation role/edge and retained provenance containment checks."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {'CURRENT', 'HISTORICAL', 'EVIDENCE', 'RELEASE'}
RELATIONS = {'history', 'evidence', 'source-edition', 'navigation'}
# Do not allow a map edit to demote selected owners merely to evade current checks.
REQUIRED_CURRENT = {'README.md', 'docs/README.md', 'docs/doc-classification.json'} | {
    f'docs/{name}.md' for name in (
        'api', 'architecture', 'attention', 'capabilities', 'chat-interface',
        'correctness', 'documentation-policy', 'dynamic-capability-budget', 'engram',
        'final-runtime-target', 'generation-policy-frontends',
        'generation-policy-investigation', 'generation', 'implementation-plan',
        'moe-hc', 'mtp-architecture-reset', 'mtp-local-release-candidate',
        'mtp-production-serving', 'operations', 'operator-chat',
        'operator-live-observability', 'ordinary-api-boundary', 'pdf-document-workflow',
        'performance', 'private-lan', 'production-mlx-prefill-substrate',
        'production-serving-parity-closure', 'production-sessions', 'provenance',
        'qualification', 'reference-release-and-promotion-strategy',
        'reference-release-r1', 'release-qualification', 'runtime-admission',
        'runtime-strategy', 'scaffold-classification', 'session-state',
        'stateful-live-delivery', 'web-application',
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


def documentation_files(root: Path) -> set[str]:
    return {'README.md'} | {
        p.relative_to(root).as_posix() for p in (root / 'docs').rglob('*') if p.is_file()
    }


def links(text: str):
    """Literal inline links plus reviewed paragraph identity; source TXT stays inert."""
    for paragraph in re.split(r'\n[ \t]*\n', text):
        digest = hashlib.sha256(paragraph.strip().encode()).hexdigest()
        for match in LINK_RE.finditer(paragraph):
            raw = match[1].strip()
            if raw.startswith('<'):
                target = raw[1:raw.index('>')]
            else:
                target = raw.split()[0]
            yield target, digest


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
    require(cmap.get('schema') == 'ds41f.docs_classification.v2', 'unsupported classification schema')
    entries = {}
    for entry in cmap['entries']:
        path = entry['path']
        require(path not in entries, f'duplicate classification: {path}')
        require(not Path(path).is_absolute() and Path(path).as_posix() == path and '..' not in Path(path).parts,
                f'noncanonical classification path: {path}')
        contained(root, root / path)
        require(not (root / path).is_symlink(), f'symlink classification alias unsupported: {path}')
        require(entry.get('classification') in CLASSES, f'unknown classification: {path}')
        require(bool(entry.get('reason', '').strip()), f'missing classification reason: {path}')
        entries[path] = entry
    expected = documentation_files(root)
    require(set(entries) == expected,
            f'classification coverage mismatch: missing={sorted(expected - entries.keys())}; stale={sorted(entries.keys() - expected)}')
    for path in REQUIRED_CURRENT:
        require(path in entries and entries[path]['classification'] == 'CURRENT', f'required CURRENT owner missing/demoted: {path}')
    for path, entry in entries.items():
        if path.startswith('docs/archive/'):
            require(entry['classification'] != 'CURRENT', f'archive cannot be CURRENT: {path}')
        if entry['classification'] == 'CURRENT':
            require(Path(path).suffix in {'.md', '.json'}, f'inert source cannot be CURRENT: {path}')
            require(entry.get('role') in {'contract', 'plan', 'policy', 'navigation', 'guide'}, f'invalid CURRENT role: {path}')
            if entry.get('role') == 'navigation':
                require(path in {'README.md', 'docs/README.md'}, f'contract cannot be relabeled navigation: {path}')

    registered = {}
    for edge in cmap['noncurrent_references']:
        key = (edge['source'], edge['target'], edge['context_sha256'])
        require(key not in registered, f'duplicate noncurrent reference: {key}')
        require(edge.get('relationship') in RELATIONS, f'forbidden normative/unknown noncurrent relationship: {key}')
        require(bool(edge.get('reason', '').strip()), f'missing noncurrent review reason: {key}')
        require(re.fullmatch(r'[0-9a-f]{64}', edge['context_sha256']) is not None, f'invalid context hash: {key}')
        require(edge['source'] in entries and entries[edge['source']]['classification'] == 'CURRENT', f'noncurrent reference source not CURRENT: {key}')
        if edge['relationship'] == 'navigation':
            require(entries[edge['source']].get('role') == 'navigation', f'navigation edge from non-navigation owner: {key}')
        registered[key] = edge

    used = set()
    anchor_cache = {}
    count = 0
    for source in sorted(expected):
        if Path(source).suffix != '.md':
            continue
        text = (root / source).read_text()
        current = entries[source]['classification'] == 'CURRENT'
        if current:
            require(not re.search(r'^\s*\[[^\]]+\]:|<[^>]+\bhref\s*=|\]\s*\[', text, re.M | re.I),
                    f'unsupported reference-style/HTML link in CURRENT; use reviewed inline links: {source}')
            for token in OLD_REPO_LIVE_TOKENS:
                require(token not in text, f'live old-repo dependency wording in {source}: {token}')
        for target, digest in links(text):
            url = urlsplit(target)
            if url.scheme or url.netloc:
                require(url.scheme != 'file', f'file URI bypasses repository containment: {source} -> {target}')
                continue
            resolved = contained(root, (root / source).parent / unquote(url.path)) if url.path else root / source
            require(resolved.exists(), f'broken markdown link: {source} -> {target}')
            dest = resolved.relative_to(root).as_posix()
            dest_current = dest in entries and entries[dest]['classification'] == 'CURRENT'
            if url.fragment and resolved.suffix == '.md' and (current or dest_current):
                if dest not in anchor_cache:
                    anchor_cache[dest] = anchors(resolved.read_text())
                require(unquote(url.fragment) in anchor_cache[dest], f'broken current-boundary anchor: {source} -> {target}')
            if current and ((dest in entries and not dest_current) or dest.startswith('artifacts/')):
                key = (source, target, digest)
                require(key in registered, f'unreviewed noncurrent dependency/reference: {source} -> {target}')
                edge = registered[key]
                if edge['relationship'] == 'source-edition':
                    require(dest.startswith('docs/archive/editorial-baseline/') and resolved.suffix == '.txt'
                            and entries[dest]['classification'] == 'EVIDENCE', f'invalid literal source-edition target: {key}')
                used.add(key)
            elif current and dest.startswith('docs/') and dest not in entries:
                raise ValueError(f'unclassified documentation target: {source} -> {target}')
            count += 1
    require(used == registered.keys(), f'stale noncurrent references: {sorted(registered.keys() - used)}')
    return {'classified_files': len(entries), 'current_owners': sum(e['classification'] == 'CURRENT' for e in entries.values()),
            'live_links': count, 'reviewed_noncurrent_references': len(used)}


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
    print(f'PASS repository self-containment: {counts}; role/context/containment and retained provenance verified (not qualification)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
