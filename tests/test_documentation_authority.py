"""Model-free positive/negative documentation boundary regressions."""
import hashlib
import json
import pytest

from tools import check_repository_self_containment as checker


@pytest.fixture
def repo(tmp_path):
    rows = []
    for name in sorted(checker.REQUIRED_CURRENT):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# Owner\n')
        rows.append(dict(path=name, classification='CURRENT',
                         role='navigation' if name in {'README.md', 'docs/README.md'} else 'contract', reason='Reviewed owner'))
    for name, kind in [('docs/archive/history.md', 'HISTORICAL'),
                       ('docs/report.md', 'EVIDENCE'),
                       ('docs/archive/editorial-baseline/old.md.txt', 'EVIDENCE')]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# Historical\n\n[Inert](missing.md)\n' if name.endswith('.txt') else '# Historical\n')
        rows.append(dict(path=name, classification=kind, role='history', reason='Original evidence'))
    return tmp_path, dict(schema='ds41f.docs_classification.v2', entries=rows, noncurrent_references=[])


def save(repo):
    root, cmap = repo
    (root / 'docs/doc-classification.json').write_text(json.dumps(cmap))
    return root


def add_reference(repo, *, source='docs/api.md', target='archive/history.md', relationship='history'):
    root, cmap = repo
    paragraph = f'Historical M1 evidence only: [Original]({target}).'
    (root / source).write_text(paragraph + '\n')
    edge = dict(source=source, target=target, relationship=relationship,
                context_sha256=hashlib.sha256(paragraph.encode()).hexdigest(), reason='Input-bound observation only')
    cmap['noncurrent_references'].append(edge)
    return edge


def test_actual_repository_closure():
    assert checker.check_documentation(checker.ROOT)['reviewed_noncurrent_references'] > 0
    checker.check_retained_provenance(checker.ROOT)


@pytest.mark.parametrize('target,relation', [('archive/history.md', 'history'),
                                           ('report.md', 'evidence'),
                                           ('archive/editorial-baseline/old.md.txt', 'source-edition')])
def test_reviewed_evidence_and_inert_editions(repo, target, relation):
    add_reference(repo, target=target, relationship=relation)
    assert checker.check_documentation(save(repo))['reviewed_noncurrent_references'] == 1


def test_current_delegation_and_anchor(repo):
    root, _ = repo
    (root / 'docs/api.md').write_text('[Contract](architecture.md#owner)\n')
    checker.check_documentation(save(repo))


@pytest.mark.parametrize('target', ['archive/history.md', 'report.md', '../artifacts/receipt.md'])
def test_undeclared_noncurrent_reference_rejected(repo, target):
    root, _ = repo
    (root / 'artifacts').mkdir()
    (root / 'artifacts/receipt.md').write_text('# Receipt\n')
    add_reference(repo, target=target)
    repo[1]['noncurrent_references'].clear()
    with pytest.raises(ValueError, match='unreviewed noncurrent'):
        checker.check_documentation(save(repo))


def test_artifact_evidence_also_requires_context(repo):
    root, _ = repo
    (root / 'artifacts').mkdir()
    (root / 'artifacts/receipt.md').write_text('# Receipt\n')
    add_reference(repo, target='../artifacts/receipt.md', relationship='evidence')
    checker.check_documentation(save(repo))


@pytest.mark.parametrize('relation', ['normative', 'allow', '', 'startup'])
def test_normative_or_unknown_relation_rejected(repo, relation):
    add_reference(repo, relationship=relation)
    with pytest.raises(ValueError, match='forbidden normative/unknown'):
        checker.check_documentation(save(repo))


def test_context_cannot_silently_become_current_instruction(repo):
    root, _ = repo
    add_reference(repo)
    (root / 'docs/api.md').write_text('Required current startup: [Original](archive/history.md).\n')
    with pytest.raises(ValueError, match='unreviewed noncurrent'):
        checker.check_documentation(save(repo))


@pytest.mark.parametrize('mutation,match', [
    ('duplicate-edge', 'duplicate noncurrent'), ('stale-edge', 'stale noncurrent'),
    ('bad-hash', 'invalid context hash'), ('no-review-reason', 'missing noncurrent review'),
    ('wrong-source-edition', 'invalid literal source-edition'),
    ('navigation-from-contract', 'navigation edge from non-navigation'),
])
def test_reference_registry_fail_closed(repo, mutation, match):
    root, cmap = repo
    edge = add_reference(repo)
    if mutation == 'duplicate-edge': cmap['noncurrent_references'].append(dict(edge))
    elif mutation == 'stale-edge': (root / 'docs/api.md').write_text('# Owner\n')
    elif mutation == 'bad-hash': edge['context_sha256'] = 'observed'
    elif mutation == 'no-review-reason': edge['reason'] = ''
    elif mutation == 'wrong-source-edition': edge['relationship'] = 'source-edition'
    elif mutation == 'navigation-from-contract': edge['relationship'] = 'navigation'
    with pytest.raises(ValueError, match=match):
        checker.check_documentation(save(repo))


def test_archive_cannot_become_current(repo):
    row = next(e for e in repo[1]['entries'] if e['path'] == 'docs/archive/history.md')
    row['classification'] = 'CURRENT'
    with pytest.raises(ValueError, match='archive cannot be CURRENT'):
        checker.check_documentation(save(repo))


def test_navigation_is_not_normative(repo):
    add_reference(repo, source='docs/README.md', relationship='navigation')
    checker.check_documentation(save(repo))


@pytest.mark.parametrize('mutation,match', [
    ('duplicate', 'duplicate classification'), ('missing', 'coverage mismatch'),
    ('stale', 'coverage mismatch'), ('unknown', 'unknown classification'),
    ('demote', 'required CURRENT owner'), ('escape', 'noncanonical classification'),
    ('no-reason', 'missing classification reason'), ('navigation', 'contract cannot be relabeled'),
])
def test_classification_fail_closed(repo, mutation, match):
    _, cmap = repo
    row = next(e for e in cmap['entries'] if e['path'] == 'docs/api.md')
    if mutation == 'duplicate': cmap['entries'].append(dict(row))
    elif mutation == 'missing': cmap['entries'].remove(row)
    elif mutation == 'stale': cmap['entries'].append(dict(path='docs/gone.md', classification='EVIDENCE', reason='Missing'))
    elif mutation == 'unknown': row['classification'] = 'CANONICAL'
    elif mutation == 'demote': row['classification'] = 'HISTORICAL'
    elif mutation == 'escape': row['path'] = '../outside.md'
    elif mutation == 'no-reason': row['reason'] = ''
    elif mutation == 'navigation': row['role'] = 'navigation'
    with pytest.raises(ValueError, match=match):
        checker.check_documentation(save(repo))


@pytest.mark.parametrize('target,match', [('../../outside.md', 'escapes repository'),
                                        ('gone.md', 'broken markdown link'),
                                        ('architecture.md#gone', 'broken current-boundary anchor'),
                                        ('file:///outside.md', 'file URI bypasses')])
def test_links_fail_closed(repo, target, match):
    (repo[0] / 'docs/api.md').write_text(f'[Bad]({target})\n')
    with pytest.raises(ValueError, match=match):
        checker.check_documentation(save(repo))


def test_historical_incoming_current_anchor_is_checked(repo):
    (repo[0] / 'docs/archive/history.md').write_text('[Owner](../api.md#missing)\n')
    with pytest.raises(ValueError, match='broken current-boundary anchor'):
        checker.check_documentation(save(repo))


@pytest.mark.parametrize('text', ['[Old][ref]\n\n[ref]: archive/history.md',
                                 '<a href="archive/history.md">Old</a>', '<area href="archive/history.md">'])
def test_unsupported_link_syntax_cannot_bypass_registry(repo, text):
    (repo[0] / 'docs/api.md').write_text(text)
    with pytest.raises(ValueError, match='unsupported reference-style/HTML'):
        checker.check_documentation(save(repo))


def test_symlink_escape_is_rejected(repo, tmp_path):
    outside = tmp_path.parent / (tmp_path.name + '-outside.md')
    outside.write_text('# Outside\n')
    target = tmp_path / 'docs/report.md'
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(ValueError, match='escapes repository'):
        checker.check_documentation(save(repo))


def test_internal_symlink_cannot_alias_archive_into_current(repo):
    root, _ = repo
    path = root / 'docs/api.md'
    path.unlink()
    path.symlink_to(root / 'docs/archive/history.md')
    with pytest.raises(ValueError, match='symlink classification alias'):
        checker.check_documentation(save(repo))


def test_live_old_repository_instruction_is_still_rejected(repo):
    (repo[0] / 'docs/api.md').write_text('inspect the old repo\n')
    with pytest.raises(ValueError, match='live old-repo dependency'):
        checker.check_documentation(save(repo))


@pytest.mark.parametrize('fault', ['origin', 'dependency', 'cmake', 'import', 'reconciliation'])
def test_retained_native_provenance_guards(repo, fault):
    root, _ = repo
    provenance = ('deepseek-v41-flash-mlx@1b7d0a2c7d33602437dffd44e26a33f39f189661\n'
                  'runtime source dependency: none\nbuild dependency: none\ntest dependency: none\n')
    if fault == 'origin': provenance = provenance.replace('1b7d0a2', 'wrong')
    if fault == 'dependency': provenance = provenance.replace('build dependency: none', 'missing')
    (root / 'docs/provenance.md').write_text(provenance)
    (root / 'native').mkdir()
    (root / 'native/CMakeLists.txt').write_text('benchmark' if fault == 'cmake' else '')
    (root / 'artifacts/provenance').mkdir(parents=True)
    imports = [{'destination_path': 'native/missing.cpp'}] if fault == 'import' else []
    (root / 'artifacts/provenance/legacy-import.json').write_text(json.dumps({'imports': imports}))
    if fault == 'reconciliation':
        (root / 'artifacts/legacy-evidence-boundary-reconciliation.json').write_text(json.dumps(
            {'local_legacy_import_resolution': {'entries': [{'local_path': 'native/missing.cpp'}]}}))
    with pytest.raises(ValueError):
        checker.check_retained_provenance(root)
