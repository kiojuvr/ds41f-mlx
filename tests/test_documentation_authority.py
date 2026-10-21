"""Model-free current-owner/link and retained provenance regressions."""
import json
import pytest

from tools import check_repository_self_containment as checker


@pytest.fixture
def repo(tmp_path):
    for name in sorted(checker.REQUIRED_CURRENT | {'tools/README.md'}):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# Owner\n')
    (tmp_path / 'docs/report.md').write_text('# Historical\n')
    return tmp_path, dict(schema='ds41f.docs_classification.v3', default='EVIDENCE',
                          current=sorted(checker.REQUIRED_CURRENT))


def save(repo):
    root, cmap = repo
    (root / 'docs/doc-classification.json').write_text(json.dumps(cmap))
    return root


def test_actual_repository_closure():
    assert checker.check_documentation(checker.ROOT)['live_links'] > 0
    checker.check_retained_provenance(checker.ROOT)


def test_evidence_needs_no_exhaustive_registry(repo):
    (repo[0] / 'docs/architecture.md').write_text('Recorded evidence only: [Report](report.md).\n')
    assert checker.check_documentation(save(repo))['live_links'] == 1


@pytest.mark.parametrize('mutation,match', [
    ('duplicate', 'duplicate current'), ('missing', 'required CURRENT owner'),
    ('stale', 'stale current'), ('escape', 'noncanonical owner'),
    ('archive', 'archive cannot be CURRENT'), ('default', 'source-scoped evidence'),
])
def test_owner_registry_fail_closed(repo, mutation, match):
    root, cmap = repo
    if mutation == 'duplicate': cmap['current'].append('docs/architecture.md')
    elif mutation == 'missing': cmap['current'].remove('docs/architecture.md')
    elif mutation == 'stale': cmap['current'].append('docs/gone.md')
    elif mutation == 'escape': cmap['current'].append('../outside.md')
    elif mutation == 'default': cmap['default'] = 'CURRENT'
    elif mutation == 'archive':
        (root / 'docs/archive').mkdir()
        (root / 'docs/archive/old.md').write_text('# Old\n')
        cmap['current'].append('docs/archive/old.md')
    with pytest.raises(ValueError, match=match):
        checker.check_documentation(save(repo))


@pytest.mark.parametrize('source', ['docs/architecture.md', 'docs/report.md', 'tools/README.md'])
@pytest.mark.parametrize('target,match', [('../../outside.md', 'escapes repository'),
                                        ('gone.md', 'broken markdown link'),
                                        ('../docs/architecture.md#gone', 'broken current-boundary anchor'),
                                        ('file:///outside.md', 'file URI bypasses')])
def test_live_links_fail_closed(repo, source, target, match):
    (repo[0] / source).write_text(f'[Bad]({target})\n')
    with pytest.raises(ValueError, match=match):
        checker.check_documentation(save(repo))


def test_current_anchor_and_inert_patch(repo):
    (repo[0] / 'docs/architecture.md').write_text('[Contract](capabilities.md#owner)\n')
    (repo[0] / 'docs/source.patch').write_text('[Not Markdown](missing.md)\n')
    checker.check_documentation(save(repo))


@pytest.mark.parametrize('text', ['[Old][ref]\n\n[ref]: report.md',
                                 '<a href="report.md">Old</a>', '<area href="report.md">'])
def test_unsupported_current_links_rejected(repo, text):
    (repo[0] / 'docs/architecture.md').write_text(text)
    with pytest.raises(ValueError, match='unsupported reference-style/HTML'):
        checker.check_documentation(save(repo))


def test_symlink_owner_alias_rejected(repo):
    path = repo[0] / 'docs/architecture.md'
    path.unlink()
    path.symlink_to(repo[0] / 'docs/report.md')
    with pytest.raises(ValueError, match='symlink owner alias'):
        checker.check_documentation(save(repo))


def test_live_old_repository_instruction_rejected(repo):
    (repo[0] / 'docs/architecture.md').write_text('inspect the old repo\n')
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
