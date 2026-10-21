"""M49R restart integrity outside the explicitly qualified M51R source delta.

The two connection files use their M51R evidence hashes instead of restart bytes.
All other runtime files and archived-producer exclusions keep the reset guard.
Physical/state/performance qualification is recorded separately in M51R receipts.
"""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = '5e784d9e83a85c497cd192f87184e9b01a1343eb'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def test_reconstruction_archive_and_restored_active_paths():
    manifest = json.loads((ROOT / 'docs/archive/mtp-reconstruction/source-manifest.json').read_text())
    assert manifest['restart_commit'] == BASE
    assert len(manifest['entries']) == 57
    baseline = set(git('ls-tree', '-r', '--name-only', BASE).decode().splitlines())
    for entry in manifest['entries']:
        data = (ROOT / entry['archive_path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry['sha256']
        assert data == git('show', f"{manifest['source_commit']}:{entry['original_path']}")
        active = ROOT / entry['original_path']
        if entry['original_path'] in baseline:
            assert active.read_bytes() == git('show', f"{BASE}:{entry['original_path']}")
        else:
            assert not active.exists()


def test_all_baseline_runtime_files_preserved_and_no_new_producer():
    allowed = {'ds41f_mlx/runtime/mtp_lifecycle.py', 'ds41f_mlx/serving/internal_mtp.py'}
    qualified = json.loads(gzip.decompress((ROOT / 'artifacts/m51r/qualification-identity.json.gz').read_bytes()))['identity']['runtime']
    for path in git('ls-tree', '-r', '--name-only', BASE, '--', 'ds41f_mlx').decode().splitlines():
        data = (ROOT / path).read_bytes()
        if path in allowed:
            assert hashlib.sha256(data).hexdigest() == qualified[path], path
        else:
            assert data == git('show', f'{BASE}:{path}'), path
    for name in ('accepted_prefix', 'dspark_proposal', 'hidden_taps', 'semantic_cycle'):
        assert not (ROOT / 'ds41f_mlx/runtime' / f'{name}.py').exists()


def test_reset_owner_and_retained_source_evidence_policy():
    classification = json.loads((ROOT / 'docs/doc-classification.json').read_text())
    assert classification['schema'] == 'ds41f.docs_classification.v3'
    assert classification['default'] == 'EVIDENCE'
    assert 'docs/mtp-architecture-reset.md' in classification['current']
    assert not any(path.startswith('docs/archive/') for path in classification['current'])
    # Literal reconstruction assets are still verified above. Superseded milestone
    # prose lives in Git, without requiring duplicate in-tree editions or receipts.
