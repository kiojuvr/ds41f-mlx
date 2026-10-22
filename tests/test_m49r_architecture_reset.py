"""Historical reconstruction integrity, not a freeze of today's implementation.

Current owner/resource/transaction invariants live in the MTP, execution identity,
state producer and target transaction tests.
"""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = '5e784d9e83a85c497cd192f87184e9b01a1343eb'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def test_reconstruction_archive_against_recorded_historical_authority():
    manifest = json.loads((ROOT / 'docs/archive/mtp-reconstruction/source-manifest.json').read_text())
    assert manifest['restart_commit'] == BASE
    assert len(manifest['entries']) == 57
    for entry in manifest['entries']:
        data = (ROOT / entry['archive_path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry['sha256']
        assert data == git('show', f"{manifest['source_commit']}:{entry['original_path']}")


def test_no_archived_experimental_producer_reintroduced():
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
