"""Historical M41 qualified owners and durable evidence, not current approval."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_m41_receipt_and_content_bindings():
    q=json.loads((ROOT/'artifacts/m41/qualification.json').read_text())
    assert q['decision']=='QUALIFIED_EXPLICIT_BOUNDED_LOCAL_MTP_RELEASE_CANDIDATE'
    assert q['default_profile']=='standard-off' and not q['public_release_blockers']
    commit = subprocess.check_output(
        ['git', 'log', '-1', '--format=%H', '--', 'artifacts/m41/qualification.json'],
        cwd=ROOT, text=True).strip()
    assert commit
    for sources in [q['sources'], q['inheritance']['unchanged_modules']]:
        for path, expected in sources.items():
            source = ((ROOT/path).read_bytes() if path.startswith('artifacts/') else
                      subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT))
            assert hashlib.sha256(source).hexdigest() == expected, path
    for path, expected in q['evidence'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected, path
    x=json.loads((ROOT/'artifacts/m41/final-composed.json').read_text())
    assert x['status']=='PASS' and x['identity_sha256']==q['dependency_identity']
    assert len(x['cases'])==18 and x['effects']==4
    for row in x['cases']:
        m=row['outcome']['metrics'];c=m['settlement']
        assert m['aligned_idle'] and c['target_forwards']<=1
        assert c['new_verify_cycles']==c['new_proposals']==c['history_replay']==c['full_cache_repack']==0
    assert any(row['outcome']['outcome_state']=='unrecoverable' for row in x['cases'])
    manifest=json.loads(subprocess.check_output(
        ['git', 'show', f'{commit}:release/ds41f-release.json'], cwd=ROOT))
    assert manifest['runtime']['mtp']==manifest['runtime']['dspark']==manifest['runtime']['speculative_decode']=='OFF'
