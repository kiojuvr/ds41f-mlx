"""M41 current qualified owners and durable evidence, not old-path runtime authority."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_m41_receipt_and_content_bindings():
    q=json.loads((ROOT/'artifacts/m41/qualification.json').read_text())
    assert q['decision']=='QUALIFIED_EXPLICIT_BOUNDED_LOCAL_MTP_RELEASE_CANDIDATE'
    assert q['default_profile']=='standard-off' and not q['public_release_blockers']
    for group in ('sources','evidence'):
        for path,expected in q[group].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
    for path,expected in q['inheritance']['unchanged_modules'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected
    x=json.loads((ROOT/'artifacts/m41/final-composed.json').read_text())
    assert x['status']=='PASS' and x['identity_sha256']==q['dependency_identity']
    assert len(x['cases'])==18 and x['effects']==4
    for row in x['cases']:
        m=row['outcome']['metrics'];c=m['settlement']
        assert m['aligned_idle'] and c['target_forwards']<=1
        assert c['new_verify_cycles']==c['new_proposals']==c['history_replay']==c['full_cache_repack']==0
    assert any(row['outcome']['outcome_state']=='unrecoverable' for row in x['cases'])
    manifest=json.loads((ROOT/'release/ds41f-release.json').read_text())
    assert manifest['runtime']['mtp']==manifest['runtime']['dspark']==manifest['runtime']['speculative_decode']=='OFF'
