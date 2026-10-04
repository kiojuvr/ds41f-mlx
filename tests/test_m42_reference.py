"""R1 identity/evidence regression; semantic execution lives in reference/R1."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OWN=ROOT/'reference/R1'
OUT=ROOT/'artifacts/m42'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text())


def test_reference_material_identity():
    m=load(OWN/'manifest.json')
    assert sha(ROOT/'ds41f_mlx/reference.py')==m['verifier_sha256']
    for name,digest in m['files'].items(): assert sha(OWN/name)==digest
    c=load(OWN/'contract.json')
    assert c['profiles']['standard-off']['default'] is True
    assert c['profiles']['mtp-singleton-v1']['default'] is False
    assert c['profiles']['mtp-singleton-v1']['persistence'] is False
    assert c['identities']['checkpoint_tokenizer_sha256']!=c['identities']['tokenizer_sha256']


def test_fresh_closure_and_inheritance():
    q=load(OUT/'qualification.json')
    assert q['decision']=='SEMANTIC_CLOSURE_REFERENCE_R1' and q['status']=='PASS'
    assert q['reference_sha256']==sha(OWN/'manifest.json')
    assert not q['runtime_behavior_changed']
    assert q['fresh']['semantic_checks']==249 and q['fresh']['mutations_rejected']==3
    for name,row in q['fresh']['receipts'].items():assert sha(ROOT/row['path'])==row['sha256']
    for path,digest in q['evidence'].items():assert sha(ROOT/path)==digest
    for row in q['inheritance']['source_comparisons']:
        assert row['equal'] and sha(ROOT/row['path'])==row['current']==row['previous']
    r=load(OUT/'independent.json')
    assert r['decision']=='CONFORMANT' and all(g['returncode']==0 for g in r['gates'])
    for path,digest in r['candidate_source'].items():assert sha(ROOT/path)==digest


def test_independence_and_semantic_rejection():
    isolation=load(OUT/'isolation.json')
    assert isolation['status']=='PASS' and isolation['restored']
    assert all(a['unavailable'] for a in isolation['authorities'])
    assert len(isolation['authorities'])>=16
    mutations=load(OUT/'mutations.json')
    for case in mutations['cases']:
        r=load(OUT/(case['name']+'.json'))
        assert r['status']=='FAIL' and 'drift' not in r['error']
        failed=next(g for g in r['gates'] if g['returncode'])
        log=OUT/(case['name']+'-gates')/(failed['name']+'.log')
        assert sha(log)==failed['sha256']
        text=log.read_text()
        assert 'DID NOT RAISE' in text or 'AssertionError' in text
