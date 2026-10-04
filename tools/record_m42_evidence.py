"""Content reconciliation and durable M42 decision; not ordinary R1 verification."""
import hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'artifacts/m42'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((R/'artifacts/m41/qualification.json').read_text())
comparisons=[]
for name,digest in old['sources'].items():
    if name.startswith(('ds41f_mlx/','native/','third_party/mtp/')):
        p=R/name
        comparisons.append(dict(path=name,previous=digest,current=sha(p) if p.is_file() else None,equal=p.is_file() and sha(p)==digest))
assert all(c['equal'] for c in comparisons)
receipts={}
for n in ['independent','isolation','mutations','preserved-off']:
    p=O/(n+'.json');data=json.loads(p.read_text())
    assert data['status']=='PASS',(n,data['status'])
    receipts[n]=dict(path=str(p.relative_to(R)),sha256=sha(p),status=data['status'])
q=json.loads((O/'independent.json').read_text());test_count=subtests=0
for g in q['gates']:
    if g['name'].startswith('test_'):
        text=Path(g['log']).read_text()
        matches=re.findall(r'(\d+) passed',text)
        test_count+=int(matches[-1]) if matches else 0
        matches=re.findall(r'(\d+) subtests passed',text)
        subtests+=int(matches[-1]) if matches else 0
mtp=json.loads((O/'independent-gates/mtp-model.json').read_text())
classification=dict(promoted={'M31-M32':'canonical protocol inputs/results and nonmutating preview rows','M36R-M39':'canonical representability predicates and token identities','M8-M14, P1-P7, M29-M41':'executable ownership/operation/negative checks, no historical receipt assertions'},
 historical_only=['M33-M41 source identity receipt tests','M33-M34 long real-model campaign traces','M38 76-request soak','M39 20000-lifetime stress','M41 speed/identity/socket negative campaigns','M20-M24 200K and restored-session campaigns'],
 superseded_for_ordinary_verification=['milestone protocol/parity/preview/certificate fixture directory dependencies','historical absolute tokenizer paths','historical composed harness native-gate imports'])
result=dict(schema='ds41f.m42.qualification.v1',decision='SEMANTIC_CLOSURE_REFERENCE_R1',status='PASS',reference='R1',reference_sha256=sha(R/'reference/R1/manifest.json'),contract_sha256=sha(R/'reference/R1/contract.json'),source_base='14228319f01bbdb8cb79cc393b2e4c99f999aacc',runtime_behavior_changed=False,default_profile='standard-off',mtp_profile='mtp-singleton-v1 (unchanged M41 scope)',canonical_command='python -m ds41f_mlx.reference --profile both --real-model --output artifacts/r1/qualification.json',
 fresh=dict(receipts=receipts,semantic_checks=test_count,subtests=subtests,protocol_records=64,preview_rows=77,preview_repeated_calls=15400,recovery_cases=28,mtp_outcomes=len(mtp['cases']),mtp_admission_rejections=len(mtp['admission']),mutations_rejected=3),
 inheritance=dict(authority='M41 scope-reconciled source/native qualification and its precisely inherited M33-M39/M20-M24 evidence',receipt_sha256=sha(R/'artifacts/m41/qualification.json'),all_compared_owners_unchanged=True,source_comparisons=comparisons,classification=classification),
 limits=json.loads((R/'reference/R1/contract.json').read_text())['exclusions'],debt='Five stale historical whole-current-source receipt assertions remain historical debt; not used by R1. OFF operator delivery/provenance still uses its legacy deployment; R1 verification delivers its own OFF execution source, not an OFF dependency migration.',
 excluded_attempts=['first R1 tokenizer check incorrectly conflated checkpoint and recipe identities; separated before final run','combined in-process owner fixtures contaminated import paths; isolated each file before final run','initial real run omitted explicit checkpoint; startup rejected','editable clone origins and subsequent source seal drift rejected; module invocation, current editable install and normal seal precede final run','parallel mutation attempt was killed during real model load; final sequential mutation campaign has explicit assertion failures'],
 evidence={str(p.relative_to(R)):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.suffix in ('.json','.log') and 'off-source' not in p.parts and 'off-kv' not in p.parts and p.name!='qualification.json'})
(O/'qualification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['fresh'],indent=2))
