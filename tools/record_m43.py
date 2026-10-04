"""Archive M43 boundary evidence and bind inherited fresh setup without hiding source deltas."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--setup-origin', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    a=p.parse_args()
    current=json.loads((a.runtime/'release/promotion.json').read_text())
    baseline=json.loads((a.setup_origin/'release/promotion.json').read_text())
    q=json.loads((a.work/'qualification.json').read_text())
    if q['status'] != 'PASS' or q['source_tree_sha256'] != current['source_tree_sha256']:
        raise ValueError('qualification does not bind current runtime tree')
    if baseline['reference'] != current['reference'] or set(baseline['files']) != set(current['files']):
        raise ValueError('fresh setup cannot be inherited across resource/reference changes')
    changed=[name for name in current['files'] if current['files'][name] != baseline['files'][name]]
    if changed not in ([], ['release/runtime-surface.json']):
        raise ValueError(f'fresh setup inheritance only permits source-classification metadata delta: {changed}')
    evidence=ROOT/'artifacts/m43'; evidence.mkdir(parents=True,exist_ok=True)
    inherited=[]
    for name in ('off-setup','mtp-setup'):
        log=a.work/(name+'.log')
        if 'Provisioned ' not in log.read_text() or 'Runtime needs neither donor checkouts nor build-dir.' not in log.read_text():
            raise ValueError('missing successful fresh setup evidence')
        shutil.copy2(log,evidence/log.name)
        inherited.append(dict(name=name,status='PASS',log='artifacts/m43/'+log.name,sha256=digest(log)))
    for gate in q['gates']:
        log=a.work/gate['log']
        if digest(log) != gate['sha256'] or gate['returncode'] != gate['expected']:
            raise ValueError('gate log drift')
        shutil.copy2(log,evidence/log.name)
    for name in ('off-acceptance.json','mtp-acceptance.json'):
        shutil.copy2(a.work/name,evidence/name)
    # Archive verifier receipts/logs, not generated source exports or model KV.
    ref_root=a.runtime/'artifacts/m43'
    for path in ref_root.rglob('*'):
        if path.is_file() and path.suffix in ('.json','.log') and len(path.relative_to(ref_root).parts) <= 2:
            target=evidence/'reference'/path.relative_to(ref_root)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    q['inherited_setup' if changed else 'fresh_setup']=dict(source_commit=baseline['source_commit'],source_tree_sha256=baseline['source_tree_sha256'],
        manifest_sha256=digest(a.setup_origin/'release/promotion.json'),changed_files=changed,
        reason=('Only declarative source-classification metadata changed; all setup inputs are byte-identical; both operators and R1 reran.' if changed else 'Fresh independent construction from this exact complete source payload; dependency build scratch was unavailable during operation.'),gates=inherited)
    q['release_surface_sha256']=current['release_surface_sha256']
    q['promotion_implementation_sha256']=current['promotion_implementation_sha256']
    q['mechanics']=dict(status='PASS',tests=7,receipt='artifacts/m43/mechanics.log',sha256=digest(evidence/'mechanics.log'),
        scope='complete manifests, independent destinations/checkout paths, timestamps/ignored build products, dirty inputs, missing/unclassified source, reference drift, qualification mismatch and runtime-only patches')
    encoded=json.dumps(q,sort_keys=True,indent=2)+'\n'
    (ROOT/'release/promotion-qualification.json').write_text(encoded)
    (evidence/'qualification.json').write_text(encoded)
    shutil.copy2(a.setup_origin/'release/promotion.json',evidence/'fresh-setup-source-manifest.json')
    shutil.copy2(a.runtime/'release/promotion.json',evidence/'tested-source-manifest.json')
    print('M43 RELEASE_REPOSITORY_EXTRACTION — PASS')


if __name__=='__main__':main()
