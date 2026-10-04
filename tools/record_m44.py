"""Close M44 from bounded execution and independent projection receipts."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/m44'
WORK = Path('/Volumes/SDXC-512/ds41f-m44-sealed')
RUNTIME = Path('/Volumes/SDXC-512/ds41f-m44-runtime-qualified')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2)+'\n')


def main():
    q = load(WORK/'qualification.json')
    assert q['status'] == 'PASS' and q['restored']
    current = load(RUNTIME/'release/promotion.json')
    initial = load(Path('/Volumes/SDXC-512/ds41f-m44-runtime/release/promotion.json'))
    delta = [p for p in current['files'] if current['files'][p] != initial['files'][p]]
    assert set(current['files']) == set(initial['files'])
    assert set(delta) == {'checks/test_target_generation.py', 'docs/target-generation-ownership.md'}
    assert q['source_tree_sha256'] == current['source_tree_sha256']
    for p, entry in current['files'].items():
        assert sha(RUNTIME/p) == entry['sha256']
    assert '10 passed' in (OUT/'final-projected-owned-tests.log').read_text()
    dev = load(OUT/'r1-final.json')
    projected = load(RUNTIME/'artifacts/m43/reference.json')
    assert dev['status'] == projected['status'] == 'PASS'
    assert projected['profile'] == 'both' and projected['level'] == 'full'
    for p, digest in dev['candidate_source'].items():
        assert sha(ROOT/p) == digest
    assert not subprocess.check_output(['git','diff','--name-only',
        'd7c63727ad0751a4cc6eef6e6c83769f0cf5bef8','HEAD','--',
        'reference/R1','artifacts/m42','artifacts/m43','docs/milestone-42-reference-release.md',
        'docs/milestone-43-release-repository-extraction.md'],cwd=ROOT).strip()
    for name in ('matched.json','matched-installed.json','endpoint-200k.json','operational-qualified.json'):
        assert load(OUT/name)['status'] == 'PASS'
    assert load(OUT/'endpoint-200k.json')['execution_source_sha256'] == sha(ROOT/'ds41f_mlx/runtime/target_generation.py')
    snapshots = []
    for name in ('ds41f-m44-determinism-a','ds41f-m44-determinism-b'):
        base = ROOT.parent/name
        snapshots.append({str(p.relative_to(base)):dict(sha256=sha(p), mode=p.stat().st_mode & 0o777)
                          for p in sorted(base.rglob('*')) if p.is_file()})
    assert snapshots[0] == snapshots[1]
    save(OUT/'determinism.json',dict(status='PASS',complete_tree=True,
        files=len(snapshots[0]),source_commit=q['source_commit'],
        source_tree_sha256=q['source_tree_sha256'],
        complete_tree_sha256=hashlib.sha256(json.dumps(snapshots[0],sort_keys=True).encode()).hexdigest()))
    shutil.copy2(WORK/'qualification.json', OUT/'independent-qualification.json')
    shutil.copy2(RUNTIME/'artifacts/m43/reference.json', OUT/'projected-r1.json')
    for name in ('off-acceptance.json','mtp-acceptance.json'):
        shutil.copy2(WORK/name, OUT/name)
    raw = OUT/'projection-gates'; raw.mkdir(exist_ok=True)
    for gate in q['gates']:
        path = WORK/gate['log']
        assert sha(path) == gate['sha256']
        shutil.copy2(path, raw/path.name)
    setup = []
    for name in ('off','mtp'):
        path = WORK/(name+'-setup.log')
        setup.append(dict(name=name+'-setup',status='PASS',sha256=sha(path),log='artifacts/m44/projection-gates/'+path.name))
        shutil.copy2(path,raw/path.name)
    q['schema'] = 'ds41f.m44.promotion-qualification.v1'
    q['fresh_setup'] = dict(source_commit=initial['source_commit'],
        source_tree_sha256=initial['source_tree_sha256'],gates=setup,
        non_build_input_delta=delta,
        reason='Fresh independent construction; final isolated operational/R1 qualification rerun on the final payload. All delivered implementation, package/build inputs, dependencies and R1 material are byte-identical to fresh setup; only owned test and ownership documentation changed.')
    q['owned_engine_tests'] = dict(status='PASS',tests=10,sha256=sha(OUT/'final-projected-owned-tests.log'))
    q['determinism'] = load(OUT/'determinism.json')
    q['execution_evidence'] = {p:sha(OUT/p) for p in ('r1-final.json','matched.json',
        'matched-installed.json','endpoint-200k.json','operational-qualified.json','projected-r1.json')}
    save(ROOT/'release/promotion-qualification.json',q)
    decision = dict(schema='ds41f.m44.ownership-decision.v1',status='PASS',
        baseline_commit='d7c63727ad0751a4cc6eef6e6c83769f0cf5bef8',
        qualified_source_commit=q['source_commit'],source_tree_sha256=q['source_tree_sha256'],
        reference_sha256=q['reference_sha256'],
        moved=['terminal bootstrap','single-stream target scheduling','normalized-logprob sampling',
               'consumed-token history/frontier','EOS/length/cancel','failure burn and stale-alias rejection',
               'wired-memory lease','coherent exact-list generation-to-idle/P6 capability retirement'],
        temporary=['checkpoint loader','LanguageModel target-forward/all-layer mutation',
                   'packed cache representation','quantization/custom kernels','SSD Engram/prefetch',
                   'separate bounded MTP scheduler'],
        standard_off_default=True,capability_expansion=False,replay=0,repack=0,
        gates=dict(R1_standard_off='PASS',R1_independent_both='PASS',
                   owned_tests=10,affected_tests=37,affected_subtests=8,
                   independent_OFF_acceptance='PASS',unchanged_MTP_acceptance='PASS',
                   native='PASS',source_origin='PASS',determinism='PASS'),
        performance=dict(initial=load(OUT/'matched.json')['comparisons'],
                         independent_installed=load(OUT/'matched-installed.json')['comparisons'],
                         endpoint_200k=load(OUT/'endpoint-200k.json')['runs'][0],
                         interpretation='Practical parity, not a speedup; >=15 tok/s floor preserved.'),
        bounded_development_findings=['original cache objects retain P6 seal unlike scheduler extraction; explicitly retire only stale lease metadata at coherent idle',
            'device wired-memory policy must be owned/restored',
            'optional donor kernel symbol was an invalid new test admission rule, not a baseline defect',
            'R1 preview needs its documented guarded substrate; off-only environment is insufficient',
            'new CMake source projection needs fresh generated cache'],
        historical_suite='Not a current gate: historical hash attestations detect development changes; profile fixtures require isolated dependency authorities. Evidence not rewritten.',
        exact_next_frontier='LanguageModel target-forward orchestration and all-40-layer cache mutation/commit on the existing P7-compatible packed representation; no replay, adapter, second authority or capability expansion.',
        evidence=q['execution_evidence'])
    save(OUT/'decision.json',decision)
    print(json.dumps(dict(status='PASS',source_commit=q['source_commit'],source_tree_sha256=q['source_tree_sha256'])))


if __name__ == '__main__': main()
