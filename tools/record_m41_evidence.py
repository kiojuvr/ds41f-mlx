"""Record bounded M41 evidence, explicitly separating fresh gates and inheritance."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/m41'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(name):return json.loads((OUT/name).read_text())

def method(text,cls,name):
    tree=ast.parse(text)
    c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    n=next(n for n in c.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
    return ast.get_source_segment(text,n)


def main():
    composed=load('final-composed.json');sockets=load('final-sockets.json')
    negatives=load('final-identity-negatives.json');perf=load('final-performance.json')
    inspected=load('final-inspect.json');off=load('final-off-real-acceptance.json')
    protocols=load('final-off-protocols.json')
    assert all(r['status']=='PASS' for r in (composed,sockets,negatives,perf,inspected,off,protocols))
    assert composed['identity_sha256']==inspected['identity_sha256']
    for row in composed['cases']:
        metrics=row['outcome']['metrics'];c=metrics['settlement']
        assert metrics['aligned_idle'] and metrics['prompt_replay']==metrics['full_cache_repack']==0
        assert c['new_verify_cycles']==c['new_proposals']==c['history_replay']==c['full_cache_repack']==0
        assert c['target_forwards']<=1
    # Whole unchanged hard-owner modules; changed wrappers are NOT blanket inherited.
    old=json.loads((ROOT/'artifacts/m39/qualification.json').read_text())
    inherited={p:sha(ROOT/p) for p in ('ds41f_mlx/runtime/mtp_lifecycle.py',
        'ds41f_mlx/runtime/recipe_semantic_guard.py','ds41f_mlx/serving/request_fence.py')}
    assert all(v==old['preserved_ownership_layers'][p] for p,v in inherited.items())
    spans={}
    for p,cls,names in [
        ('ds41f_mlx/serving/internal_mtp.py','InternalMTPQualificationBackend',
         ('create_stateful_session','get_stateful_session','close_stateful_session')),
        ('ds41f_mlx/internal_local_client.py','InternalLocalClient',('execute_tools','retire','_require_resolved_effects','_require_known_lifecycle'))]:
        before=subprocess.check_output(['git','show','8813829:'+p],cwd=ROOT,text=True)
        now=(ROOT/p).read_text()
        for name in names:
            a=method(before,cls,name);b=method(now,cls,name);assert a==b,(p,name)
            spans[p+':'+name]=hashlib.sha256(b.encode()).hexdigest()
    for path in ('release','rust','ds41f_mlx/runtime','ds41f_mlx/prefill_fp8_mlx','ds41f_mlx/config.py'):
        assert not subprocess.check_output(['git','diff','8813829','--',path],cwd=ROOT)
    artifacts=sorted(p for p in OUT.glob('final-*') if p.is_file())
    artifacts += [OUT/'clean-clone.json',OUT/'hidden-authorities.json',OUT/'clean-setup.log',
                  OUT/'socket-boundaries-python-mismatch.json',OUT/'socket-boundaries-python-mismatch.log',
                  OUT/'evidence-owner-tests.log',OUT/'legacy-receipt-test-debt.json']
    sources={str(p.relative_to(ROOT)):sha(p) for directory in ('ds41f_mlx','third_party/mtp')
             for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'tools').glob('*m41*.py')})
    for path in ('pyproject.toml','tools/m33_semantic_tests.py','tools/run_m36r_recipe_matrix.py',
                 'tools/run_m32_native_preview.py','tools/m32_native_parity_driver.py',
                 'tests/test_m41_profile.py','tests/test_m41_evidence.py',
                 'artifacts/m31/parity-fixtures.json','artifacts/m32/canonical-base.json','artifacts/m39/recipe-matrix.json'):
        sources[path]=sha(ROOT/path)
    record=dict(schema='ds41f.m41.qualification.v1',base_commit='8813829',
        qualified_implementation_commit=load('clean-clone.json')['source_commit'],
        decision='QUALIFIED_EXPLICIT_BOUNDED_LOCAL_MTP_RELEASE_CANDIDATE',
        default_profile='standard-off',candidate_profile='mtp-singleton-v1',
        dependency_identity=inspected['identity_sha256'],native_sha256=inspected['identity']['native_sha256'],
        native_wheel_sha256=inspected['build']['wheel_sha256'],actual_origins=inspected['actual_origins'],
        primary_commands=[
            dict(cwd='/Volumes/SDXC-512/ds41f-m41-final-clone',
                 environment={'DS41F_CHECKPOINT':'/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash'},
                 commands=[
                   '/Volumes/SDXC-512/ds41f-m41-final-venv/bin/ds41f inspect --profile mtp-singleton-v1',
                   '/Volumes/SDXC-512/ds41f-m41-final-venv/bin/ds41f accept --profile mtp-singleton-v1 --output .../final-composed.json',
                   'python -m tools.run_m41_socket_boundaries .../final-sockets.json',
                   'python -m tools.run_m41_identity_negatives .../final-identity-negatives.json',
                   'python -m tools.run_m41_matched_performance --output .../final-performance.json']),
            dict(cwd='/Volumes/SDXC-512/ds41f-m41-final-clone',
                 python='/Volumes/SDXC-512/ds41f-m41-final-venv/bin/python',
                 command='-m pytest -q tests/test_m41_profile.py tests/test_m35_transport_lease.py tests/test_m36r_recovery.py tests/test_m37_local_client.py tests/test_m38_client_faults.py tests/test_m39_lifetime_admission.py tests/test_m19_web_client.py tests/test_runtime_config.py'),
            dict(cwd='/Volumes/SDXC-512/ds41f-m41-final-clone',
                 python='/Volumes/SDXC-512/ds41f-m41-final-venv/bin/python',environment={'DS41F_TEST_INSTALLED':'1'},
                 command='-m pytest -q tools/m33_semantic_tests.py tests/test_m29_mtp_lifecycle.py tests/test_m34_cache_release.py'),
            dict(cwd=str(ROOT),python='/Users/kioju/.venvs/omlx-0.7.0.release/bin/python',commands=[
                '-m ds41f_mlx.ops accept --skip-cheap-gates --output artifacts/m41/final-off-real-acceptance.json',
                '-m tools.run_m41_off_protocols artifacts/m41/final-off-protocols.json',
                '-m pytest -q tests/test_runtime_config.py tests/test_m22_release_metadata.py tests/test_m25_mtp_boundary.py tests/test_m19_web_client.py']),
            dict(cwd=str(ROOT),commands=['cargo test --workspace',
                'cmake -S native -B /Volumes/SDXC-512/ds41f-m41-core-build',
                'cmake --build /Volumes/SDXC-512/ds41f-m41-core-build -j8',
                'ctest --test-dir /Volumes/SDXC-512/ds41f-m41-core-build --output-on-failure',
                'python3 tools/check_repository_self_containment.py'])],
        fresh=dict(composed_cases=len(composed['cases']),admission_rejections=len(composed['admission']),
            client_effects=composed['effects'],native_gates=composed['native_gates'],
            socket_cases=len(sockets['cases']),identity_cases=len(negatives['cases']),
            checkpoint_asset_verification=inspected['checkpoint']['verification'],
            source_clone=load('clean-clone.json'),off_real=off['status'],off_protocol_tool_persist_restore=protocols['status']),
        inheritance=dict(unchanged_modules=inherited,unchanged_methods=spans,
            scope='M33 semantic horizon + M34 protected ownership + M35/36 response leases + M36R restrictive recovery + M38 effect/lifecycle faults + M39 20k-lifetime bounded authority. Fresh installed/native/composed regressions rerun affected edges, not a new 20k model-lifetime soak.'),
        performance=perf,
        public_release_blockers=[],historical_receipt_test_debt=load('legacy-receipt-test-debt.json'),
        deliberate_limits=['trusted single-operator IPv4 loopback','one native lane/client/live session',
            '8192 prompt+output / 768 output','fixed deterministic text/weather envelope',
            'no arbitrary tools/schema/sampling/stop','no browser/remote/Rust MTP application',
            'no persistence/restart recovery/universal DSML repair/crash-safe effects'],
        longer_term_final_runtime_work=['ds41f-owned model/decode substrate replacing temporary oMLX',
            'self-contained ordinary setup for preserved OFF delivery as well as MTP',
            'optional artifact-portability/dependency reduction only with affected qualification'],
        excluded_attempts=[
            'clean first setup auto-selected Python 3.13.14; mandatory startup identity rejected, corrected exact interpreter',
            'final-clone setup attempt1 rejected resolved OpenCV basename incorrectly; fixed resolved 4.14.0 identity, rebuilt fresh absent venv',
            'performance first invocation missing --output; argparse rejected before model work; corrected command'],
        evidence={str(p.relative_to(ROOT)):sha(p) for p in artifacts},sources=sources)
    (OUT/'qualification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(decision=record['decision'],evidence_files=len(record['evidence']))))

if __name__=='__main__':main()
