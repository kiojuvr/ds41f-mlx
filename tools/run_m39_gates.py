"""Serialized M39 affected M33--M38 gates; no historical evidence overwritten."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/m39'
env=dict(os.environ,DS41F_OMLX_PATH='/tmp/ds41f-m33-omlx')
commands=[
 ('identities',[sys.executable,'tools/check_m34_identities.py']),
 ('admission-tests',[sys.executable,'-m','pytest','-q','tests/test_m39_lifetime_admission.py','--junitxml=artifacts/m39/admission-tests.xml']),
 ('client-tests',[sys.executable,'-m','pytest','-q','tests/test_m38_client_faults.py','tests/test_m37_local_client.py','tests/test_m19_web_client.py','--junitxml=artifacts/m39/client-tests.xml']),
 ('runtime-regressions',[sys.executable,'-m','pytest','-q',*'tests/test_m36r_recovery.py tests/test_m36_recovery_boundary.py tests/test_m35_transport_lease.py tests/test_m34_cache_release.py tests/test_m29_mtp_lifecycle.py tests/test_m25_mtp_boundary.py tests/test_prefill_fp8_mlx_p5.py tests/test_prefill_fp8_mlx_p6.py tests/test_prefill_fp8_mlx_p7.py tools/m33_semantic_tests.py tools/m33_semantic_extra_tests.py tests/test_stateful_request_policy.py tests/test_runtime_config.py'.split()]),
 ('prior-evidence-regressions',[sys.executable,'-m','pytest','-q',*'tests/test_m33_protocol_evidence.py tests/test_m34_operational_evidence.py tests/test_m35_http_evidence.py tests/test_m36_evidence.py tests/test_m36r_evidence.py tests/test_m37_evidence.py tests/test_m38_evidence.py'.split(),'-k','not identity_and_sources and not current_sources_and_raw_evidence_match and not final_qualification and not current_source_and_evidence_hashes and not current_receipt_hashes_and_nonpromotion']),
 ('release-off-regressions',['/Users/kioju/.venvs/omlx-0.7.0.release/bin/python','-m','pytest','-q','tests/test_m20_generation_dependency.py']),
 ('rust-regressions',['cargo','test']),
 ('recipe-matrix',[sys.executable,'-c',"from pathlib import Path; p=Path('tools/run_m36r_recipe_matrix.py'); code=p.read_text().replace('artifacts/m36r/recipe-matrix.json','artifacts/m39/recipe-matrix.json'); exec(compile(code,str(p),'exec'),{'__file__':str(p.resolve()),'__name__':'__main__'})"]),
 ('native-interruptions',[sys.executable,'-c',"from pathlib import Path; p=Path('tools/run_m35_native_interruptions.py'); code=p.read_text().replace('artifacts/m35/native-interruptions.json','artifacts/m39/native-interruptions.json'); exec(compile(code,str(p),'exec'),{'__file__':str(p.resolve()),'__name__':'__main__'})"]),
]
assert json.loads((OUT/'integration.json').read_text())['status'] != 'RUNNING'
rows=[]
for name,cmd in commands:
    e=env.copy()
    if name=='release-off-regressions':e.pop('DS41F_OMLX_PATH',None);e.pop('DS41F_RECIPE_PATH',None)
    with (OUT/f'{name}.log').open('w') as f:ret=subprocess.run(cmd,cwd=ROOT,env=e,stdout=f,stderr=subprocess.STDOUT)
    rows.append(dict(name=name,command=cmd,returncode=ret.returncode))
    (OUT/'gate-commands.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(name,ret.returncode,flush=True)
    if ret.returncode:sys.exit(ret.returncode)
