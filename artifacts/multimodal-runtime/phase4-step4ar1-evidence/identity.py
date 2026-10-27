"""Isolated normal test-install admission before serving; no model weights."""
import hashlib,json,subprocess,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
sys.path.append(str(ROOT))
from tools.qualify_installed_admission import prove_installed_admission
p=prove_installed_admission(checkpoint=False)
r=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),weights_loaded=0,kind='MODEL_FREE_NORMAL_INSTALLED_ADMISSION_PRE_WOA',profile=p['profile'],execution_identity=p['runtime']['identity_sha256'],actual_pre_woa_materialized=p['actual_pre_woa_materialized'],result_sha256=p['result_sha256'],identity_object_preserved=p['identity_object_preserved'],environment_valid=p['runtime']['environment_valid'],full_proof_sha256=hashlib.sha256(json.dumps(p,sort_keys=True,default=str).encode()).hexdigest(),note='Fresh normal test install because retained /tmp Phase1 dependency Python files were absent; immutable repository-pinned payload, no operational install modification. Full checkpoint byte admission still required in eventual Step4B worker.')
assert r['head']=='2cb84bf06b2eb72e41effe65e484e66b655b912e'
(OUT/'identity.json').write_text(json.dumps(r,indent=2)+'\n');print('PASS: normal test installation admission, actual zero-weight BF16 pre_woa, no weights')
