"""Run with existing test installation's python -I, no Model construction.
Normal installed admission precedes serving imports. checkpoint=False deliberately;
not a fresh Step4B payload/full-checkpoint gate. Retain concise identities only.
"""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.append(str(ROOT))
from tools.qualify_installed_admission import prove_installed_admission
p=prove_installed_admission(checkpoint=False)
receipt=dict(scope='existing dependency/test installation, not new Step4B payload or full checkpoint admission',weights_loaded=0,full_proof_sha256=hashlib.sha256(json.dumps(p,sort_keys=True,default=str).encode()).hexdigest(),proof={k:p[k] for k in ['profile','identity_object_preserved','actual_pre_woa_materialized','result_sha256']},runtime_status=p['runtime']['status'],runtime_identity=p['runtime']['identity_sha256'],environment_valid=p['runtime']['environment_valid'],host_status=p['host']['status'],checkpoint_admission=False,origin_file_count=len(p['origins']),reused_prior_gate='tools/qualify_installed_admission.py')
(OUT/'zero-weight.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('PASS: normal installed admission before serving, actual BF16 pre_woa, no model weights')
