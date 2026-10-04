"""One-time promotion of semantic material; not called by R1 verification."""
import json, shutil, hashlib, tarfile, subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1]; O=R/'reference/R1'; O.mkdir(parents=True,exist_ok=True)
(O/'checks').mkdir(exist_ok=True); (O/'fixtures').mkdir(exist_ok=True)
names=['m8_continuation_contract','m11_tool_boundary_contract','m14_termination_contract','m29_mtp_lifecycle','m34_cache_release','m35_transport_lease','m36_recovery_boundary','m36r_recovery','m37_local_client','m38_client_faults','m39_lifetime_admission','m41_profile','stateful_request_policy']
for n in names:
    s=(R/f'tests/test_{n}.py').read_text().replace('from tests.test_', 'from test_')
    s=s.replace('pytest.importorskip("deepseek_recipe")','__import__("deepseek_recipe")')
    s=s.replace('"/Volumes/SDXC-512/deepseek-v41-flash-mlx/third_party/deepseek-recipe/static/tokenizers/v41/tokenizer.json"','str(__import__("pathlib").Path(__import__("sys").prefix)/"share/ds41f-mtp/recipe/static/tokenizers/v41/tokenizer.json")')
    (O/f'checks/test_{n}.py').write_text(s)
for src,dst in [('m31/parity-fixtures.json','protocol-inputs.json'),('m31/source-preview-mapping.json','preview.json')]:
    shutil.copyfile(R/'artifacts'/src,O/'fixtures'/dst)
base=json.loads((R/'artifacts/m32/canonical-base.json').read_text())
(O/'fixtures/protocol-expected.json').write_text(json.dumps(base['records'],indent=2,ensure_ascii=False)+'\n')
matrix=json.loads((R/'artifacts/m39/recipe-matrix.json').read_text())['rows']
keys=('representable','exact_prefix','semantic_complete','executable_tools','first_mismatch')
(O/'fixtures/recovery-expected.json').write_text(json.dumps([dict(name=r['name'],canonical=r['canonical'],certificate={k:r['certificate'].get(k) for k in keys}) for r in matrix],indent=2)+'\n')
for src,dst in [('m32_native_parity_driver.py','protocol.py'),('run_m32_native_preview.py','preview.py'),('run_m36r_recipe_matrix.py','recovery.py'),('run_m41_off_protocols.py','off_model.py')]:
    s=(R/'tools'/src).read_text().replace("ROOT = Path(__file__).resolve().parents[1]","ROOT = Path(__file__).resolve().parents[2]\nOWN = Path(__file__).resolve().parent")
    s=s.replace("ROOT / 'artifacts/m31/source-preview-mapping.json'","OWN / 'fixtures/preview.json'").replace("ROOT / 'artifacts/m31/parity-fixtures.json'","OWN / 'fixtures/protocol-inputs.json'")
    s=s.replace('from tools.run_m11_tool_boundary_qualification import initial_body', 'from ds41f_mlx.mtp_profile import WEATHER\ndef initial_body(city):\n    return dict(model="deepseek-v4.1-flash",messages=[dict(role="user",content=f"Use the tool to look up the weather in {city}, then answer concisely.")],tools=[WEATHER],tool_choice={"type":"function","function":{"name":"lookup_weather"}},reasoning_effort="none",temperature=0,max_tokens=96)')
    s=s.replace("Path('/tmp/ds41f-m32-recipe')", "Path(sys.prefix)/'share/ds41f-mtp/recipe'")
    (O/dst).write_text(s)
s=(R/'ds41f_mlx/mtp_acceptance.py').read_text()
a=s.index('    from .mtp_setup import ROOT'); b=s.index('    process = None',a)
s=s[:a]+"    result = dict(schema='ds41f.r1.model.v1',status='RUNNING',identity_sha256=provenance['identity_sha256'],cases=[],admission=[])\n"+s[b:]
s=s.replace('from .','from ds41f_mlx.').replace("Path('artifacts/m41/composed.json')","Path('artifacts/m42/mtp-model.json')")
(O/'mtp_model.py').write_text(s)
# Deliver the already-qualified OFF execution source, not a new implementation.
p=Path('/Users/kioju/omlx-0.7.0.release')
with tarfile.open(O/'off-source.tar.gz','w:gz') as t:
    for root in ['omlx','LICENSE']:
        for f in ([p/root] if (p/root).is_file() else sorted((p/root).rglob('*'))):
            if f.is_file() and '__pycache__' not in f.parts and f.suffix not in ('.pyc','.DS_Store'):
                t.add(f,arcname=str(f.relative_to(p)))
inspect=json.loads((R/'artifacts/m41/final-inspect.json').read_text())
(O/'environment.json').write_text(json.dumps({k:inspect[k] for k in ['checkpoint']},indent=2)+'\n')
print('promoted',O)
