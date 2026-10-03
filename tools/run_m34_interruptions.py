"""Fresh rerun of the exact M33 checkpoint interruption/fault matrix.
Evidence is not inherited. Only the output destination changes.
"""
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'tools/run_m33_interruptions.py'
identity=json.loads((ROOT/'artifacts/m33/runtime-identities.json').read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest()==identity['ds41f_source_hashes']['tools/run_m33_interruptions.py']
code=source.read_text().replace("OUT=ROOT/'artifacts/m33/interruptions.json'","OUT=ROOT/'artifacts/m34/interruptions.json'")
runtime_sha=hashlib.sha256((ROOT/'ds41f_mlx/runtime/mtp_lifecycle.py').read_bytes()).hexdigest()
namespace=dict(__file__=str(source),__name__='__main__')
exec(compile(code,str(source),'exec'),namespace)
result=namespace['result']
result.update(m34_runtime_sha256=runtime_sha,m34_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
namespace['OUT'].write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
