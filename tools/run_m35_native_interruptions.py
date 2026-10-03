"""Rerun the unchanged M33/M34 checkpoint interruption matrix after M35.
Only evidence destination changes; historical artifacts are never rewritten.
"""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
source = ROOT/'tools/run_m33_interruptions.py'
identity = json.loads((ROOT/'artifacts/m33/runtime-identities.json').read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest() == identity['ds41f_source_hashes']['tools/run_m33_interruptions.py']
old = "OUT=ROOT/'artifacts/m33/interruptions.json'"
code = source.read_text()
assert code.count(old) == 1
code = code.replace(old, "OUT=ROOT/'artifacts/m35/native-interruptions.json'")
namespace = dict(__file__=str(source), __name__='__main__')
exec(compile(code, str(source), 'exec'), namespace)
result = namespace['result']
result['m35_driver_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
result['m34_corrected_runtime_sha256'] = hashlib.sha256((ROOT/'ds41f_mlx/runtime/mtp_lifecycle.py').read_bytes()).hexdigest()
namespace['OUT'].write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
