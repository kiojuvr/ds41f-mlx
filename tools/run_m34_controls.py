"""Reuse M33 paired benchmark math unchanged, extend stream to 512 at 4K/12K.
Each context loads one model and runs warmup/OFF/unguarded/guarded sequentially.
Only workload dimensions and evidence destination differ from M33.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'tools/run_m33_guard_bench.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()==json.loads((ROOT/'artifacts/m33/runtime-identities.json').read_text())['ds41f_source_hashes']['tools/run_m33_guard_bench.py']
if len(sys.argv)==1:
    for context in (4096,12288):
        subprocess.run([sys.executable,__file__,str(context)],check=True)
else:
    context=int(sys.argv[1]);assert context in (4096,12288)
    code=source.read_text().replace("OUT=ROOT/'artifacts/m33/performance.json'",f"OUT=ROOT/'artifacts/m34/control-{context}.json'")
    code=code.replace('4096',str(context)).replace('128','512')
    code=code.replace('One warmup + paired 4K/512 diagnostic; not broad soak','M34 paired sustained 512-token control')
    exec(compile(code,str(source),'exec'),dict(__file__=str(source),__name__='__main__'))
