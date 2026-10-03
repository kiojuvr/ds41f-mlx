"""Current-source M33--M36R native/HTTP gates, serialized (one model owner)."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/m37';OUT.mkdir(parents=True,exist_ok=True)
env=dict(os.environ, DS41F_OMLX_PATH='/tmp/ds41f-m33-omlx')
commands=[
 ('identities',[sys.executable,'tools/check_m34_identities.py']),
 ('recipe-matrix',[sys.executable,'-c',"from pathlib import Path; p=Path('tools/run_m36r_recipe_matrix.py'); code=p.read_text().replace('artifacts/m36r/recipe-matrix.json','artifacts/m37/recipe-matrix.json'); exec(compile(code,str(p),'exec'),{'__file__':str(p.resolve()),'__name__':'__main__'})"]),
 ('m35-http',[sys.executable,'-c',"import asyncio; from tools import run_m35_http_qualification as q; q.OUT=q.ROOT/'artifacts/m37/m35-http.json'; asyncio.run(q.main())"]),
 ('m36-probe',[sys.executable,'-c',"import asyncio; from tools import run_m36_recovery_probe as q; q.OUT=q.q.ROOT/'artifacts/m37/m36-probe.json'; asyncio.run(q.main())"]),
 ('m36r-http',[sys.executable,'-c',"import asyncio; from tools import run_m36r_http as q; q.OUT=q.q.ROOT/'artifacts/m37/m36r-http.json'; asyncio.run(q.main())"]),
 ('native-interruptions',[sys.executable,'-c',"from pathlib import Path; p=Path('tools/run_m35_native_interruptions.py'); code=p.read_text().replace('artifacts/m35/native-interruptions.json','artifacts/m37/native-interruptions.json'); exec(compile(code,str(p),'exec'),{'__file__':str(p.resolve()),'__name__':'__main__'})"]),
]
rows=[]
for name,cmd in commands:
    with (OUT/f'{name}.log').open('w') as f:
        ret=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    rows.append(dict(name=name,command=cmd,returncode=ret.returncode))
    (OUT/'native-gate-commands.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(name,ret.returncode,flush=True)
    if ret.returncode:sys.exit(ret.returncode)
