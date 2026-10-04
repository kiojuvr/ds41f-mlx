"""M42 fresh isolation proof. Always restores unavailable historical authorities."""
import json, os, subprocess, sys, time
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'artifacts/m42';O.mkdir(exist_ok=True)
paths=[Path('/tmp/ds41f-m33-omlx'),Path('/tmp/ds41f-m32-recipe'),Path('/Users/kioju/omlx-0.7.0.release'),Path('/Volumes/SDXC-512/deepseek-v41-flash-mlx/third_party/deepseek-recipe'),Path('/Volumes/SDXC-512/ds41f-m41-final-build')]
paths += [R/f'artifacts/m{n}' for n in list(range(31,42))+['36r']]
record=dict(schema='ds41f.m42.isolation.v1',authorities=[],status='RUNNING')
moved=[]
try:
    for path in paths:
        hidden=path.with_name(path.name+'.m42-unavailable')
        row=dict(path=str(path),existed=path.exists(),hidden=str(hidden))
        if path.exists():
            if hidden.exists():raise RuntimeError(f'collision {hidden}')
            path.rename(hidden);moved.append((path,hidden))
        row['unavailable']=not path.exists();record['authorities'].append(row)
    cmd=[sys.executable,'-m','ds41f_mlx.reference','--profile','both','--real-model','--output',str(O/'independent.json')]
    record['command']=cmd
    (O/'isolation.json').write_text(json.dumps(record,indent=2)+'\n')
    with (O/'independent.log').open('w') as log:
        proc=subprocess.run(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT,timeout=5400)
    record.update(status='PASS' if proc.returncode==0 else 'FAIL',returncode=proc.returncode)
finally:
    for path,hidden in reversed(moved):hidden.rename(path)
    record['restored']=all(p.exists() and not h.exists() for p,h in moved)
    (O/'isolation.json').write_text(json.dumps(record,indent=2)+'\n')
