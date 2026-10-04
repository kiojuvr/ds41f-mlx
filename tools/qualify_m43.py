"""Fresh real release-boundary qualification; always restores hidden authorities."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--skip-setup', action='store_true', help='retry operation after an already successful fresh setup')
    a=p.parse_args(); runtime=a.runtime.resolve(); work=a.work.resolve(); work.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((runtime/'release/promotion.json').read_text())
    result=dict(schema='ds41f.m43.qualification.v1',status='RUNNING',source_commit=manifest['source_commit'],
                source_tree_sha256=manifest['source_tree_sha256'],reference_sha256=manifest['reference']['sha256'],
                gates=[],isolation=[])
    def save(): (work/'qualification.json').write_text(json.dumps(result,indent=2)+'\n')
    env={k:v for k,v in os.environ.items() if not k.startswith(('DS41F_','OMLX_','DYLD_','UVICORN_')) and k not in ('PYTHONPATH','WEB_CONCURRENCY')}
    env.update(DS41F_CHECKPOINT=str(a.checkpoint.resolve()),DS41F_KV_ROOT=str(work/'kv'))
    def run(name, command, expected=0):
        start=time.monotonic(); log=work/(name+'.log')
        with log.open('w') as f:
            proc=subprocess.run([str(x) for x in command],cwd=runtime,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=7200)
        result['gates'].append(dict(name=name,command=[str(x) for x in command],returncode=proc.returncode,
            expected=expected,wall_s=time.monotonic()-start,log=log.name,sha256=hashlib.sha256(log.read_bytes()).hexdigest()))
        save()
        if proc.returncode != expected: raise RuntimeError(f'{name} failed; see {log}')
    moved=[]
    try:
        if not (runtime/'.git').exists():
            run('git-init',['git','init'])
            run('git-add',['git','add','.'])
            run('git-import',['git','-c','user.name=ds41f promotion','-c','user.email=promotion@local','commit','-m','Canonical source projection'])
        for profile, name in (('standard-off','off'),('mtp-singleton-v1','mtp')):
            if not a.skip_setup:
                run(name+'-setup',[sys.executable,'-m','ds41f_mlx.mtp_setup','--profile',profile,
                    '--venv',work/(name+'-env'),'--build-dir',work/(name+'-build')])
        paths=[ROOT,Path('/tmp/ds41f-m33-omlx'),Path('/tmp/ds41f-m32-recipe'),
            Path.home()/'omlx-0.7.0.release',Path('/Volumes/SDXC-512/deepseek-v41-flash-mlx/third_party/deepseek-recipe'),
            Path('/Volumes/SDXC-512/ds41f-m41-final-build'), work/'off-build',work/'mtp-build']
        for path in paths:
            hidden=path.with_name(path.name+'.m43-unavailable')
            row=dict(path=str(path),existed=path.exists())
            if path.exists():
                if hidden.exists():raise RuntimeError(f'hide collision: {hidden}')
                path.rename(hidden);moved.append((path,hidden))
            row['unavailable']=not path.exists();result['isolation'].append(row)
        save()
        off=work/'off-env/bin/python'; mtp=work/'mtp-env/bin/python'
        run('off-inspect',[off,'-m','ds41f_mlx.ops','inspect'])
        run('off-accept',[off,'-m','ds41f_mlx.ops','accept','--output',work/'off-acceptance.json'])
        run('mtp-inspect',[mtp,'-m','ds41f_mlx.ops','inspect','--profile','mtp-singleton-v1'])
        run('mtp-accept',[mtp,'-m','ds41f_mlx.ops','accept','--profile','mtp-singleton-v1','--output',work/'mtp-acceptance.json'])
        run('reference',[mtp,'-m','ds41f_mlx.reference','--profile','both','--real-model','--output',runtime/'artifacts/m43/reference.json'])
        # Delivery overrides fail closed before real execution, without changing R1.
        env['DS41F_OMLX_PATH']='/nonexistent-donor-checkout'
        run('off-bad-origin',[off,'-m','ds41f_mlx.ops','inspect'],2)
        run('mtp-bad-origin',[mtp,'-m','ds41f_mlx.ops','inspect','--profile','mtp-singleton-v1'],2)
        env.pop('DS41F_OMLX_PATH')
        run('off-no-validate',[off,'-m','ds41f_mlx.serve','--no-validate','--print-config'],2)
        run('source-origin',[mtp,'-m','ds41f_mlx.projection'])
        result['results']={name:json.loads((work/name).read_text())['status'] for name in ('off-acceptance.json','mtp-acceptance.json')}
        result['results']['reference']=json.loads((runtime/'artifacts/m43/reference.json').read_text())['status']
        if set(result['results'].values()) != {'PASS'}:raise RuntimeError('failed qualification receipt')
        result['origins']={name:json.loads((work/(name+'-inspect.log')).read_text()) for name in ('off','mtp')}
        result['status']='PASS'
    except BaseException as exc:
        result.update(status='FAIL',error=repr(exc));raise
    finally:
        for path,hidden in reversed(moved): hidden.rename(path)
        result['restored']=all(p.exists() and not h.exists() for p,h in moved)
        save()
    print(work/'qualification.json')


if __name__=='__main__':main()
