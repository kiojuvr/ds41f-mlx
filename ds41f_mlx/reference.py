"""Repository-owned Reference Release R1 verifier (no historical authorities)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OWN = ROOT / 'reference/R1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--profile', choices=['standard-off', 'mtp-singleton-v1', 'both'], default='both')
    p.add_argument('--real-model', action='store_true', help='required for full qualification; otherwise seam-only')
    args = p.parse_args(argv)
    out = args.output.resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    work = out.parent / (out.stem + '-gates'); work.mkdir(exist_ok=True)
    contract = json.loads((OWN/'contract.json').read_text())
    result = dict(schema='ds41f.reference.qualification.v1', reference='R1',
        reference_sha256=digest(OWN/'manifest.json'), contract_sha256=digest(OWN/'contract.json'),
        profile=args.profile, level='full' if args.real_model else 'seams', status='RUNNING',
        environment=dict(python=sys.version,platform=platform.platform(),prefix=sys.prefix),
        candidate_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        candidate_source={str(f.relative_to(ROOT)):digest(f) for f in sorted((ROOT/'ds41f_mlx').rglob('*.py'))}, gates=[])
    def save(): out.write_text(json.dumps(result,indent=2)+'\n')
    def run(name, cmd, env=None, timeout=3600):
        start=time.monotonic()
        proc=subprocess.run(cmd,cwd=ROOT,env=env,text=True,capture_output=True,timeout=timeout)
        log=work/(name+'.log');log.write_text(proc.stdout+proc.stderr)
        result['gates'].append(dict(name=name,command=cmd,returncode=proc.returncode,
            wall_s=time.monotonic()-start,log=str(log),sha256=digest(log)))
        save()
        if proc.returncode: raise RuntimeError(f'{name} failed; see {log}')
        return proc.stdout
    save()
    try:
        manifest=json.loads((OWN/'manifest.json').read_text())
        if digest(Path(__file__)) != manifest['verifier_sha256']:
            raise ValueError('R1 verifier drift')
        for name,sha in manifest['files'].items():
            if digest(OWN/name)!=sha: raise ValueError(f'R1 material drift: {name}')
        # Isolate modules: old lifecycle fixtures intentionally manipulate import paths.
        for f in sorted((OWN/'checks').glob('test_*.py')):
            run(f.stem,[sys.executable,'-m','pytest','-q','-p','no:cacheprovider',str(f)])
        recipe=Path(sys.prefix)/'share/ds41f-mtp/recipe'
        tok=recipe/'static/tokenizers/v41/tokenizer.json'
        if digest(tok)!=contract['identities']['tokenizer_sha256']:raise ValueError('wrong tokenizer')
        actual=json.loads(run('protocol',[sys.executable,'-m','reference.R1.protocol',str(OWN/'fixtures/protocol-inputs.json'),str(tok)]))
        if actual['records']!=json.loads((OWN/'fixtures/protocol-expected.json').read_text()):
            raise ValueError('canonical protocol mismatch')
        run('preview',[sys.executable,'-m','reference.R1.preview',str(tok),str(work/'preview.json')])
        run('recovery',[sys.executable,'-m','reference.R1.recovery',str(recipe),str(work/'recovery.json')])
        rows=json.loads((work/'recovery.json').read_text())['rows']
        keys=('representable','exact_prefix','semantic_complete','executable_tools','first_mismatch')
        projected=[dict(name=r['name'],canonical=r['canonical'],certificate={k:r['certificate'].get(k) for k in keys}) for r in rows]
        if projected!=json.loads((OWN/'fixtures/recovery-expected.json').read_text()):raise ValueError('recovery contract mismatch')
        if args.real_model:
            if args.profile in ('standard-off','both'):
                off=work/'off-source'; off.mkdir(exist_ok=True)
                with tarfile.open(OWN/'off-source.tar.gz') as t:t.extractall(off,filter='data')
                env=dict(os.environ,DS41F_OMLX_PATH=str(off),DS41F_RECIPE_PATH=str(recipe),DS41F_KV_ROOT=str(work/'kv'))
                run('standard-off',[sys.executable,'-m','reference.R1.off_model',str(work/'off-model.json')],env)
                if json.loads((work/'off-model.json').read_text())['status']!='PASS':raise ValueError('OFF model gate')
            if args.profile in ('mtp-singleton-v1','both'):
                run('mtp-singleton-v1',[sys.executable,'-m','reference.R1.mtp_model','--output',str(work/'mtp-model.json')])
                if json.loads((work/'mtp-model.json').read_text())['status']!='PASS':raise ValueError('MTP model gate')
        result['status']='PASS'; result['decision']='CONFORMANT' if args.real_model else 'SEAM_CONFORMANT_NOT_FULL_QUALIFICATION'
        result['artifacts']={str(f.relative_to(work)):digest(f) for f in sorted(work.glob('*.json'))}
    except Exception as exc:
        result.update(status='FAIL',decision='NONCONFORMANT',error=str(exc));save();return 1
    save(); print(json.dumps({k:result[k] for k in ('reference','reference_sha256','status','decision')}));return 0

if __name__=='__main__':raise SystemExit(main())
