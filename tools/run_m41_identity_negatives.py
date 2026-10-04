"""Fail-closed source/package/native/resource/startup identity fault gate.

Only the selected private test venv and source clone are temporarily changed;
no shared Homebrew library, donor checkout or preserved OFF environment is edited.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from ds41f_mlx.mtp_identity import RECORD, ROOT, config
from ds41f_mlx.mtp_setup import sha

OUT=Path(sys.argv[1]);OUT.parent.mkdir(parents=True,exist_ok=True)
result=dict(schema='ds41f.m41.identity-negatives.v1',status='RUNNING',cases=[])


def command(name,argv=None,env=None,pass_expected=False):
    p=subprocess.run([sys.executable,'-m','ds41f_mlx.ops',*(argv or ['inspect','--profile','mtp-singleton-v1'])],
                     text=True,capture_output=True,env=env)
    result['cases'].append(dict(name=name,returncode=p.returncode,stdout=p.stdout[-1000:],stderr=p.stderr[-2000:]))
    assert (p.returncode==0)==pass_expected,(name,p.stdout,p.stderr)


def changed(path,action,name):
    content=path.read_bytes();mode=path.stat().st_mode
    temporary=path.with_name(path.name+'.fault')
    try:
        temporary.write_bytes(content);temporary.chmod(mode)
        action(temporary)
        os.replace(temporary,path)  # never truncate a loaded native inode
        command(name)
    finally:
        temporary.write_bytes(content);temporary.chmod(mode)
        os.replace(temporary,path)
    assert sha(path)==__import__('hashlib').sha256(content).hexdigest()

try:
    command('pristine',pass_expected=True)
    changed(ROOT/'ds41f_mlx/mtp_profile.py',lambda p:p.write_bytes(p.read_bytes()+b'\n# identity fault fixture\n'), 'runtime_source_drift')
    omlx=Path(importlib.util.find_spec('omlx').origin).parent
    changed(omlx/'_version.py',lambda p:p.write_bytes(p.read_bytes()+b'\n# identity fault fixture\n'), 'candidate_package_drift')
    tokenizer=Path(sys.prefix)/'share/ds41f-mtp/recipe/static/tokenizers/v41/tokenizer.json'
    changed(tokenizer,lambda p:p.write_bytes(b'{}'),'wrong_tokenizer')
    original=tokenizer.with_suffix('.saved')
    tokenizer.rename(original)
    try:command('missing_tokenizer')
    finally:original.rename(tokenizer)
    spec=importlib.util.find_spec('deepseek_recipe._native');native=Path(spec.origin)
    saved=native.with_suffix('.saved');native.rename(saved)
    try:command('missing_native_module')
    finally:saved.rename(native)
    # Wrong OFF wheel payload, same package version: not accepted as the candidate.
    off_python=Path.home()/'.venvs/omlx-0.7.0.release/bin/python'
    off_native=Path(subprocess.check_output([str(off_python),'-c','import deepseek_recipe._native as n;print(n.__file__)'],text=True).strip())
    changed(native,lambda p:p.write_bytes(off_native.read_bytes()),'wrong_native_same_version')
    # Actual missing dependent dylib, without altering a shared host library.
    dependency=next(line.strip().split(' (')[0] for line in subprocess.check_output(['otool','-L',str(native)],text=True).splitlines()[1:]
                    if line.strip().startswith('/opt/homebrew/'))
    def broken_link(path):
        subprocess.run(['install_name_tool','-change',dependency,'/private/var/empty/ds41f-missing.dylib',str(path)],check=True,capture_output=True)
        subprocess.run(['codesign','--force','--sign','-',str(path)],check=True,capture_output=True)
    changed(native,broken_link,'missing_native_dylib')
    for key,value in [('DS41F_HOST','0.0.0.0'),('DS41F_HOST','localhost'),('DS41F_MAX_LIVE_SESSIONS','4'),
                      ('DS41F_OMLX_PATH','/tmp/forbidden'),('DS41F_RECIPE_PATH','/tmp/forbidden'),
                      ('PYTHONPATH','.'),('DS41F_ENABLE_DIAGNOSTIC_ENDPOINTS','1'),
                      ('DS41F_TRACE_HISTORY_LIMIT','2048'),('OMLX_MTP_ROW_EXACT_VERIFY','0'),
                      ('DS41F_P8_TILE_NATIVE_CARRY','1'),('WEB_CONCURRENCY','2')]:
        env=dict(os.environ);env[key]=value;command(key+'_'+value,env=env)
    command('unknown_profile',['inspect','--profile','unknown'])
    command('validation_bypass',['start','--profile','mtp-singleton-v1','--no-validate'])
    command('worker_override',['start','--profile','mtp-singleton-v1','--workers','2'])
    command('nonloopback_cli',['start','--profile','mtp-singleton-v1','--host','0.0.0.0'])
    with tempfile.TemporaryDirectory() as tmp:
        env=dict(os.environ,DS41F_CHECKPOINT=tmp)
        command('missing_checkpoint',env=env)
        checkpoint=config().checkpoint_path
        for name in ('config.json','model.safetensors.index.json','tokenizer.json'):
            (Path(tmp)/name).write_bytes((checkpoint/name).read_bytes())
        (Path(tmp)/'config.json').write_text('{}')
        command('wrong_checkpoint_metadata',env=env)
    command('restored_pristine',pass_expected=True)
    result['status']='PASS'
except BaseException as exc:result.update(status='FAIL',error=repr(exc));raise
finally:OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],cases=len(result['cases']))))
