"""Reuse immutable R1 assertions with explicit M51R qualification source bytes.
Two synthetic ownership fixtures need new metadata fields; their active versions
retain the original assertions. No R1 material or installed admission is edited.
The model fixture retains its default owned-process/local-port path.
"""
import json
from pathlib import Path
import subprocess
import sys
from ds41f_mlx import mtp_identity as identity
identity.RECORD = Path('artifacts/m51r/qualification-identity.json')
bootstrap = "from pathlib import Path; import runpy,sys; import ds41f_mlx.mtp_identity as m; m.RECORD=Path('artifacts/m51r/qualification-identity.json'); module=sys.argv[1]; sys.argv=[module,*sys.argv[2:]]; runpy.run_module(module,run_name='__main__')"
# ops start only execs canonical serve. Preserve that exact launcher/application,
# while retaining the explicit source allowance in the new interpreter.
child_bootstrap = f'''import subprocess,sys
original_popen = subprocess.Popen
def launch(cmd,*args,**kwargs):
    if cmd[:4] == [sys.executable,'-m','ds41f_mlx.ops','start']:
        cmd = [sys.executable,'-c',{bootstrap!r},'ds41f_mlx.serve',*cmd[4:]]
    return original_popen(cmd,*args,**kwargs)
subprocess.Popen = launch
{bootstrap}
'''
original_run = subprocess.run
adaptations = []
def run(cmd, *args, **kwargs):
    if len(cmd) > 3 and cmd[:2] == [sys.executable, '-m']:
        if cmd[2] == 'pytest':
            for i, arg in enumerate(cmd):
                name = Path(arg).name
                if name in ('test_m34_cache_release.py', 'test_m35_transport_lease.py'):
                    adaptations.append({'original':arg, 'fixture_adapter':'tests/'+name})
                    cmd[i] = 'tests/'+name
        cmd[:] = [sys.executable, '-c', child_bootstrap, *cmd[2:]]
    return original_run(cmd, *args, **kwargs)
subprocess.run = run
from ds41f_mlx.reference import main
result = main(['--profile','mtp-singleton-v1','--real-model',
               '--output','artifacts/m51r/r1.json'])
path = Path('artifacts/m51r/r1.json')
receipt = json.loads(path.read_text())
receipt['qualification_adaptation'] = dict(source_allowance=str(identity.RECORD),
    fixtures=adaptations, server='canonical serve; qualification-only source record; owned process',
    not_normal_profile_promotion=True)
path.write_text(json.dumps(receipt, indent=2)+'\n')
assert result == 0, receipt.get('error')
