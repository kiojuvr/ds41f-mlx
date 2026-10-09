"""Reuse M52R's existing R1 synthetic-fixture adapters, with strict installed admission.
No source allowance, interpreter bootstrap, server substitution or identity override.
"""
import json
from pathlib import Path
import subprocess
import sys

original_run = subprocess.run
adaptations = []
names = {'test_m34_cache_release.py', 'test_m35_transport_lease.py',
         'test_m36_recovery_boundary.py', 'test_m39_lifetime_admission.py',
         'test_m41_profile.py'}


def run(cmd, *args, **kwargs):
    cmd = list(cmd)
    if cmd[:3] == [sys.executable, '-m', 'pytest']:
        for i, arg in enumerate(cmd):
            if Path(arg).name in names:
                replacement = 'tests/' + Path(arg).name
                adaptations.append(dict(original=arg, fixture_adapter=replacement))
                cmd[i] = replacement
    return original_run(cmd, *args, **kwargs)


subprocess.run = run
from ds41f_mlx.reference import main
result = main(['--profile', 'mtp-singleton-v1', '--real-model',
               '--output', 'artifacts/m53r/r1.json'])
path = Path('artifacts/m53r/r1.json')
receipt = json.loads(path.read_text())
receipt['qualification_adaptation'] = dict(fixtures=adaptations,
    source_allowance=None, server='unchanged normal operator start; strict installed admission')
path.write_text(json.dumps(receipt, indent=2) + '\n')
raise SystemExit(result)
