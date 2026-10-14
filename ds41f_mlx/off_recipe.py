"""Publication of the exact retained M45/M47 qualified OFF recipe wheel.

Artifact receipt is a delivery binding, not permission to change resource pins.
MTP's separately qualified patched recipe authority is unchanged.
"""
import hashlib
import json
import zipfile
from pathlib import Path

from .mtp_setup import ROOT, SOURCES, sha


def artifact_authority(wheel=None):
    receipt = json.loads((SOURCES/'standard-off-recipe.json').read_text())
    if receipt['schema'] != 'ds41f.off-recipe-artifact.v1':
        raise ValueError('invalid OFF recipe artifact authority')
    source = json.loads((SOURCES/'sources.json').read_text())['sources']['recipe']
    if receipt['source'] != source or sha(SOURCES/source['archive']) != source['sha256']:
        raise ValueError('OFF recipe source authority drift')
    path = Path(wheel) if wheel is not None else SOURCES/receipt['wheel']['file']
    if sha(path) != receipt['wheel']['sha256']:
        raise ValueError('OFF recipe wheel identity drift')
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('OFF recipe duplicate wheel membership')
        payload = {n.removeprefix('deepseek_recipe/'): hashlib.sha256(z.read(n)).hexdigest()
                   for n in names if n.startswith('deepseek_recipe/') and not n.endswith('/')}
    if payload != receipt['payload']:
        raise ValueError('OFF recipe wheel payload drift')
    pin = json.loads((ROOT/'ds41f_mlx/runtime/admitted_resources.json').read_text())
    if payload['_native.abi3.so'] != pin['modules']['deepseek_recipe._native']['sha256']:
        raise ValueError('OFF recipe artifact differs from qualified resource pin')
    return path, receipt
