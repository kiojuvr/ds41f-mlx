"""Exact GLM distribution owner shared by packaging and provisioning.

No local observation enrollment and no runtime profile fallback.
"""
import json
from pathlib import Path
import tarfile

from .mtp_setup import SOURCES, sha

PREFIX = 'custom_kernels/glm_moe_dsa/'


def default_profile(serving_distribution):
    authority = json.loads((SOURCES/'execution-profiles.json').read_text())
    selected = authority['defaults'][serving_distribution]
    if selected not in ('native', 'portable'):
        raise ValueError('invalid production GLM default')
    return selected


def bundle_authority():
    receipt = json.loads((SOURCES/'glm-native.json').read_text())
    archive = SOURCES/receipt['archive']
    if receipt['schema'] != 'ds41f.glm-bundle.v1' or receipt['profile'] != 'native':
        raise ValueError('invalid native bundle authority')
    if sha(archive) != receipt['archive_sha256']:
        raise ValueError('native distribution archive drift')
    return receipt


def provision(site, profile):
    if profile not in ('native', 'portable'):
        raise ValueError('unknown GLM execution profile')
    directory = Path(site)/'omlx'/PREFIX
    if any(p.suffix in ('.so', '.dylib', '.metallib') for p in directory.iterdir()):
        raise ValueError('GLM provisioning requires extension-absent base')
    if profile == 'portable':
        return
    receipt = bundle_authority()
    with tarfile.open(SOURCES/receipt['archive']) as tar:
        members = tar.getmembers()
        if {m.name for m in members} != set(receipt['files']) or not all(m.isfile() for m in members):
            raise ValueError('native archive membership drift')
        tar.extractall(directory, filter='data')
    for name, identity in receipt['files'].items():
        path = directory/name
        if path.stat().st_size != identity['size'] or sha(path) != identity['sha256']:
            raise ValueError('native provisioned identity drift')
