"""Installed standard-OFF admission and the isolated historical R1 evidence lane.

No production source/export or local seal equality. R1's private verifier remains
an evidence consumer, not an alternative ordinary runtime admission path.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import sysconfig
import tarfile

from .mtp_identity import CHECKPOINT, checkpoint_inventory, verify_checkpoint_bytes
from .execution_identity import MANIFEST, inventory, verify_tree
from .mtp_setup import ROOT, sha

def paths():
    return Path(sysconfig.get_paths()['purelib']), Path(sys.prefix)/'share/ds41f-mtp/recipe'


def inspect_reference_execution(cfg):
    """R1 private export lane: archive attribution is qualification authority only."""
    from .projection import verify
    verify()
    exported = cfg.omlx_path.resolve()
    if not exported.is_relative_to((ROOT/'artifacts').resolve()):
        raise ValueError('R1 requires its private OFF export')
    if cfg.recipe_path.resolve() != paths()[1].resolve():
        raise ValueError('private Reference recipe must be provisioned')
    from hashlib import sha256
    expected = {}
    with tarfile.open(ROOT/'reference/R1/off-source.tar.gz') as tar:
        for member in tar.getmembers():
            if member.isfile():
                expected[member.name.removeprefix('./')] = sha256(tar.extractfile(member).read()).hexdigest()
    if inventory(exported) != expected:
        raise ValueError('private Reference OFF export drift')
    # Authenticate installed shared payload independently from the R1 export.
    manifest = json.loads(MANIFEST.read_text())
    site = paths()[0]
    for name, payload in manifest['payloads'].items():
        verify_tree(site/name, payload, name)
    for name, digest in manifest['files'].items():
        if sha(site/name) != digest:
            raise ValueError('private Reference support payload drift')
    if sha(cfg.recipe_path/'static/tokenizers/v41/tokenizer.json') != manifest['tokenizer_sha256']:
        raise ValueError('private Reference tokenizer drift')
    from deepseek_recipe import _native
    from .execution_identity import host_closure
    native = Path(_native.__file__).resolve()
    if native != site/'deepseek_recipe'/native.name or sha(native) != manifest['payloads']['deepseek_recipe'].get(native.name):
        raise ValueError('private Reference recipe extension origin/identity drift')
    from .runtime.resource_admission import check_loaded_image
    check_loaded_image(native)
    graph = {Path(p).name: Path(p) for p in host_closure(native)}
    if any(name not in graph or sha(graph[name]) != digest for name,digest in manifest['semantic_native'].items()):
        raise ValueError('private Reference semantic native drift')
    origin = importlib.util.find_spec('omlx').origin
    if not origin or not Path(origin).resolve().is_relative_to(exported):
        raise ValueError('private Reference import origin drift')
    if not os.environ.get('DS41F_CHECKPOINT'):
        raise ValueError('explicit official external checkpoint required')
    for name,digest in CHECKPOINT.items():
        if sha(cfg.checkpoint_path/name) != digest:
            raise ValueError('private Reference checkpoint drift')
    assets = checkpoint_inventory(cfg.checkpoint_path)
    if assets != manifest['checkpoint']:
        raise ValueError('private Reference weights inventory drift')
    verify_checkpoint_bytes(cfg.checkpoint_path, assets)
    return dict(schema='ds41f.off.reference-execution.v2',status='PASS',
                lane='R1 private qualification, not production startup', checkpoint=assets)


def inspect(cfg):
    from .execution_identity import inspect_payload
    authority, compatibility = inspect_payload()
    if authority['profile'] != 'standard-off':
        raise ValueError('standard-off requires its separate installed execution package')
    if not os.environ.get('DS41F_CHECKPOINT'):
        raise ValueError('explicit official external DS41F_CHECKPOINT required')
    if any(os.environ.get(key) for key in ('DS41F_OMLX_PATH','DS41F_RECIPE_PATH','PYTHONPATH')):
        raise ValueError('external import paths unavailable in installed OFF')
    site, recipe = paths()
    if cfg.omlx_path.resolve() != site.resolve() or cfg.recipe_path.resolve() != recipe.resolve():
        raise ValueError('OFF imports must use provisioned installed resources')
    metadata = {**CHECKPOINT, **authority['checkpoint_metadata']}
    if {p.name for p in cfg.checkpoint_path.glob('*.json')} != set(metadata):
        raise ValueError('OFF checkpoint metadata set drift')
    for name, digest in metadata.items():
        if sha(cfg.checkpoint_path/name) != digest:
            raise ValueError(f'checkpoint metadata drift: {name}')
    if any((cfg.checkpoint_path/name).exists() for name in ('chat_template.jinja', 'chat_templates')):
        raise ValueError('OFF checkpoint template override')
    assets = checkpoint_inventory(cfg.checkpoint_path)
    if assets != authority['checkpoint']:
        raise ValueError('checkpoint shard identity drift')
    verify_checkpoint_bytes(cfg.checkpoint_path, assets)
    from .config import validate_runtime_config
    validation = validate_runtime_config(cfg)
    if any(v['status'] == 'FAIL' for v in validation):
        raise ValueError('OFF configuration invalid')
    return dict(schema='ds41f.off.delivery.v2', status='PASS', profile='standard-off',
                config=cfg.to_json(), validation=validation, compatibility=compatibility,
                build=authority['provenance'], checkpoint=assets)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['inspect'])
    p.parse_args()
    from .config import load_runtime_config
    print(json.dumps(inspect(load_runtime_config()),indent=2))


if __name__ == '__main__':
    main()
