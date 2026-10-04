"""Source-delivered standard-OFF provenance; no donor Git checkout authority."""
import argparse
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
import tarfile

from .mtp_identity import CHECKPOINT, checkpoint_inventory, dylibs, inventory
from .mtp_setup import ROOT, SOURCES, sha

RECORD = Path(sys.prefix)/'share/ds41f-off/identity.json'


def paths():
    return Path(sysconfig.get_paths()['purelib']), Path(sys.prefix)/'share/ds41f-mtp/recipe'


def snapshot():
    site, recipe = paths()
    lock = json.loads((SOURCES/'sources.json').read_text())
    for dep in lock['sources'].values():
        if sha(SOURCES/dep['archive']) != dep['sha256']:
            raise ValueError('dependency source export drift')
    from hashlib import sha256
    for archive, prefix, installed in (
        (ROOT/'reference/R1/off-source.tar.gz', 'omlx/', site/'omlx'),
        (SOURCES/lock['sources']['recipe']['archive'], 'deepseek-recipe-python/python/deepseek_recipe/', site/'deepseek_recipe')):
        expected = {}
        with tarfile.open(archive) as tar:
            for m in tar.getmembers():
                name = m.name.removeprefix('./')
                if name.startswith(prefix) and name.endswith(('.py','.pyi')):
                    expected[name[len(prefix):]] = sha256(tar.extractfile(m).read()).hexdigest()
        actual = {p: h for p,h in inventory(installed).items() if p.endswith(('.py','.pyi'))}
        if actual != expected:
            raise ValueError('installed attributed execution source drift')
    if any((site/'omlx').rglob('*.so')):
        raise ValueError('unqualified compiled oMLX kernels')
    origins = {}
    for name in ('omlx','deepseek_recipe','deepseek_recipe._native','mlx','mlx_lm'):
        spec = importlib.util.find_spec(name)
        if not spec or not spec.origin or not Path(spec.origin).resolve().is_relative_to(site.resolve()):
            raise ValueError(f'non-provisioned module origin: {name}')
        origins[name] = str(Path(spec.origin).resolve().relative_to(site))
    if sha(recipe/'static/tokenizers/v41/tokenizer.json') != '81f64d1248a68ce3663e07ab3ee48b851e5df0e32d27cb98e4c9a268151e8d99':
        raise ValueError('recipe tokenizer drift')
    if platform.python_version() != '3.13.15':
        raise ValueError('unqualified Python')
    for line in (SOURCES/'requirements.lock').read_text().splitlines():
        if '==' in line:
            name, version = line.split('==')
            if metadata.version(name) != version:
                raise ValueError(f'dependency version drift: {name}')
    if metadata.version('mlx-lm') != '0.31.4.dev132+g94cdcae13':
        raise ValueError('MLX-LM revision drift')
    runtime = {str(p.relative_to(ROOT)):sha(p) for tree in ('ds41f_mlx','native')
               for p in sorted((ROOT/tree).rglob('*')) if p.is_file() and
               p.suffix in ('.py','.metal','.c','.cc','.cpp','.h','.hpp') and
               not any(x.startswith('build') or x == '__pycache__' for x in p.relative_to(ROOT/tree).parts)}
    return dict(runtime=runtime, payload=inventory(site), origins=origins,
                native_links=dylibs(site/origins['deepseek_recipe._native']),
                tokenizer_sha256=sha(recipe/'static/tokenizers/v41/tokenizer.json'),
                source_lock_sha256=sha(SOURCES/'sources.json'),
                off_source_sha256=sha(ROOT/'reference/R1/off-source.tar.gz'))


def inspect(cfg):
    from .projection import verify
    if (ROOT/'release/promotion.json').exists():
        promotion = verify()
    else:
        promotion = None
    if not os.environ.get('DS41F_CHECKPOINT'):
        raise ValueError('explicit official external DS41F_CHECKPOINT required')
    for key in ('DS41F_OMLX_PATH','DS41F_RECIPE_PATH','PYTHONPATH'):
        if os.environ.get(key):
            raise ValueError(f'{key} is unsupported by source-delivered OFF')
    site, recipe = paths()
    if cfg.omlx_path.resolve() != site.resolve() or cfg.recipe_path.resolve() != recipe.resolve():
        raise ValueError('OFF must use provisioned source delivery')
    saved = json.loads(RECORD.read_text())
    actual = snapshot()
    if actual != saved['identity']:
        raise ValueError('sealed OFF executable identity drift; rebuild and qualify in ds41f-mlx')
    for name, digest in CHECKPOINT.items():
        if sha(cfg.checkpoint_path/name) != digest:
            raise ValueError(f'checkpoint metadata drift: {name}')
    from .config import validate_runtime_config
    validation = validate_runtime_config(cfg)
    if any(v['status'] == 'FAIL' for v in validation):
        raise ValueError('OFF configuration invalid')
    return dict(schema='ds41f.off.delivery.v1', status='PASS', profile='standard-off',
                promotion=promotion, config=cfg.to_json(), validation=validation,
                identity=actual, build=saved['build'], checkpoint=checkpoint_inventory(cfg.checkpoint_path),
                actual_origins={name:importlib.util.find_spec(name).origin for name in
                                ('ds41f_mlx','omlx','deepseek_recipe','deepseek_recipe._native','mlx_lm')})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['seal','inspect'])
    p.add_argument('--wheel', type=Path)
    a = p.parse_args()
    if a.action == 'seal':
        if a.wheel is None: p.error('--wheel required')
        record = dict(identity=snapshot(), build=dict(wheel_sha256=sha(a.wheel),
            rustc=subprocess.check_output(['rustc','-Vv'],text=True),
            opencv=subprocess.check_output(['pkg-config','--modversion','opencv4'],text=True)))
        RECORD.parent.mkdir(parents=True,exist_ok=True)
        RECORD.write_text(json.dumps(record,indent=2)+'\n')
    else:
        from .config import load_runtime_config
        print(json.dumps(inspect(load_runtime_config()),indent=2))


if __name__ == '__main__':
    main()
