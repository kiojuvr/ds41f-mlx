"""Fail-closed MTP environment/source identity, separate from OFF provenance.

Seal records a locally built artifact, not a release qualification or signature.
Operator controls this environment. Unknown drift requires rebuild + acceptance.
"""
import argparse
from dataclasses import replace
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from .config import load_runtime_config
from .mtp_setup import ROOT, SOURCES, sha
from .mtp_profile import PROFILE, LIMITS, WEATHER

RECORD = Path(sys.prefix)/'share/ds41f-mtp/identity.json'
CHECKPOINT_REVISION = 'dba1be0a40aa45a94ad051997016db3960a90277'
CHECKPOINT = {'config.json':'8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879',
    'model.safetensors.index.json':'74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8',
    'tokenizer.json':'c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b'}


def config(host=None, port=None):
    allowed = {'DS41F_CHECKPOINT','DS41F_KV_ROOT','DS41F_HOST','DS41F_PORT',
               'DS41F_MAX_LIVE_SESSIONS','DS41F_TRACE_HISTORY_LIMIT',
               'DS41F_ENABLE_DIAGNOSTIC_ENDPOINTS','DS41F_MODEL_ID'}
    unknown = sorted(k for k in os.environ if
        (k.startswith('DS41F_') and k not in allowed) or k.startswith(('OMLX_','UVICORN_','DYLD_')))
    if unknown:
        raise ValueError(f'unqualified profile environment settings: {unknown}')
    if os.environ.get('WEB_CONCURRENCY', '1') != '1':
        raise ValueError('MTP requires one worker')
    if os.environ.get('DS41F_TRACE_HISTORY_LIMIT', '32') != '32':
        raise ValueError('MTP trace budget is fixed at 32')
    for name in ('DS41F_OMLX_PATH','DS41F_RECIPE_PATH'):
        if os.environ.get(name):
            raise ValueError(f'{name} unavailable in MTP; provisioned packages own imports')
    if os.environ.get('PYTHONPATH'):
        raise ValueError('PYTHONPATH unavailable in MTP')
    if os.environ.get('DS41F_MAX_LIVE_SESSIONS', '1') != '1':
        raise ValueError('MTP requires exactly one live session')
    cfg = load_runtime_config()
    host = host if host is not None else cfg.host
    port = port if port is not None else cfg.port
    if host != '127.0.0.1' or type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('MTP requires literal 127.0.0.1 and a fixed port')
    if cfg.enable_diagnostics or cfg.model_id != 'deepseek-v4.1-flash':
        raise ValueError('diagnostics/custom model configuration unavailable in MTP')
    spec = importlib.util.find_spec('omlx')
    if spec is None or not spec.origin:
        raise ValueError('provision MTP source dependencies first')
    parent = Path(spec.origin).resolve().parent.parent
    if not parent.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError('oMLX must be installed inside the selected environment')
    return replace(cfg, omlx_path=parent,
        recipe_path=Path(sys.prefix)/'share/ds41f-mtp/recipe', host=host, port=port,
        max_live_sessions=1, trace_history_limit=32, mtp='ON', dspark='ON', speculative_decode='ON')


def inventory(root):
    root = Path(root)
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.DS_Store')}


def dylibs(native):
    seen = {}
    def visit(path):
        path = Path(path).resolve()
        if str(path) in seen:
            return
        seen[str(path)] = sha(path)
        output = subprocess.check_output(['otool','-L',str(path)], text=True)
        install_id = subprocess.check_output(['otool','-D',str(path)], text=True).splitlines()[1:]
        for line in output.splitlines()[1:]:
            dep = line.strip().split(' (')[0]
            if dep in install_id:
                continue
            if dep.startswith('/opt/homebrew/'):
                visit(dep)
            elif dep.startswith(('@rpath/', '@loader_path/', '@executable_path/')):
                # Homebrew OpenCV's @rpath dependencies resolve in its own lib
                # directory. Reject unresolved/search-path-dependent alternatives.
                candidate = path.parent/Path(dep).name
                if candidate.exists():
                    visit(candidate)
                else:
                    raise ValueError(f'unresolved native link {dep}')
    visit(native)
    return seen


def snapshot():
    cfg = config()
    import deepseek_recipe
    from deepseek_recipe import _native
    native = Path(_native.__file__).resolve()
    if not native.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError('native recipe must be installed in this environment')
    source = {str(p.relative_to(ROOT)):sha(p) for tree in ('ds41f_mlx','native')
              for p in sorted((ROOT/tree).rglob('*')) if p.is_file() and
              p.suffix in ('.py','.metal','.c','.cc','.cpp','.h','.hpp') and
              not any(part.startswith('build') for part in p.relative_to(ROOT/tree).parts)}
    source.update({str(p.relative_to(ROOT)):sha(p) for p in
                   (ROOT/'pyproject.toml', SOURCES/'sources.json', SOURCES/'requirements.lock')})
    lock = json.loads((SOURCES/'sources.json').read_text())
    for item in lock['sources'].values():
        if sha(SOURCES/item['archive']) != item['sha256']:
            raise ValueError('unapproved source export')
    # Imported Python must match the delivered candidate, not merely a package
    # version. Optional compiled oMLX kernels are outside this stock-JIT profile.
    import tarfile
    for name, prefix, installed in (
            ('omlx','omlx/',cfg.omlx_path/'omlx'),
            ('recipe','deepseek-recipe-python/python/deepseek_recipe/',Path(deepseek_recipe.__file__).parent)):
        with tarfile.open(SOURCES/lock['sources'][name]['archive']) as archive:
            expected = {}
            for member in archive.getmembers():
                path = member.name.removeprefix('./')
                if path.startswith(prefix) and path.endswith(('.py','.pyi')):
                    relative = path[len(prefix):]
                    expected[relative] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
            for relative, digest in expected.items():
                if sha(installed/relative) != digest:
                    raise ValueError(f'unapproved {name} candidate content: {relative}')
        if name == 'omlx' and any(installed.rglob('*.so')):
            raise ValueError('compiled oMLX kernels unavailable in stock-JIT profile')
    tokenizer = cfg.recipe_path/'static/tokenizers/v41/tokenizer.json'
    if sha(tokenizer) != '81f64d1248a68ce3663e07ab3ee48b851e5df0e32d27cb98e4c9a268151e8d99':
        raise ValueError('wrong recipe V41 tokenizer')
    dependencies = {}
    for name in ('mlx','mlx_lm','mlx_vlm','numpy','tokenizers','fastapi','starlette','anyio','pydantic',
                 'pydantic_core','uvicorn','h11','transformers','safetensors','huggingface_hub'):
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.submodule_search_locations:
            raise ValueError(f'missing runtime dependency {name}')
        roots = list(spec.submodule_search_locations)
        if len(roots) != 1 or not Path(roots[0]).resolve().is_relative_to(Path(sys.prefix).resolve()):
            raise ValueError(f'unapproved dependency import origin {name}')
        dependencies[name] = str(Path(roots[0]).resolve().relative_to(cfg.omlx_path))
    links = dylibs(native)
    if not any(Path(p).name == 'libopencv_core.4.14.0.dylib' for p in links):
        raise ValueError('full target-native recipe/OpenCV link closure required')
    if not hasattr(deepseek_recipe.StreamProcessor, 'preview_tokens') or not hasattr(deepseek_recipe.StreamProcessor, 'semantic_snapshot'):
        raise ValueError('native recipe lacks qualified semantic preview capability')
    return dict(schema='ds41f.mtp.identity.v1', profile=PROFILE, limits=LIMITS,
                tool=WEATHER, sources=lock, runtime=source,
                dependencies=dependencies, package_payload=inventory(cfg.omlx_path),
                omlx=inventory(cfg.omlx_path/'omlx'), recipe=inventory(Path(deepseek_recipe.__file__).parent),
                native_sha256=sha(native), native_links=links, tokenizer_sha256=sha(tokenizer),
                packages=dict(sorted((d.metadata['Name'].lower(),d.version) for d in metadata.distributions())),
                python=platform.python_version(), platform=platform.platform())


def inspect(cfg, *, checkpoint=True):
    if (ROOT/'release/promotion.json').exists():
        from .projection import verify
        verify()
    if checkpoint and not os.environ.get('DS41F_CHECKPOINT'):
        raise ValueError('DS41F_CHECKPOINT must explicitly identify the official external asset')
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise ValueError('qualified MTP platform is Apple Silicon macOS')
    hardware = subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
    memory = int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True))
    if hardware != 'Apple M3 Ultra' or memory < 500_000_000_000:
        raise ValueError('MTP qualification requires M3 Ultra 512 GB class')
    saved = json.loads(RECORD.read_text())
    actual = snapshot()
    expected = saved['identity']
    if actual != expected:
        changed = [k for k in actual if actual[k] != expected.get(k)]
        raise ValueError(f'unapproved executable identity drift: {changed}')
    if actual['python'] != '3.13.15' or actual['packages'].get('mlx') != '0.32.2' or actual['packages'].get('mlx-lm') != '0.31.4.dev132+g94cdcae13':
        raise ValueError('unqualified Python/MLX dependency identity')
    assets = {}
    if checkpoint:
        for file, digest in CHECKPOINT.items():
            if sha(cfg.checkpoint_path/file) != digest:
                raise ValueError(f'unqualified checkpoint metadata: {file}')
        assets = checkpoint_inventory(cfg.checkpoint_path)
    identity_digest = hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest()
    return dict(status='PASS', profile=PROFILE, environment_valid=True,
                release_qualified=False, identity_sha256=identity_digest,
                config=cfg.to_json(), identity=actual, checkpoint=assets, build=saved['build'],
                hardware=dict(cpu=hardware, memory_bytes=memory),
                actual_origins={name:importlib.util.find_spec(name).origin for name in
                                ('ds41f_mlx','omlx','deepseek_recipe','deepseek_recipe._native','mlx_lm')})


def checkpoint_inventory(root):
    shards = []
    for p in sorted(root.glob('model-*.safetensors')):
        meta = root/'.cache/huggingface/download'/(p.name+'.metadata')
        lines = meta.read_text().splitlines() if meta.exists() else []
        if (len(lines) < 2 or lines[0] != CHECKPOINT_REVISION or len(lines[1]) != 64 or
                any(c not in '0123456789abcdef' for c in lines[1])):
            raise ValueError('qualified checkpoint source revision/LFS hash metadata required')
        shards.append(dict(name=p.name, bytes=p.stat().st_size, source_revision=lines[0], lfs_sha256=lines[1]))
    if len(shards) != 48 or len({s['source_revision'] for s in shards}) != 1:
        raise ValueError('official 48-shard source inventory required')
    return dict(metadata=CHECKPOINT, shards=shards,
                verification='metadata/source inventory only; weights not freshly rehashed')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['seal','inspect'])
    parser.add_argument('--wheel', type=Path)
    args = parser.parse_args(argv)
    if args.action == 'seal':
        if args.wheel is None:
            parser.error('--wheel required for seal')
        build = dict(wheel_sha256=sha(args.wheel), host_linked=True,
                     rustc=subprocess.check_output(['rustc','-Vv'],text=True),
                     cargo=subprocess.check_output(['cargo','-V'],text=True),
                     cmake=subprocess.check_output(['cmake','--version'],text=True),
                     opencv=subprocess.check_output(['pkg-config','--modversion','opencv4'],text=True))
        RECORD.parent.mkdir(parents=True,exist_ok=True)
        RECORD.write_text(json.dumps(dict(identity=snapshot(),build=build),indent=2)+'\n')
        print(RECORD)
    else:
        print(json.dumps(inspect(config()),indent=2))


if __name__ == '__main__':
    main()
