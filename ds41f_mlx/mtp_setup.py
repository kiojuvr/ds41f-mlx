"""Provision a fresh package-fixed MTP installation from repository build inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / 'third_party/mtp'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(*args, env=None):
    subprocess.run([str(a) for a in args], check=True, env=env)


OVERLAYS = ('compressed-growth', 'bit-exact-kernels', 'template-reuse', 'context-copy')


def apply_mtp_patches(source):
    """Apply the ordered overlays forward-only, without fuzz or interaction."""
    for label in OVERLAYS:
        subprocess.run(['patch', '--batch', '--forward', '--fuzz=0', '-p1', '-i',
                        str(SOURCES/f'{label}.patch')], cwd=source, check=True)


def main(argv=None):
    from .mtp_identity import DEFAULT_VARIANT, VARIANTS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=['mtp-singleton-v1'], default='mtp-singleton-v1',
                   help='MTP installation for singleton/ordinary serving; Standard-OFF is retired')
    p.add_argument('--checkpoint-variant', choices=VARIANTS, default=DEFAULT_VARIANT)
    p.add_argument('--glm-profile', choices=['native', 'portable'],
                   help='explicit execution distribution; no fallback')
    p.add_argument('--venv', type=Path, required=True)
    p.add_argument('--build-dir', type=Path, required=True)
    args = p.parse_args(argv)
    from .glm_profile import default_profile
    args.glm_profile = args.glm_profile or default_profile('mtp')
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        p.error('qualified setup requires Apple Silicon macOS')
    venv, build = args.venv.resolve(), args.build_dir.resolve()
    if venv.exists():
        raise ValueError('target venv must be absent; construct a fresh locked environment')
    lock = json.loads((SOURCES/'sources.json').read_text())
    normal = json.loads((SOURCES/'normal-local.json').read_text())
    lock['sources']['recipe'] = normal['recipe_source']
    build.mkdir(parents=True, exist_ok=True)
    for name, spec in lock['sources'].items():
        target = build/name
        if target.exists():
            raise ValueError('build-dir must have no prior dependency source exports')
        target.mkdir()
        # Ordinary MTP sources have revision provenance, not SHA admission.
        with tarfile.open(SOURCES/spec['archive']) as tar:
            tar.extractall(target, filter='data')
    apply_mtp_patches(build/'omlx')
    patch = ROOT/normal['recipe_source']['patch']
    subprocess.run(['git', 'apply', '--check', str(patch)], cwd=build/'recipe', check=True)
    subprocess.run(['git', 'apply', str(patch)], cwd=build/'recipe', check=True)
    run('uv', 'venv', '--python', '/opt/homebrew/opt/python@3.13/bin/python3.13', venv)
    python = venv/'bin/python'
    if subprocess.check_output([str(python),'-c','import platform; print(platform.python_version())'],text=True).strip() != '3.13.15':
        raise ValueError('qualified setup requires Python 3.13.15; recreate the target venv')
    run('uv','pip','install','--python',python,'-r',SOURCES/'requirements-normal-local.lock')
    env = dict(os.environ)
    for key in list(env):
        if key in ('DOCS_RS','PYTHONPATH','WEB_CONCURRENCY','RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS',
                   'RUSTC','RUSTC_WRAPPER','RUSTC_WORKSPACE_WRAPPER','CARGO_BUILD_TARGET',
                   'CFLAGS','CXXFLAGS','LDFLAGS') or key.startswith(('DS41F_','OMLX_','UVICORN_','DYLD_')):
            env.pop(key, None)
    env.update(PKG_CONFIG_PATH='/opt/homebrew/opt/opencv@4/lib/pkgconfig',
               OpenCV_DIR='/opt/homebrew/opt/opencv@4/lib/cmake/opencv4',
               CMAKE_PREFIX_PATH='/opt/homebrew/opt/opencv@4',
               LIBCLANG_PATH='/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib',
               CARGO_TARGET_DIR=str(build/'cargo'))
    if subprocess.check_output(['pkg-config','--modversion','opencv4'],env=env,text=True).strip() != '4.14.0':
        raise ValueError('qualified native build requires OpenCV 4.14.0')
    wheel = SOURCES/normal['wheel']['file']
    if sha(wheel) != normal['wheel']['sha256']:
        raise ValueError('unapproved normal-local native wheel')
    run('uv','pip','install','--reinstall','--no-deps','--python',python,wheel,build/'omlx')
    resource = venv/'share/ds41f-mtp/recipe/static/tokenizers/v41'
    resource.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(build/'recipe/static/tokenizers/v41/tokenizer.json',resource/'tokenizer.json')
    site = Path(subprocess.check_output([str(python), '-c', 'import sysconfig; print(sysconfig.get_paths()["purelib"])'], text=True).strip())
    from .glm_profile import provision
    provision(site, args.glm_profile)
    run(python, ROOT/'tools/build_execution_package.py', '--output', build/'runtime-wheels',
        '--glm-profile', args.glm_profile, '--checkpoint-variant', args.checkpoint_variant)
    runtime_wheels = list((build/'runtime-wheels').glob('ds41f_mlx-*.whl'))
    if len(runtime_wheels) != 1:
        raise ValueError('exactly one execution package required')
    run('uv','pip','install','--no-deps','--python',python,runtime_wheels[0])
    subprocess.run([str(python), '-I', '-c',
                    'from ds41f_mlx.execution_identity import inspect_payload; inspect_payload()'],
                   check=True, env=env)
    print(f'Provisioned {venv} ({args.checkpoint_variant}). Runtime needs neither donor checkouts nor build-dir.')


if __name__ == '__main__':
    main()
