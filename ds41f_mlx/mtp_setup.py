"""Target-native source-clone provisioning; build inputs are repository authority."""
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--venv', type=Path, required=True)
    p.add_argument('--build-dir', type=Path, required=True)
    args = p.parse_args(argv)
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        p.error('qualified setup requires Apple Silicon macOS')
    venv, build = args.venv.resolve(), args.build_dir.resolve()
    lock = json.loads((SOURCES/'sources.json').read_text())
    build.mkdir(parents=True, exist_ok=True)
    for name, spec in lock['sources'].items():
        archive = SOURCES/spec['archive']
        if sha(archive) != spec['sha256']:
            raise ValueError('unapproved dependency source archive')
        target = build/name
        if target.exists():
            raise ValueError('build-dir must have no prior dependency source exports')
        target.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(target, filter='data')
    if venv.exists():
        raise ValueError('target venv must be absent; construct a fresh locked environment')
    run('uv', 'venv', '--python', '/opt/homebrew/opt/python@3.13/bin/python3.13', venv)
    python = venv/'bin/python'
    if subprocess.check_output([str(python),'-c','import platform; print(platform.python_version())'],text=True).strip() != '3.13.15':
        raise ValueError('qualified setup requires Python 3.13.15; recreate the target venv')
    run('uv','pip','install','--python',python,'-r',SOURCES/'requirements.lock')
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
    if not subprocess.check_output(['rustc','-V'],text=True).startswith('rustc 1.98.1 '):
        raise ValueError('qualified native build requires Rust 1.98.1')
    run(venv/'bin/maturin','build','--release','--locked','--skip-auditwheel',
        '--manifest-path',build/'recipe/deepseek-recipe-python/Cargo.toml',
        '-i',python,'-o',build/'wheels',env=env)
    wheels = list((build/'wheels').glob('*.whl'))
    if len(wheels) != 1:
        raise ValueError('exactly one native recipe wheel required')
    run('uv','pip','install','--reinstall','--no-deps','--python',python,wheels[0],build/'omlx')
    run('uv','pip','install','--no-deps','--python',python,'-e',ROOT)
    resource = venv/'share/ds41f-mtp/recipe/static/tokenizers/v41'
    resource.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(build/'recipe/static/tokenizers/v41/tokenizer.json',resource/'tokenizer.json')
    # Record actual executable and link identity inside the target environment.
    run(python,'-m','ds41f_mlx.mtp_identity','seal','--wheel',wheels[0],env=env)
    print(f'Provisioned {venv}. Runtime needs neither donor checkouts nor build-dir.')


if __name__ == '__main__':
    main()
