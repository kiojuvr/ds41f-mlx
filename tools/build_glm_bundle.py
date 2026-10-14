"""Publisher-only GLM build from SHA-admitted export; never a runtime reseal.

Freeze the resulting tar and receipt after affected acceptance. Provisioning copies
these exact bytes, not a new local compilation. No other optional kernels built.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ds41f_mlx.mtp_setup import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    work, output = args.work.resolve(), args.output.resolve()
    work.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    authority = json.loads((ROOT/'third_party/mtp/sources.json').read_text())['sources']['omlx']
    archive = ROOT/'third_party/mtp'/authority['archive']
    if sha(archive) != authority['sha256']:
        raise ValueError('unadmitted oMLX source')
    source = work/'source'
    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers() if m.isfile() and
                   m.name.removeprefix('./').startswith('omlx/custom_kernels/')]
        tar.extractall(source, members=members, filter='data')
    inputs = {str(path.relative_to(source)): sha(path)
              for folder in ('glm_moe_dsa', 'common')
              for path in sorted((source/'omlx/custom_kernels'/folder/'csrc').rglob('*'))
              if path.is_file()}
    commands = []
    def run(*command):
        command = list(map(str, command))
        commands.append(command)
        subprocess.run(command, check=True)
    env = work/'env'
    run('uv', 'venv', '--python', '/opt/homebrew/opt/python@3.13/bin/python3.13', env)
    python = env/'bin/python'
    run('uv', 'pip', 'install', '--python', python,
        'mlx==0.32.2', 'nanobind==2.15.0', 'cmake==4.3.1')
    bundle = work/'bundle'
    run(env/'bin/cmake', '-S', source/'omlx/custom_kernels/glm_moe_dsa/csrc',
        '-B', work/'build', f'-DPython_EXECUTABLE={python}', '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_OSX_DEPLOYMENT_TARGET=15.0', f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={bundle}')
    run(env/'bin/cmake', '--build', work/'build', '-j', '4')
    names = ('_ext.cpython-313-darwin.so', 'libomlx_glm_kernel_ops.dylib', 'omlx_glm_kernels.metallib')
    if {p.name for p in bundle.iterdir()} != set(names):
        raise ValueError('unexpected GLM build output')
    # CMake embeds transient build/env rpaths. Only installed sibling/MLX
    # search paths belong to production. Keep source and compiler flags unchanged.
    for name in names[:2]:
        path = bundle/name
        lines = subprocess.check_output(['otool', '-l', str(path)], text=True).splitlines()
        for i, line in enumerate(lines):
            if line.strip() == 'cmd LC_RPATH':
                rpath = lines[i+2].strip().split('path ', 1)[1].split(' (offset', 1)[0]
                if rpath.startswith('/'):
                    run('install_name_tool', '-delete_rpath', rpath, path)
        run('codesign', '--force', '--sign', '-', path)
    tarpath = output/'glm-native.tar.gz'
    with tarfile.open(tarpath, 'w:gz') as tar:
        for name in names:
            tar.add(bundle/name, arcname=name)
    files = {name: dict(size=(bundle/name).stat().st_size, sha256=sha(bundle/name)) for name in names}
    def capture(*cmd):
        return subprocess.check_output(cmd, text=True).strip()
    receipt = dict(schema='ds41f.glm-bundle.v1', profile='native', archive=tarpath.name,
                   archive_sha256=sha(tarpath), files=files, extension=names[0],
                   provenance=dict(source=authority, inputs=inputs, commands=commands,
                       python=capture(str(python), '-V'), xcode=capture('xcodebuild', '-version'),
                       compiler=capture('clang', '--version'), sdk=capture('xcrun', '--show-sdk-version'),
                       mlx='0.32.2', nanobind='2.15.0', cmake='4.3.1', deployment_target='15.0',
                       build_owner_sha256=sha(Path(__file__))))
    (output/'glm-native.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    print(json.dumps(files, sort_keys=True))


if __name__ == '__main__':
    main()
