"""Build the current Runtime wheel; MTP uses versions/origins, not Python seals.

OFF retains its separate historical payload authority. Neither lane enrolls
locally observed native binaries or checkpoint bytes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import sysconfig
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ds41f_mlx.execution_identity import inventory, links
from ds41f_mlx.mtp_setup import sha, OVERLAYS
from ds41f_mlx.mtp_identity import flash_1017_inventory


def off_dependency_payloads(off, pin):
    """Cross-check inherited OFF payload metadata against numerical authority."""
    for name, expected in off['payloads'].items():
        if off['versions'][name] != pin['versions'][name]:
            raise ValueError(f'OFF dependency authority version drift: {name}')
        for module, identity in pin['modules'].items():
            if module.split('.')[0] != name:
                continue
            relative = '/'.join(module.split('.')[1:])
            candidates = [path for path in expected
                          if path in ((relative+'.py', relative+'/__init__.py') if relative else ('__init__.py',))
                          or relative and path.startswith(relative+'.') and path.endswith('.so')]
            if len(candidates) != 1 or expected[candidates[0]] != identity['sha256']:
                raise ValueError(f'OFF dependency authority module drift: {module}')
        for path, identity in pin['package_resources'].get(name, {}).items():
            if expected.get(path) != identity['sha256']:
                raise ValueError(f'OFF dependency authority native drift: {name}/{path}')
    return off['payloads']


def build_wheel(manifest, output):
    """Stage package metadata without rewriting the checkout or an installation."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob('ds41f_mlx-*.whl')):
        raise ValueError('package output must contain no prior Runtime wheels')
    with tempfile.TemporaryDirectory(prefix='ds41f-package-') as scratch:
        stage = Path(scratch)
        shutil.copy2(ROOT/'pyproject.toml', stage/'pyproject.toml')
        shutil.copytree(ROOT/'ds41f_mlx', stage/'ds41f_mlx',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.DS_Store'))
        if manifest['profile'] == 'mtp':
            release = json.loads((ROOT/'release/ds41f-release.json').read_text())
            release['dependencies']['deepseek_recipe'].update(
                version=manifest['dependencies']['deepseek-recipe'],
                revision=manifest['provenance']['recipe_source']['revision'])
            (stage/'ds41f_mlx/runtime_release.json').write_text(json.dumps(release, indent=2)+'\n')
        else:
            shutil.copy2(ROOT/'release/ds41f-release.json', stage/'ds41f_mlx/runtime_release.json')
        (stage/'ds41f_mlx/execution_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
        subprocess.run(['uv', 'build', '--wheel', '--out-dir', str(output), str(stage)], check=True)
    wheels = list(output.glob('ds41f_mlx-*.whl'))
    if len(wheels) != 1:
        raise ValueError('ambiguous package output')
    return wheels[0]


def semantic_native(old):
    """Inherit the reachable Recipe/OpenCV native graph, never enroll local bytes."""
    semantic = {}
    old_by_name = {Path(path).name: digest for path, digest in old['native_links'].items()}
    def visit(path):
        path = Path(path).resolve()
        if path.name in semantic:
            return
        digest = sha(path)
        if digest != old_by_name.get(path.name):
            raise ValueError(f'no inherited semantic-native identity: {path}')
        semantic[path.name] = digest
        for target, _ in links(path):
            visit(target)
    for module in ('core', 'imgproc', 'imgcodecs'):
        visit(Path('/opt/homebrew/opt/opencv@4/lib')/f'libopencv_{module}.414.dylib')
    return semantic


def mtp_manifest(old, glm_profile):
    """Ordinary delivery metadata: locked versions plus genuine native/asset pins."""
    from ds41f_mlx.glm_profile import bundle_authority
    # Keep module names for installed-origin checks, but never Python fingerprints.
    support = json.loads((ROOT/'third_party/mtp/runtime-support-payload.json').read_text())['files']
    # The existing retained-wheel authority owns all repaired native resources.
    # Read exact artifact bytes, never enroll a locally installed library graph.
    import zipfile
    recipe_wheel = ROOT/'third_party/mtp'/old['wheel']['file']
    if sha(recipe_wheel) != old['wheel']['sha256']:
        raise ValueError('unapproved Recipe 0.2 wheel')
    with zipfile.ZipFile(recipe_wheel) as wheel:
        recipe_native = {
            (name.removeprefix('deepseek_recipe/') if name.startswith('deepseek_recipe/') else '../'+name):
                hashlib.sha256(wheel.read(name)).hexdigest()
            for name in wheel.namelist() if name.endswith(('.so', '.dylib')) and
            name.startswith(('deepseek_recipe/', 'deepseek_recipe.dylibs/'))}
    payloads = {'ds41f_mlx': {}, 'omlx': {}, 'deepseek_recipe': recipe_native}
    for source in (old['dependency_payload'], support):
        for path, digest in source.items():
            if '/' in path:
                name, relative = path.split('/', 1)
                payloads.setdefault(name, {})[relative] = digest
    native_suffixes = ('.so', '.dylib', '.metallib')
    payloads = {name: {path: digest for path, digest in files.items() if path.endswith(native_suffixes)}
                for name, files in payloads.items()}
    glm = dict(profile=glm_profile, glm_profiles={'portable': {}, 'native': {}}, glm_extension=None)
    if glm_profile == 'native':
        bundle = bundle_authority()
        glm.update(glm_profiles={'portable': {}, 'native': bundle['files']},
                   glm_extension=bundle['extension'], provenance=bundle)
    versions = {name: old['packages'][name] for name in ('omlx', 'deepseek-recipe')}
    for line in (ROOT/'third_party/mtp/requirements-normal-local.lock').read_text().splitlines():
        if line and not line.startswith('#'):
            if '==' in line:
                name, version = line.split('==')
            else:
                # pip/uv installs this exact upstream git revision from the lock.
                name, _ = line.split(' @ ')
                version = old['packages'][name]
            versions[name] = version
    return dict(schema='ds41f.installed-admission.v3', profile='mtp', glm=glm,
                payloads=payloads, files={path: None for path in support if '/' not in path},
                dependencies=versions, semantic_native={},
                tokenizer_sha256='81f64d1248a68ce3663e07ab3ee48b851e5df0e32d27cb98e4c9a268151e8d99',
                checkpoint=flash_1017_inventory(old['checkpoint']), checkpoint_metadata=old['checkpoint_metadata'],
                provenance=dict(source_commit=subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    overlays=list(OVERLAYS),
                    native_build=old['build'], recipe_source=old['recipe_source']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--profile', choices=['mtp', 'standard-off'], default='mtp')
    p.add_argument('--recipe-wheel', type=Path)
    p.add_argument('--glm-profile', choices=['native', 'portable'])
    args = p.parse_args()
    from ds41f_mlx.glm_profile import default_profile
    args.glm_profile = args.glm_profile or default_profile(args.profile)
    legacy_path = ROOT/'third_party/mtp/normal-local.json'
    old = json.loads(legacy_path.read_text())
    if args.profile == 'mtp':
        wheel = build_wheel(mtp_manifest(old, args.glm_profile), args.output)
        print(json.dumps(dict(wheel=str(wheel), sha256=sha(wheel))))
        return
    site = Path(sysconfig.get_paths()['purelib'])
    payloads = {'ds41f_mlx': inventory(ROOT/'ds41f_mlx'),
                'omlx': old['omlx'], 'deepseek_recipe': old['recipe']}
    payloads['ds41f_mlx']['runtime_release.json'] = sha(ROOT/'release/ds41f-release.json')
    overlays = {}  # OFF does not consume MTP overlays.
    if args.profile == 'standard-off':
        from ds41f_mlx.off_recipe import artifact_authority
        args.recipe_wheel, off_recipe = artifact_authority(args.recipe_wheel)
        import tarfile, zipfile
        with tarfile.open(ROOT/'reference/R1/off-source.tar.gz') as tar:
            payloads['omlx'] = {m.name.removeprefix('./').removeprefix('omlx/'): hashlib.sha256(tar.extractfile(m).read()).hexdigest()
                                for m in tar.getmembers() if m.isfile() and m.name.removeprefix('./').startswith('omlx/')}
        with zipfile.ZipFile(args.recipe_wheel) as wheel:
            payloads['deepseek_recipe'] = {name[len('deepseek_recipe/'):]: hashlib.sha256(wheel.read(name)).hexdigest()
                                         for name in wheel.namelist() if name.startswith('deepseek_recipe/') and not name.endswith('/')}
        with tarfile.open(ROOT/'third_party/mtp/recipe-source.tar.gz') as tar:
            prefix = 'deepseek-recipe-python/python/deepseek_recipe/'
            source_python = {m.name.removeprefix('./')[len(prefix):]: hashlib.sha256(tar.extractfile(m).read()).hexdigest()
                             for m in tar.getmembers() if m.isfile() and m.name.removeprefix('./').startswith(prefix)
                             and m.name.endswith(('.py', '.pyi'))}
        if {name: digest for name,digest in payloads['deepseek_recipe'].items() if name.endswith(('.py','.pyi'))} != source_python:
            raise ValueError('OFF recipe wheel attribution differs from delivered base source')
    from ds41f_mlx.glm_profile import bundle_authority, PREFIX
    glm = dict(profile=args.glm_profile, glm_profiles={'portable': {}, 'native': {}}, glm_extension=None)
    if args.glm_profile == 'native':
        bundle = bundle_authority()
        glm.update(glm_profiles={'portable': {}, 'native': bundle['files']},
                   glm_extension=bundle['extension'], provenance=bundle)
        for name, identity in bundle['files'].items():
            payloads['omlx'][PREFIX+name] = identity['sha256']
    for name in {path.split('/')[0] for path in old['dependency_payload']}:
        payloads[name] = {path[len(name)+1:]: digest for path, digest in old['dependency_payload'].items()
                          if path.startswith(name+'/')}
    if args.profile == 'standard-off':
        # OFF numerical admission predates the MTP normal-local dependency set.
        # Inherit its independently qualified payload, not the current shell.
        off = json.loads((ROOT/'third_party/mtp/standard-off-dependencies.json').read_text())
        pin = json.loads((ROOT/'ds41f_mlx/runtime/admitted_resources.json').read_text())
        payloads.update(off_dependency_payloads(off, pin))
    support = json.loads((ROOT/'third_party/mtp/runtime-support-payload.json').read_text())['files']
    for name in {path.split('/')[0] for path in support if '/' in path}:
        payloads[name] = {path[len(name)+1:]: digest for path, digest in support.items()
                          if path.startswith(name+'/')}
    files = {path: digest for path, digest in support.items() if '/' not in path}
    if any(sha(site/path) != digest for path, digest in files.items()):
        raise ValueError('runtime support file drift')
    for name, expected in payloads.items():
        if name != 'ds41f_mlx' and inventory(site/name) != expected:
            raise ValueError(f'candidate dependency drift during packaging: {name}')
    manifest = dict(schema='ds41f.installed-admission.v2', profile=args.profile, glm=glm, payloads=payloads, files=files,
                    semantic_native=semantic_native(old),
                    tokenizer_sha256='81f64d1248a68ce3663e07ab3ee48b851e5df0e32d27cb98e4c9a268151e8d99',
                    checkpoint=flash_1017_inventory(old['checkpoint']), checkpoint_metadata=old['checkpoint_metadata'],
                    provenance=dict(compressed_growth=overlays.get('compressed-growth'),
                                    bit_exact_kernels=overlays.get('bit-exact-kernels'),
                                    template_reuse=overlays.get('template-reuse'), context_copy=overlays.get('context-copy'), source_commit_is_base=True, lockfiles={name: sha(ROOT/'third_party/mtp'/name) for name in ('sources.json', 'requirements.lock', 'requirements-normal-local.lock')},
                                    package_build_input_sha256=sha(ROOT/'pyproject.toml'),
                                    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                                    source_payload_sha256=hashlib.sha256(json.dumps(payloads['ds41f_mlx'],sort_keys=True).encode()).hexdigest(),
                                    native_build=old['build'] if args.profile == 'mtp' else dict(wheel_sha256=sha(args.recipe_wheel), artifact_authority=off_recipe),
                                    recipe_source=old['recipe_source'] if args.profile == 'mtp' else json.loads((ROOT/'third_party/mtp/sources.json').read_text())['sources']['recipe'],
                                    off_dependency_authority_sha256=sha(ROOT/'third_party/mtp/standard-off-dependencies.json') if args.profile == 'standard-off' else None),
                    evidence=dict(inherited_candidate_sha256=sha(legacy_path), reference='R1'))
    wheel_path = build_wheel(manifest, args.output)
    # OFF retains exact Python payload admission.
    import zipfile
    with zipfile.ZipFile(wheel_path) as wheel:
        shipped = {name[len('ds41f_mlx/'):]: hashlib.sha256(wheel.read(name)).hexdigest()
                   for name in wheel.namelist() if name.startswith('ds41f_mlx/')
                   and not name.endswith('/') and name != 'ds41f_mlx/execution_manifest.json'}
    if shipped != payloads['ds41f_mlx']:
        raise ValueError('wheel payload differs from package inventory')
    print(json.dumps(dict(wheel=str(wheel_path), sha256=sha(wheel_path))))


if __name__ == '__main__':
    main()
