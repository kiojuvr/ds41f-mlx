"""One-way package construction, never a runtime reseal.

Dependency and reachable native hashes must inherit the admitted candidate.
Only ds41f's newly qualified package payload is constructed from this checkout.
Run qualification before publishing the wheel; publisher/operator is trusted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ds41f_mlx.execution_identity import inventory, links
from ds41f_mlx.mtp_setup import sha


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
    site = Path(sysconfig.get_paths()['purelib'])
    import shutil
    shutil.copy2(ROOT/'release/ds41f-release.json', ROOT/'ds41f_mlx/runtime_release.json')
    payloads = {'ds41f_mlx': inventory(ROOT/'ds41f_mlx'),
                'omlx': old['omlx'], 'deepseek_recipe': old['recipe']}
    overlays = {}
    if args.profile == 'mtp':
        for label in ('compressed-growth', 'bit-exact-kernels', 'template-reuse', 'context-copy'):
            overlay = json.loads((ROOT/f'third_party/mtp/{label}.json').read_text())
            if sha(ROOT/f'third_party/mtp/{label}.patch') != overlay['patch_sha256']:
                raise ValueError(f'{label} patch drift')
            for name, identity in overlay['files'].items():
                if payloads['omlx'].get(name) != identity['before']:
                    raise ValueError(f'{label} inherited base drift: {name}')
                payloads['omlx'][name] = identity['after']
            overlays[label] = overlay
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
    # Recipe source calls ONLY these OpenCV modules. Their full Homebrew
    # dependency subgraph is conservative exact image/numerical execution.
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
    manifest = dict(schema='ds41f.installed-admission.v2', profile=args.profile, glm=glm, payloads=payloads, files=files,
                    semantic_native=semantic,
                    tokenizer_sha256='81f64d1248a68ce3663e07ab3ee48b851e5df0e32d27cb98e4c9a268151e8d99',
                    checkpoint=old['checkpoint'], checkpoint_metadata=old['checkpoint_metadata'],
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
    (ROOT/'ds41f_mlx/execution_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    args.output.mkdir(parents=True,exist_ok=True)
    subprocess.run(['uv','build','--wheel','--out-dir',str(args.output),str(ROOT)],check=True)
    # Ensure the installed execution inventory is precisely what the wheel ships.
    import zipfile
    wheels = list(args.output.glob('ds41f_mlx-*.whl'))
    if len(wheels) != 1:
        raise ValueError('ambiguous package output')
    with zipfile.ZipFile(wheels[0]) as wheel:
        shipped = {name[len('ds41f_mlx/'):]: hashlib.sha256(wheel.read(name)).hexdigest()
                   for name in wheel.namelist() if name.startswith('ds41f_mlx/')
                   and not name.endswith('/') and name != 'ds41f_mlx/execution_manifest.json'}
    if shipped != payloads['ds41f_mlx']:
        raise ValueError('wheel payload differs from package inventory')
    print(json.dumps(dict(wheel=str(wheels[0]), sha256=sha(wheels[0]))))


if __name__ == '__main__':
    main()
