"""Build a package-fixed MTP Runtime wheel, without Python-source seals.

Native payloads, dependencies, tokenizer and the selected official checkpoint
remain exact publisher authorities. Standard-OFF production delivery is retired.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ds41f_mlx.mtp_setup import sha, OVERLAYS
from ds41f_mlx.mtp_identity import (
    DEFAULT_VARIANT, VARIANTS, checkpoint_variant, flash_1017_inventory,
)


def build_wheel(manifest, output):
    """Stage package metadata without rewriting the checkout or an installation."""
    if manifest['profile'] != 'mtp':
        raise ValueError('Standard-OFF production packaging is retired')
    checkpoint_variant(manifest)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob('ds41f_mlx-*.whl')):
        raise ValueError('package output must contain no prior Runtime wheels')
    with tempfile.TemporaryDirectory(prefix='ds41f-package-') as scratch:
        stage = Path(scratch)
        shutil.copy2(ROOT/'pyproject.toml', stage/'pyproject.toml')
        shutil.copytree(ROOT/'ds41f_mlx', stage/'ds41f_mlx',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.DS_Store'))
        release = json.loads((ROOT/'release/ds41f-release.json').read_text())
        release['dependencies']['deepseek_recipe'].update(
            version=manifest['dependencies']['deepseek-recipe'],
            revision=manifest['provenance']['recipe_source']['revision'])
        (stage/'ds41f_mlx/runtime_release.json').write_text(json.dumps(release, indent=2)+'\n')
        (stage/'ds41f_mlx/execution_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
        subprocess.run(['uv', 'build', '--wheel', '--out-dir', str(output), str(stage)], check=True)
    wheels = list(output.glob('ds41f_mlx-*.whl'))
    if len(wheels) != 1:
        raise ValueError('ambiguous package output')
    return wheels[0]


def mtp_manifest(old, glm_profile, variant=DEFAULT_VARIANT):
    """Ordinary delivery metadata: locked versions plus genuine native/asset pins."""
    from ds41f_mlx.glm_profile import bundle_authority
    selected = VARIANTS[variant]
    checkpoint = flash_1017_inventory(old['checkpoint']) if variant == 'flash-1017' else old['checkpoint']
    selection = dict(checkpoint_variant=selected.name, numerics=selected.numerics,
                     checkpoint=checkpoint, checkpoint_metadata=old['checkpoint_metadata'])
    checkpoint_variant(selection)
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
                name, _ = line.split(' @ ')
                version = old['packages'][name]
            versions[name] = version
    return dict(schema='ds41f.installed-admission.v3', profile='mtp', glm=glm,
                payloads=payloads, files={path: None for path in support if '/' not in path},
                dependencies=versions, semantic_native={},
                tokenizer_sha256='81f64d1248a68ce3663e07ab3ee48b851e5df0e32d27cb98e4c9a268151e8d99',
                **selection,
                provenance=dict(source_commit=subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    overlays=list(OVERLAYS),
                    native_build=old['build'], recipe_source=old['recipe_source']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--profile', choices=['mtp'], default='mtp',
                   help='MTP only; Standard-OFF is retired')
    p.add_argument('--checkpoint-variant', choices=VARIANTS, default=DEFAULT_VARIANT)
    p.add_argument('--glm-profile', choices=['native', 'portable'])
    args = p.parse_args()
    from ds41f_mlx.glm_profile import default_profile
    args.glm_profile = args.glm_profile or default_profile('mtp')
    old = json.loads((ROOT/'third_party/mtp/normal-local.json').read_text())
    wheel = build_wheel(mtp_manifest(old, args.glm_profile, args.checkpoint_variant), args.output)
    print(json.dumps(dict(wheel=str(wheel), sha256=sha(wheel), checkpoint_variant=args.checkpoint_variant)))


if __name__ == '__main__':
    main()
