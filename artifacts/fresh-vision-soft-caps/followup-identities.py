"""Read-only identity snapshot; run with system Python, no Production imports."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PROD = Path('/Volumes/SDXC-512/ds41f-1017/lib/python3.13/site-packages')
TEST = Path('/private/tmp/ds41f-step4ar1-tests/lib/python3.13/site-packages')
OLD = json.loads((ROOT / 'artifacts/fresh-vision-soft-caps/identities.json').read_text())

def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()

def hashes(root, names):
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}

def installed(root):
    manifest = json.loads((root/'ds41f_mlx/execution_manifest.json').read_text())
    names = list(OLD['production_sha256']) + [
        'mlx/core/__init__.pyi', 'mlx/core.cpython-313-darwin.so',
        'mlx/lib/libmlx.dylib', 'deepseek_recipe/_native.abi3.so',
        'omlx/patches/deepseek_v41/language.py']
    versions = {}
    for name in ('ds41f_mlx', 'deepseek_recipe', 'mlx', 'mlx_lm', 'omlx'):
        for path in root.glob(name+'-*.dist-info/METADATA'):
            versions[path.parent.name] = [s for s in path.read_text().splitlines()
                                         if s.startswith(('Name:', 'Version:'))]
    return dict(root=str(root), versions=versions, sha256=hashes(root, names),
                profile=manifest['profile'], numerics=manifest['numerics'],
                provenance=manifest['provenance'])

upstream = Path('/private/tmp/ds41f-fresh-vision-upstream')
up_hash = hashes(upstream, OLD['upstream']['source_sha256'])
prod = installed(PROD)
print(json.dumps(dict(head=git('rev-parse', 'HEAD'), branch=git('branch', '--show-current'),
    status=git('status', '--short'), remote=git('remote', 'get-url', 'origin'),
    production=prod, isolated_tests=installed(TEST),
    production_matches_prior={n: prod['sha256'][n] == h for n,h in OLD['production_sha256'].items()},
    source_sha256=hashes(ROOT, [
        'ds41f_mlx/runtime/multimodal.py', 'ds41f_mlx/runtime/mtp_resources.py',
        'ds41f_mlx/serving/production_mtp.py', 'ds41f_mlx/serving/internal_mtp.py',
        'ds41f_mlx/serving/paired_checkpoint.py',
        'ds41f_mlx/prefill_fp8_mlx/p6_append.py']),
    upstream=dict(reviewed=OLD['upstream']['reviewed'], sha256=up_hash,
        matches_prior=up_hash == OLD['upstream']['source_sha256'])), indent=2))
