"""Installed executable admission. Trusted package publisher, non-hostile operator.

No checkout, build directory, source export or local observation seal is authority.
Host Mach-O dependencies use dyld's ABI contract; reachable image math is exact.
"""
import importlib.util
import json
import marshal
from pathlib import Path
import platform
import re
import subprocess
import sys
import sysconfig

MANIFEST = Path(__file__).with_name('execution_manifest.json')


def inventory(root):
    from .mtp_setup import sha
    return {str(p.relative_to(root)): sha(p) for p in sorted(Path(root).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts
            and p.name not in ('.DS_Store', 'execution_manifest.json') and p.suffix != '.pyc'}


def normalized(code):
    import types
    return code.replace(co_filename='', co_consts=tuple(
        normalized(v) if isinstance(v, types.CodeType) else v for v in code.co_consts))


def verify_tree(root, expected, label):
    if inventory(root) != expected:
        raise ValueError(f'executable payload drift: {label}')
    # Python may have imported timestamp-valid but substituted cached bytecode.
    for name in expected:
        if name.endswith('.py'):
            path = Path(root)/name
            cached = Path(importlib.util.cache_from_source(str(path)))
            if cached.exists():
                if normalized(marshal.loads(cached.read_bytes()[16:])) != normalized(
                        compile(path.read_bytes(), str(path), 'exec', optimize=sys.flags.optimize)):
                    raise ValueError(f'executable bytecode drift: {label}/{name}')


def links(path):
    """Resolve this profile's Homebrew install names, not a guessed file inventory."""
    path = Path(path).resolve()
    ids = subprocess.check_output(['otool', '-D', str(path)], text=True).splitlines()[1:]
    output = subprocess.check_output(['otool', '-L', str(path)], text=True)
    for line in output.splitlines()[1:]:
        match = re.match(r'\s*(.*?) \(compatibility version ([\d.]+), current version ([\d.]+)\)', line)
        if not match:
            raise ValueError('unsupported Mach-O dependency')
        name, compatibility, current = match.groups()
        if name in ids or name.startswith(('/usr/lib/', '/System/Library/')):
            continue  # dyld shared cache / OS framework contract
        if name.startswith('/opt/homebrew/'):
            target = Path(name)
        elif name.startswith(('@rpath/', '@loader_path/')):
            target = path.parent/Path(name).name
        else:
            raise ValueError(f'unsupported native search path: {name}')
        if not target.exists():
            raise ValueError(f'unresolved native dependency: {name}')
        yield target.resolve(), compatibility


def version(value):
    return tuple(int(v) for v in value.split('.'))


def host_closure(native):
    """Enforce each consumer's Mach-O ABI version, with actual dyld load too.

    Successful RTLD_NOW import is required by the caller. This scan rejects
    missing/redirected non-Homebrew search paths and ABI version regressions.
    No transitive host bytes enter the execution identity digest.
    """
    seen = set()
    def visit(path):
        path = Path(path).resolve()
        if path in seen:
            return
        seen.add(path)
        for target, minimum in links(path):
            own = subprocess.check_output(['otool', '-L', str(target)], text=True).splitlines()[1]
            provided = re.search(r'compatibility version ([\d.]+)', own).group(1)
            if version(provided) < version(minimum):
                raise ValueError(f'incompatible native ABI: {target.name}')
            visit(target)
    visit(native)
    return sorted(str(p) for p in seen)


def inspect_payload():
    from .mtp_setup import sha
    authority = json.loads(MANIFEST.read_text())
    site = Path(sysconfig.get_paths()['purelib']).resolve()
    if Path(__file__).resolve().parent != site/'ds41f_mlx':
        raise ValueError('production requires installed ds41f package; checkout execution refused')
    if sys.flags.optimize:
        raise ValueError('optimized Python unavailable in admitted profile')
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise ValueError('Apple Silicon macOS required')
    if sys.version_info[:2] != (3, 13) or sys.version_info[:3] < (3, 13, 15):
        raise ValueError('Python 3.13 >= 3.13.15 required')
    if not 26 <= int(platform.mac_ver()[0].split('.')[0]) < 27:
        raise ValueError('supported macOS 26 ABI family required')
    for name, digest in authority['files'].items():
        spec = importlib.util.find_spec(Path(name).stem)
        if spec is None or Path(spec.origin).resolve() != site/name or sha(site/name) != digest:
            raise ValueError(f'executable support file drift: {name}')
        cached = Path(importlib.util.cache_from_source(str(site/name)))
        if cached.exists() and normalized(marshal.loads(cached.read_bytes()[16:])) != normalized(
                compile((site/name).read_bytes(), str(site/name), 'exec')):
            raise ValueError(f'executable support bytecode drift: {name}')
    for name, expected in authority['payloads'].items():
        spec = importlib.util.find_spec(name)
        roots = list(spec.submodule_search_locations or []) if spec else []
        if roots != [str(site/name)]:
            raise ValueError(f'non-installed executable origin: {name}')
        verify_tree(site/name, expected, name)
    import deepseek_recipe
    from deepseek_recipe import _native
    native = Path(_native.__file__).resolve()
    expected_native = site/'deepseek_recipe'/native.name
    if native != expected_native or sha(native) != authority['payloads']['deepseek_recipe'].get(native.name):
        raise ValueError('redirected/unadmitted recipe extension')
    if authority['profile'] == 'mtp' and not all(hasattr(deepseek_recipe.StreamProcessor, name) for name in
               ('preview_tokens', 'semantic_snapshot', 'preview_eof_tokens',
                'preview_certified_eof_tokens', 'semantic_terminal')):
        raise ValueError('recipe semantic capabilities missing')
    closure = host_closure(native)
    by_name = {Path(p).name: Path(p) for p in closure}
    if len(by_name) != len(closure):
        raise ValueError('ambiguous native image names')
    from .runtime.resource_admission import check_loaded_image, loaded_images
    loaded = set(loaded_images())
    if not set(by_name.values()).issubset(loaded):
        raise ValueError('resolved native graph differs from loaded images')
    if any(p.name in by_name and p != by_name[p.name]
           and not str(p).startswith(('/usr/lib/', '/System/Library/')) for p in loaded):
        raise ValueError('mixed/redirected native images')
    check_loaded_image(native)
    for name, digest in authority['semantic_native'].items():
        path = by_name.get(name)
        if path is None or sha(path) != digest:
            raise ValueError(f'semantic native drift: {name}')
        check_loaded_image(path)
    tokenizer = Path(sys.prefix)/'share/ds41f-mtp/recipe/static/tokenizers/v41/tokenizer.json'
    if sha(tokenizer) != authority['tokenizer_sha256']:
        raise ValueError('recipe tokenizer drift')
    return authority, dict(status='PASS', python=platform.python_version(),
                           macos=platform.mac_ver()[0], native_images=closure)
