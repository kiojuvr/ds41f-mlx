"""Record incremental native recipe patch and exact source/build provenance.

Run only against the operator's build tree after building/installing the wheel.
The pre-edit copies are used to certify the previous admitted recipe source.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recipe', type=Path, default=Path('/Volumes/SDXC-512/ds41f-mtp-investigation-build/recipe'))
    ap.add_argument('--before', type=Path, default=Path('/tmp/m54-recipe-before'))
    ap.add_argument('--out', type=Path, default=Path('artifacts/m54-eof'))
    args = ap.parse_args()
    files = {
        'deepseek-recipe/src/stream/processor.rs': 'processor.rs',
        'deepseek-recipe/src/stream/semantic.rs': 'semantic.rs',
        'deepseek-recipe/src/stream/mod.rs': 'mod.rs',
        'deepseek-recipe-python/src/response.rs': 'response.rs',
        'deepseek-recipe/src/protocol/openai/chat_completion/response/chunk_generator.rs': None,
        'deepseek-recipe/src/protocol/openai/chat_completion/response/schema.rs': None,
    }
    patch = []
    inventory = {}
    sha = lambda b: hashlib.sha256(b).hexdigest()
    for name, original in files.items():
        after = (args.recipe/name).read_text()
        if original:
            before = (args.before/original).read_text()
        else:
            # Only Clone derives were added to these two files. Existing Clone
            # on the finish enum is untouched by this exact inverse.
            before = after.replace('#[derive(Debug, Clone, Serialize)]', '#[derive(Debug, Serialize)]').replace('#[derive(Debug, Clone)]', '#[derive(Debug)]')
        inventory[name] = dict(base_sha256=sha(before.encode()), modified_sha256=sha(after.encode()))
        patch.extend(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile='a/'+name, tofile='b/'+name))
    args.out.mkdir(parents=True, exist_ok=True)
    data = ''.join(patch).encode()
    (args.out/'recipe-consuming-eof-preview.patch').write_bytes(data)
    import deepseek_recipe._native as native
    import deepseek_recipe
    binary = Path(native.__file__)
    native_data = binary.read_bytes()
    wheel = args.recipe.parent/'wheels/deepseek_recipe-0.1.1-cp310-abi3-macosx_11_0_arm64.whl'
    record = dict(schema='ds41f.m54.native-consuming-eof-preview.v1',
        patch_sha256=sha(data), sources=inventory,
        base='previous M32 native semantic-session recipe in current admitted build tree',
        application='patch -p1; verify base file hashes first; no parser replacement',
        native=dict(path=str(binary), sha256=sha(native_data), size=len(native_data)),
        python=dict(path=deepseek_recipe.__file__,sha256=sha(Path(deepseek_recipe.__file__).read_bytes())),
        wheel=dict(path=str(wheel),sha256=sha(wheel.read_bytes())),
        cargo_lock_sha256=sha((args.recipe/'Cargo.lock').read_bytes()),
        rustc=subprocess.check_output(['rustc','--version'],text=True).strip(),
        build='PKG_CONFIG_PATH=/opt/homebrew/opt/opencv@4/lib/pkgconfig CARGO_TARGET_DIR=/Volumes/SDXC-512/ds41f-mtp-investigation-build/cargo .venv/bin/maturin build --release --offline --skip-auditwheel --manifest-path <recipe>/deepseek-recipe-python/Cargo.toml -o <wheels>',
        scope='development native binding only; no release/packaging/runtime promotion')
    (args.out/'native-provenance.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__ == '__main__': main()
