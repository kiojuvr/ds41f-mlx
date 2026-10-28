"""Read-only reuse of Step 3B identities and original Runtime inventory; no Runtime import."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts/multimodal-runtime/phase4-integrated-evidence'
prior = ROOT / 'artifacts/multimodal-runtime/phase3-step3b-evidence'
identity = json.loads((prior / 'installed-identities.json').read_text())
install = Path.home() / '.local/share/deepseek-harness/node_modules/@deepseek-ai'
packages = {}
for name, record in identity['packages'].items():
    for relative, expected in record['files'].items():
        assert hashlib.sha256((install / name / relative).read_bytes()).hexdigest() == expected, (name, relative)
    packages[name] = dict(version=record['version'], verified_files=len(record['files']))
baseline_path = ROOT / 'artifacts/multimodal-runtime/phase2-step2br2-original-before.json'
baseline = json.loads(baseline_path.read_text())['files']
operational = Path('/Volumes/SDXC-512/ds41f-1017')
for name, expected in baseline.items():
    path = operational / name
    stat = path.stat()
    actual = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size=stat.st_size,
                  mtime_ns=stat.st_mtime_ns, inode=stat.st_ino)
    assert actual == expected, name
result = dict(official_packages_match_step3b=True, packages=packages,
              reused_identity_receipt='phase3-step3b-evidence/installed-identities.json',
              identity_receipt_sha256=hashlib.sha256((prior / 'installed-identities.json').read_bytes()).hexdigest(),
              original_runtime_paths=len(baseline), original_runtime_matches=True,
              baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
              weights_loaded=0, production_changes=False)
(OUT / ('integrity-' + sys.argv[1] + '.json')).write_text(json.dumps(result, indent=2) + '\n')
print('PASS: installed Step 3B package identities; 817 original Runtime hash/size/mtime/inode records')
