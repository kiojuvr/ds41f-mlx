"""External retention after failed sandbox preflight; no renderer/model retry."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/Volumes/SDXC-512/ds41f-mlx')
OUT = ROOT/'artifacts/visual-feedback-loop/phase-3-evidence/continuation-1'
setup = json.loads((OUT/'preflight-setup.json').read_text())
temp = Path(setup['temporary_root'])
assert temp.name.startswith('ds41f-phase3-continuation-1-')
workspace = temp/'workspace'
manifest = []
for source in sorted(workspace.rglob('*')):
    if not source.is_file():
        continue
    assert not source.is_symlink()
    relative = source.relative_to(workspace)
    target = OUT/'preflight-workspace'/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(source, target)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
    manifest.append(dict(path=str(relative), bytes=source.stat().st_size, sha256=digest))
assert any(r['path']=='phase3-stage/artifact-00.html' for r in manifest)
assert not (OUT/'denial-probe.txt').exists()
assert not list(workspace.glob('render-*.png'))
assert not list((workspace/'phase3-stage').glob('render-*.json'))
processes = subprocess.check_output(['ps','-axo','pid,ppid,command'],text=True)
# Search executable invocations, not this supervisor or unrelated remote desktop.
owned = []
for line in processes.splitlines():
    fields = line.strip().split(None, 2)
    if len(fields) != 3:
        continue
    command = fields[2]
    if command.startswith('/bin/bash -c'):
        continue  # supervising shell can contain source-code search strings
    if any(s in command for s in [
        '/Applications/Google Chrome.app/', 'playwright/driver/node',
        '/bin/python /Volumes/SDXC-512/ds41f-mlx/tools/phase3_render.py',
        '/bin/python /Volumes/SDXC-512/ds41f-mlx/tools/visual_feedback_renderer.py'
    ]):
        owned.append(line)
assert not owned, owned
(OUT/'retention-manifest.json').write_text(json.dumps(dict(
    files=manifest, all_archive_hashes_verified_before_removal=True,
    scope='sandbox preflight only; no real-model evidence'),indent=2)+'\n')
shutil.rmtree(temp)
(OUT/'cleanup.json').write_text(json.dumps(dict(
    temporary_root=str(temp), removed=not temp.exists(), renderer_browser_processes=owned,
    runtime_started=False, model_loaded=False, generation_requests=0,
    official_conversations=0, screenshots=0, screenshot_guided_corrections=0,
    repository_write_probe_denied=True),indent=2)+'\n')
print(json.dumps(dict(archived_files=len(manifest),hashes_verified=True,temporary_root_removed=True)))
