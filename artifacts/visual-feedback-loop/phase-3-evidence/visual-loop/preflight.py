"""One bounded real sandbox/host handoff, no model or Harness conversation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path('/Volumes/SDXC-512/ds41f-mlx')
OUT = Path(__file__).resolve().parent
PREF = OUT/'preflight'
PREF.mkdir(exist_ok=False)
TEMP = Path(tempfile.mkdtemp(prefix='ds41f-handoff-preflight-'))
workspace = TEMP/'workspace'
workspace.mkdir()
(workspace/'.render-handoff').mkdir()
task = json.loads((OUT.parent/'task.json').read_text())
(workspace/'index.html').write_text(task['starter'])
env = dict(os.environ, PHASE3_WORKSPACE=str(workspace), PHASE3_PREFLIGHT_RESULT=str(PREF/'sandbox.json'), PYTHONDONTWRITEBYTECODE='1')
stop = TEMP/'stop'
p = None
try:
    p = subprocess.Popen(['/tmp/ds41f-phase2-renderer/bin/python',str(ROOT/'tools/phase3_render_handoff.py'),'supervise','--workspace',str(workspace),'--evidence',str(PREF),'--stop',str(stop),'--timeout','40'], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    run = subprocess.run(['node',str(OUT/'preflight.mjs')],env=env,capture_output=True,text=True,timeout=60)
    (PREF/'launcher.json').write_text(json.dumps(dict(returncode=run.returncode,stdout=run.stdout,stderr=run.stderr),indent=2))
    assert run.returncode == 0, 'supported sandbox preflight failed; see sandbox.json'
    result = json.loads((PREF/'sandbox.json').read_text())
    completion = json.loads((PREF/'completion-00.json').read_text())
    source_hash = hashlib.sha256((workspace/'index.html').read_bytes()).hexdigest()
    image_hash = hashlib.sha256((workspace/'render-00.png').read_bytes()).hexdigest()
    assert source_hash == completion['source_sha256'] and image_hash == completion['image_sha256']
    from PIL import Image
    with Image.open(workspace/'render-00.png') as im:
        assert im.size == (1280,720) and im.format == 'PNG'
    assert not (OUT/'denial-probe.txt').exists()
    (PREF/'result.json').write_text(json.dumps(dict(status='PASS',source_sha256=source_hash,image_sha256=image_hash,dimensions=[1280,720],confined_request=True,host_renderer=True,repository_write_denied=True,workspace_screenshot_readable=True),indent=2))
finally:
    stop.touch()
    if p:
        stdout,stderr = p.communicate(timeout=30)
        (PREF/'supervisor.json').write_text(json.dumps(dict(pid=p.pid,exit=p.returncode,stdout=stdout,stderr=stderr),indent=2))
    if workspace.exists():
        shutil.copytree(workspace,PREF/'workspace')
    shutil.rmtree(TEMP)
    (PREF/'cleanup.json').write_text(json.dumps(dict(temporary_root=str(TEMP),removed=not TEMP.exists(),supervisor_exit=p.returncode if p else None),indent=2))
