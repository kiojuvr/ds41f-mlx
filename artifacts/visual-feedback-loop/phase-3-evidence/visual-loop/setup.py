"""Build a fresh official headless profile; no user profile/session is copied."""
import hashlib
import json
import os
from pathlib import Path
import time
import tempfile

ROOT = Path('/Volumes/SDXC-512/ds41f-mlx')
TEMP = Path(tempfile.mkdtemp(prefix='ds41f-visual-loop-'))
OUT = Path(__file__).resolve().parent
(OUT/'location.json').write_text(json.dumps(dict(temp=str(TEMP))))
INSTALL = Path.home()/'.local/share/deepseek-harness/node_modules/@deepseek-ai'
workspace = TEMP/'workspace'
for p in [workspace, TEMP/'agents', TEMP/'runtime', TEMP/'home/profiles/headless']:
    p.mkdir(parents=True, exist_ok=True)
(workspace/'.render-handoff').mkdir()
profile = TEMP/'home/profiles/headless'
(profile/'package.json').write_text(json.dumps(dict(name='ds41f-phase3-headless', private=True, dependencies={}, dsh=dict(profile=dict(bundles=['@deepseek-ai/dsh-base','@deepseek-ai/dsh-headless'])))))
(profile/'cordis.yml').write_text('[]\n')
patch = (ROOT/'artifacts/visual-feedback-loop/phase-0-evidence/profile.patch.yml').read_text().replace('http://127.0.0.1:8000','http://127.0.0.1:18083').replace('maxTokens: 512','maxTokens: 2600').replace('streamIdleTimeoutMs: 300000','streamIdleTimeoutMs: 600000').replace('read: false','read: true').replace('write: false','write: true').replace('edit: false','edit: true').replace('mode: read-only','mode: workspace-write').replace('workspaceRoot: /tmp/ds41f-phase0-harness.oLw1RR/workspace','workspaceRoot: '+str(workspace)).replace('defaultPreset: read-only','defaultPreset: bounded')
patch = patch.replace('      read-only:\n        sandbox: read-only\n        approval: ask','      bounded:\n        sandbox: workspace-write\n        approval: never')
patch = patch.replace('- id: tool-bash\n  disabled: true','- id: tool-bash\n  disabled: false\n  config:\n    enableRunInBackground: false\n    promoteOnTimeout: false')
patch += '''- id: bash-sandbox
  config:
    timeoutMs: 25000
    maxTimeoutMs: 25000
    maxOutputBytes: 4096
- id: approval
  config:
    policy: never
- id: agent-loop
  config:
    maxParallelToolCalls: 1
- insert:
    - id: phase3-bounded-policy
      name: /Volumes/SDXC-512/ds41f-mlx/tools/phase3_harness_guard.mjs
'''
(profile/'cordis.patch.yml').write_text(patch)
(OUT/'profile.patch.yml').write_text(patch)
command = '/tmp/ds41f-phase2-renderer/bin/python /Volumes/SDXC-512/ds41f-mlx/tools/phase3_render_handoff.py request'
policy = dict(command=command, tools=30, deadline_ms=int((time.time()+5400)*1000), audit=str(OUT/'tool-policy.jsonl'))
(TEMP/'policy.json').write_text(json.dumps(policy))
(OUT/'budgets.json').write_text(json.dumps(dict(total_s=5400, startup_s=1200, generation_idle_s=600, generation_wall_s=600, requests=24, max_tokens=2600, thinking='disabled', sampling='Official Harness/native defaults; not the Phase 2 temperature-0 driver', render_s=25, max_correction_iterations=5, max_renders=6, tools=30, note='Official render and read_image are separate calls; 30 permits bounded filesystem reads/edits as well as 18 basic effects.', source_bytes=65536, body_limit_bytes=16777216, policy=policy), indent=2)+'\n')
phase2 = json.loads((ROOT/'artifacts/visual-feedback-loop/phase-2-evidence/continuation-1/model-run/task.json').read_text())
goal = phase2['goal'].replace('saving the starter below using write_html, then render it.', 'saving the starter below to index.html using write, then render it with bash and inspect it using read_image.').replace('Do not call write_html and render in the same response.', 'Do not write/edit and render in the same response. Use one tool call per response. To render, call bash with exactly this command: '+command+'. Use the session workspace as workdir, no background execution or escalation. Then call read_image on the returned screenshot path. Use write or edit only for index.html; read only index.html if needed. The renderer does not return image content itself; always use read_image to observe it.')
(OUT/'task.json').write_text(json.dumps(dict(goal=goal, starter=phase2['starter'], rubric=phase2['rubric'], original_task_sha256=phase2['task_sha256'], task_sha256=hashlib.sha256((goal+phase2['starter']).encode()).hexdigest()), indent=2)+'\n')
(TEMP/'prompt.txt').write_text(goal+phase2['starter'])
print(TEMP)
(OUT/'identities.json').write_text(json.dumps(dict(node=os.popen('node --version').read().strip(), packages={n:dict(version=json.loads((INSTALL/n/'package.json').read_text())['version'], implementation_sha256=hashlib.sha256((INSTALL/n/('lib/bin.js' if n=='dsh' else 'lib/index.js')).read_bytes()).hexdigest()) for n in ['dsh','dsh-base','dsh-headless','dsh-llm-deepseek','dsh-llm-deepseek-api-key','dsh-tool-fs','dsh-attachment','dsh-attachment-local','dsh-tools','dsh-tool-bash','dsh-bash-sandbox','dsh-bash-local','dsh-sandbox-local','dsh-sandbox-policy','dsh-fs-sandbox','dsh-user-approval','dsh-permission-presets']}), indent=2)+'\n')
