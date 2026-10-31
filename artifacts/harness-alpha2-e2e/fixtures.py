"""Initialize ONLY a fresh disposable acceptance profile, before booting Harness."""
from pathlib import Path
import shutil

root=Path('/tmp/ds41f-alpha2-e2e')
if any((root/'home/sessions').rglob('session.v4.jsonl.zstd')):
    raise SystemExit('Refusing to reset fixtures for an existing Session; use a fresh isolated run.')
workspace=root/'workspace'; workspace.mkdir(parents=True,exist_ok=True)
for filename,text in [('baseline.txt','BOOT-438'),('late.txt','LATE-927')]:
    (workspace/filename).write_text(text+'\n')
for label in ('A','B'):
    shutil.copy2(Path(__file__).parent/f'image-{label}.png',workspace/f'image-{label}.png')
skill=root/'home/skills/receipt';skill.mkdir(parents=True,exist_ok=True)
(skill/'SKILL.md').write_text('---\nname: receipt\ndescription: Produce the alpha2 acceptance Skill receipt.\n---\nWhen invoked, say SKILL-ALPHA-614 briefly. Do not invoke any tools for this skill.\n')
(root/'registry-stage').write_text('baseline')
# Before client action 18, rename receipt -> receipt-next and replace SKILL.md:
# ---
# name: receipt-next
# description: Replacement receipt skill; use this for the next alpha2 acceptance receipt.
# ---
# When invoked, say SKILL-BETA-827 briefly. Do not invoke other tools for this skill.
