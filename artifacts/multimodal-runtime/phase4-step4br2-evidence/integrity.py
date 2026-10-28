"""Reuse frozen read-only package and 817-path operational integrity inspector."""
from pathlib import Path
ROOT=Path('/Volumes/SDXC-512/ds41f-mlx')
source=ROOT/'artifacts/multimodal-runtime/phase4-step4ar1-evidence/integrity.py'
text=source.read_text().replace("OUT = ROOT / 'artifacts/multimodal-runtime/phase4-step4ar1-evidence'", "OUT = ROOT / 'artifacts/multimodal-runtime/phase4-step4br2-evidence'")
exec(compile(text,str(source),'exec'),{'__file__':str(source),'__name__':'__main__'})
