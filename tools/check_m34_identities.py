"""Fail closed on drift from the exact M33 candidate and preserved OFF runtime."""
import ast
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),'/tmp/ds41f-m33-omlx']
import deepseek_recipe._native as native
import omlx.api.tool_calling as tool_api
q=json.loads((ROOT/'artifacts/m33/runtime-identities.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(p,*args):return subprocess.check_output(['git','-C',str(p),*args],text=True).strip()
correction=json.loads((ROOT/'artifacts/m34/runtime-correction.json').read_text())
current=dict(q['ds41f_source_hashes']);current[correction['path']]=correction['m34_sha256']
def primitive_source(text):
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='canonical_quiesce_native_singleton')
    return ast.get_source_segment(text,node)
base_source=subprocess.check_output(['git','-C',str(ROOT),'show','2e47dff:ds41f_mlx/runtime/mtp_lifecycle.py'],text=True)
checks={
 'performance_primitive_unchanged':primitive_source(base_source)==primitive_source((ROOT/correction['path']).read_text()),
 'narrow_correction':correction['m33_sha256']==q['ds41f_source_hashes'][correction['path']] and sha(ROOT/correction['patch'])==correction['patch_sha256'],
 'native':sha(native.__file__)==q['recipe_native_sha256'],
 'checkpoint_config':sha(Path(q['checkpoint'])/'config.json')==q['checkpoint_config_sha256'],
 'tokenizer':sha('/tmp/ds41f-m32-recipe/static/tokenizers/v41/tokenizer.json')==q['tokenizer_sha256'],
 'recipe_commit':git('/tmp/ds41f-m32-recipe','rev-parse','HEAD')==q['recipe_candidate'],
 'recipe_clean':not git('/tmp/ds41f-m32-recipe','status','--porcelain'),
 'omlx_commit':git(q['omlx_path'],'rev-parse','HEAD')==q['omlx_candidate'],
 'omlx_clean':not git(q['omlx_path'],'status','--porcelain'),
 'actual_import':str(Path(q['omlx_path'])) in tool_api.__file__,
 'full_native_build':os.environ.get('DOCS_RS') is None,
 'packages':all(md.version(n)==v for n,v in q['packages'].items()),
 'runtime_sources':all(sha(ROOT/p)==current[p] for p in ['ds41f_mlx/runtime/recipe_semantic_guard.py','ds41f_mlx/runtime/mtp_lifecycle.py','ds41f_mlx/prefill_fp8_mlx/handoff.py']),
 'candidate_sources':all(sha(Path(q['omlx_path'])/p)==v for p,v in q['omlx_source_hashes'].items()),
 'release_native':all(sha(a['path'])==a['sha256'] for a in q['preserved_release_cpp_artifacts']),
 'release_recipe_native':sha(Path.home()/'.venvs/omlx-0.7.0.release/lib/python3.13/site-packages/deepseek_recipe/_native.abi3.so')==q['preserved_release_native_sha256'],
 'historical_m25_m33_unchanged':not git(ROOT,'diff','2e47dff','--',*[f'artifacts/m{n}' for n in range(25,34)]),
}
result=dict(schema='ds41f.m34.identities.v1',status='PASS' if all(checks.values()) else 'FAIL',checks=checks,python=sys.version,executable=sys.executable,platform=platform.platform(),native_path=native.__file__,tool_api=tool_api.__file__,packages={n:md.version(n) for n in q['packages']},m33_identity_sha256=sha(ROOT/'artifacts/m33/runtime-identities.json'),current_runtime_sources=current,runtime_correction=correction)
(ROOT/'artifacts/m34/identities.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2));assert all(checks.values())
