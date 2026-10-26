"""Isolated installed zero-weight gates; no test numerical binding is reused."""
import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT/'artifacts/multimodal-runtime/phase2-step2dr1-worker.py.txt'


@pytest.mark.parametrize('early', [False, True])
def test_actual_installed_transformers_import_order(early):
    code = f'''
import sys, json
sys.path.append({str(ROOT)!r})
from tools.qualify_transformers_import_order import characterize
print(json.dumps(characterize(early={early!r}), default=str))
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code],
        text=True, capture_output=True, check=True)
    proof = json.loads(result.stdout)
    assert proof['before']['locations'][0]['type'] == 'str'
    assert proof['after']['locations'][0]['type'] == 'PosixPath'
    assert proof['before']['resolved_locations'] == proof['after']['resolved_locations']
    assert proof['after']['actual_directory'] == proof['after']['installed_directory']
    if early:
        assert proof['admission']['actual_pre_woa_materialized']
        assert proof['preserved']['actual_post_import_pre_woa']
        assert proof['preserved']['identity_object'] and proof['preserved']['function_object']
    else:
        assert proof['late_error'] == 'non-installed executable origin: transformers'


def test_real_worker_admits_before_serving_and_stays_zero_weight(tmp_path):
    source = WORKER.read_text()
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'prove_installed_admission']
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
               and (n.module or '').startswith('ds41f_mlx.serving')]
    assert len(calls) == 1 and calls[0].lineno < min(n.lineno for n in imports)
    # Redirect diagnostic receipt output only; never overwrite retained model
    # evidence or change any installed module/import/admission behavior.
    worker = tmp_path/'worker.py'
    destination = "OUT = ROOT/'artifacts/multimodal-runtime'"
    assert source.count(destination) == 1
    worker.write_text(source.replace(destination, f'OUT = Path({str(tmp_path)!r})'))
    result = subprocess.run([sys.executable, '-I', '-B', str(worker), '--preflight-only'],
        text=True, capture_output=True, check=True)
    assert 'zero loads' in result.stdout and 'sole model load' not in result.stdout
    report = json.loads((tmp_path/'phase2-step2dr1-preflight.json').read_text())
    preserved = report['preserved_admission']
    assert preserved['all_authenticated_modules'] and preserved['identity_object']
    assert preserved['actual_pre_woa_materialized']
    assert preserved['profile'] == 'flash-1017-pre-woa-fp8-v1'
    for origin in preserved['origins'].values():
        assert 'site-packages' in origin['path']
        assert not origin['path'].startswith(str(ROOT))
