"""Actual installed numerics, never the source-test numerical binding fixture."""
import json
from pathlib import Path
import subprocess
import sys


def test_isolated_installed_diagnostic_numerics():
    root = Path(__file__).resolve().parents[1]
    code = f'''
import sys, json
sys.path.append({str(root)!r})
from tools.qualify_installed_admission import prove_installed_admission
proof = prove_installed_admission()
assert proof['identity_object_preserved']
assert proof['actual_pre_woa_materialized']
print(json.dumps(proof, default=str))
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code],
                            text=True, capture_output=True, check=True)
    proof = json.loads(result.stdout)
    assert proof['profile'] == 'flash-1017-pre-woa-fp8-v1'
    assert 'site-packages' in proof['origins']['ds41f_mlx.execution_identity']['path']


def test_pre_woa_refuses_unadmitted_fresh_process():
    # This is the previous fresh-module failure without purging or mixing a
    # graph: installed activation must refuse before its own process admission.
    code = '''
import mlx.core as mx
from ds41f_mlx.model_execution.activation import pre_woa
try:
    mx.eval(pre_woa(mx.ones((1, 1, 1, 32), mx.bfloat16)))
except RuntimeError as error:
    assert str(error) == 'numerical execution requires installed MTP admission'
else:
    raise AssertionError('unadmitted numerical execution accepted')
'''
    subprocess.run([sys.executable, '-I', '-B', '-c', code], check=True)
