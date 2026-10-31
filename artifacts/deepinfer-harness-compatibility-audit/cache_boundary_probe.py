"""Isolated source probe: execute only DeepInfer's cache-strip method via AST.

No DeepInfer/MLX/Torch import, worker, network, installation or model execution.
Usage: python cache_boundary_probe.py /path/to/exact/DeepInfer
"""
import ast
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

root = Path(sys.argv[1])
revision = subprocess.check_output(
    ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
assert revision == '553741557a57df44a46fb3ab72b6aafeaf945337'
source = root / 'deepinfer/ve/multimodal_processor.py'
tree = ast.parse(source.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MultimodalProcessor')
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
              and n.name == '_compute_and_strip_cached_images')

class ImageInput:
    def __init__(self, binary):
        self.binary = binary

env = {'ImageInput': ImageInput}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), env)
results = []
for frontier, expected in [(0, 0), (2, 0), (6, 0), (7, 1), (10, 1), (12, 2)]:
    images = [ImageInput(b'A'), ImageInput(b'B')]
    data = {'segment_input': SimpleNamespace(
        meta={'image_spans': [(2, 5), (9, 3)]}, segments=images)}
    env[method.name](None, data, frontier)
    stripped = sum(image.binary is None for image in images)
    assert stripped == expected
    assert data.get('_num_cached_images', 0) == expected
    results.append({'frontier': frontier, 'stripped_images': stripped,
                    'remaining_payloads': [None if i.binary is None else i.binary.decode()
                                           for i in images]})
print(json.dumps({'revision': revision, 'scope': 'AST-extracted method; lightweight input doubles',
                  'result': 'PASS', 'cases': results}, indent=2))
